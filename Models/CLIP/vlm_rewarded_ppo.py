import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple, TypeVar, Type, Union, Dict, Any
from tqdm import tqdm
import numpy as np
import torch
import torch as th
from box import Box
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import safe_mean, check_for_correct_spaces, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from gymnasium import spaces
# The buffer is still useful for storing render_arrays for the VLM
from Models.CLIP.vlm_buffer import VLMRolloutBuffer
from Models.vlm_weights import VLMScorer

SelfVLMRewardedPPO = TypeVar("SelfVLMRewardedPPO", bound="VLMRewardedPPO")


class VLMRewardedPPO(PPO):
    """
    A PPO agent that uses a Vision-Language Model (VLM) to dynamically
    weight reward components after each rollout.
    """
    rollout_buffer: VLMRolloutBuffer

    def __init__(
            self,
            *,
            env: VecEnv,
            config: Box,
            inference_only: bool = False,
    ):
        self.config = config
        self.vlm_scorer = None

        super().__init__(
            env=env,
            policy='MultiInputPolicy',
            seed=config.seed,
            **self.config.algorithm_params,
        )

        self.inference_only = inference_only
        if not self.inference_only:
            self._setup_model()
            self._load_modules()

    def _setup_model(self):
        super()._setup_model()
        # This buffer is still needed to store the frames for the VLM
        self.rollout_buffer = VLMRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def _load_modules(self):
        """
        Loads the VLM Scorer. The CLIP model is no longer needed.
        """
        print("Initializing Quantized VLM Scorer...")
        vlm_init_params = {
            "model_name": self.config.vlm_params.model_name,
            "batch_size": self.config.vlm_params.batch_size,
            "max_new_tokens": self.config.vlm_params.max_new_tokens,
        }
        self.vlm_scorer = VLMScorer(device=self.device, **vlm_init_params)
        print("VLM Scorer initialized successfully.")

    def _compute_vlm_rewards(self) -> None:
        """
        This method is called at the end of each rollout. It uses the VLM to
        calculate weights and applies them to the base rewards from the environment.
        The final reward is the VLM-weighted base reward.
        """
        assert self.env is not None

        buffer = self.rollout_buffer
        if buffer.pos == 0: return
        
        frames_raw = buffer.render_arrays[:buffer.pos]
        speeds_raw = buffer.speeds[:buffer.pos]
        infos_raw = buffer.infos[:buffer.pos]

        # --- Step 1: Get VLM Weights for Each Timestep ---
        with torch.no_grad():
            weights_per_timestep = self.vlm_scorer.get_per_timestep_weights(
                frames_raw, speeds_raw, infos_raw,
                clip_size=self.config.vlm_params.clip_size,
                step_size=self.config.vlm_params.step_size
            )

        w1_batch = weights_per_timestep[:, 0].reshape(-1, 1)
        w2_batch = weights_per_timestep[:, 1].reshape(-1, 1)
        w3_batch = weights_per_timestep[:, 2].reshape(-1, 1)

        # --- Step 2: Calculate the Final Weighted Reward ---
        c1_batch = np.array([info[0].get("safety_reward", 0) for info in infos_raw]).reshape(-1, 1)
        c2_batch = np.array([info[0].get("progress_reward", 0) for info in infos_raw]).reshape(-1, 1)
        c3_batch = np.array([info[0].get("smoothness_reward", 0) for info in infos_raw]).reshape(-1, 1)
        col_batch = np.array([info[0].get("collision_penalty", 0) for info in infos_raw]).reshape(-1, 1)

        # The final reward is now ONLY the weighted base reward.
        final_rewards = (w1_batch * c1_batch) + (w2_batch * c2_batch) + (w3_batch * c3_batch) + col_batch
        
        # --- Step 3: Update Buffer and Info Dicts for Logging ---
        for i in range(buffer.pos):
            buffer.infos[i][0]['w1_safety'] = w1_batch[i][0]
            buffer.infos[i][0]['w2_comfort'] = w2_batch[i][0]
            buffer.infos[i][0]['w3_efficiency'] = w3_batch[i][0]
            buffer.infos[i][0]['total_reward'] = final_rewards[i][0]
        
        print("--- VLM Reward Calculation Summary ---")
        print(f"VLM Weights (first step): w1={w1_batch[0][0]:.2f}, w2={w2_batch[0][0]:.2f}, w3={w3_batch[0][0]:.2f}")
        
        buffer.clear_render_arrays()
        buffer.rewards[:buffer.pos] = final_rewards


    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: VLMRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
                
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                infos=infos,
                render_arrays=infos[0].get("render_arrays", new_obs),
                speeds=infos[0].get("speed_ms", 0.0),
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        if not self.inference_only:
            self._compute_vlm_rewards() # Use the new function name

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    # The rest of the file (log, train, save, load etc.) remains the same
    def _log(self) -> None:
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean_base", # This logs the unweighted base reward mean
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success_rate", safe_mean(self.ep_success_buffer)
            )
        self.logger.dump(step=self.num_timesteps)

    def train(self) -> None:
        self._log()
        super().train()

    def _setup_learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
            *args,
    ) -> Tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            *args,
        )
        if not hasattr(self, "ep_info_buffer") or self.ep_info_buffer is None or reset_num_timesteps:
            self.ep_info_buffer = deque(maxlen=100)
        return total_timesteps, callback

    def learn(self: SelfVLMRewardedPPO, *args, **kwargs) -> SelfVLMRewardedPPO:
        assert not self.inference_only
        return super().learn(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        # Exclude the VLM model from being saved
        super().save(*args, exclude=["vlm_scorer"], **kwargs)

    @classmethod
    def load(
            cls: Type[SelfVLMRewardedPPO],
            path: Union[str, pathlib.Path],
            *,
            env: Optional[VecEnv] = None,
            device: Union[torch.device, str] = "cuda:0",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfVLMRewardedPPO:
        # Simplified load method, assuming the VLM will be re-initialized
        # This part might need adjustment based on how you save/load full models
        model = super().load(path, env=env, device=device, custom_objects=custom_objects, **kwargs)
        return model
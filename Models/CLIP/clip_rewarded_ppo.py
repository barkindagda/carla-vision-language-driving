import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple, TypeVar, Type, Union, Dict, Any

import numpy as np
import open_clip
from gymnasium import spaces
import torch
import torch as th
from box import Box
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean, check_for_correct_spaces, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from Models.CLIP.clip_buffer import CLIPReplayBuffer, CLIPRolloutBuffer
from Models.CLIP.clip_reward_model import compute_rewards, CLIPEmbed, CLIPReward
from Models.vlm_weights import VLMScorer  # Import the new VLMScorer class

SelfCLIPRewardedPPO = TypeVar("SelfCLIPRewardedPPO", bound="CLIPRewardedPPO")


class CLIPRewardedPPO(PPO):
    rollout_buffer: CLIPReplayBuffer

    def __init__(
            self,
            *,
            env: VecEnv,
            config: Box,
            inference_only: bool = False,
    ):
        self.config = config
        self.clip_preprocess = None  # Initialize clip_preprocess
        self.ep_clip_info_buffer = None  # type: Optional[deque]
        self.vlm_scorer = None # Initialize vlm_scorer

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

    def _dump_logs(self) -> None:
        pass

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = CLIPRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def _load_modules(self):
        # This part for the CLIP reward model remains the same
        model_name = self.config.clip_reward_params.pretrained_model
        pretrained = "openai"  # Default pretrained checkpoint for OpenCLIP
        clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        clip_model = clip_model.to(self.device)
        clip_model = CLIPEmbed(clip_model)
        target_prompts = open_clip.tokenize(self.config.clip_reward_params.target_prompts).to(self.device)
        baseline_prompts = open_clip.tokenize(self.config.clip_reward_params.baseline_prompts).to(self.device)
        self.reward_model = CLIPReward(
            model=clip_model,
            alpha=self.config.clip_reward_params.alpha,
            target_prompts=target_prompts,
            baseline_prompts=baseline_prompts,
        ).eval().to(self.device)

        # ADD THIS PART to load your VLM scorer
        print("Initializing VLM Scorer...")
        self.vlm_scorer = VLMScorer(device=self.device)
        print("VLM Scorer initialized successfully.")

    def _compute_clip_rewards(self) -> None:
        assert self.env is not None

        # --- 1. GATHER DATA FROM THE ROLLOUT BUFFER ---
        frames_raw = self.rollout_buffer.render_arrays
        if len(frames_raw) == 0:
            return

        speeds_raw = self.rollout_buffer.speeds
        infos_raw = self.rollout_buffer.infos

        # --- 2. CALCULATE SYNTHETIC REWARD (R_synthetic) ---
        frames_processed_clip = torch.stack([
            self.clip_preprocess(Image.fromarray(np.transpose(arr, (1, 2, 0)).astype(np.uint8)))
            for arr in frames_raw
        ]).to(self.device)

        r_synthetic = compute_rewards(
            model=self.reward_model,
            frames=frames_processed_clip,
            batch_size=self.config.clip_reward_params.batch_size,
        )
        r_synthetic = r_synthetic.numpy().reshape(-1, 1)
        r_synthetic = np.clip(r_synthetic, 0.0, 1.0)  # Normalize

        # --- 3. GET VLM WEIGHTS (w1, w2, w3) USING YOUR VLMScorer ---
        # Your scorer processes the entire rollout as a single segment.
        # We must shape the data into a batch of size 1.
        frames_batch = frames_raw[np.newaxis, ...]
        speeds_batch = speeds_raw.T[np.newaxis, ...]
        infos_batch = [infos_raw]  # List of lists

        # Call your VLM scorer
        vlm_scores = self.vlm_scorer.score_segment_batch(frames_batch, speeds_batch, infos_batch)
        
        # The scorer returns one set of weights for the whole segment.
        # We will apply these same weights to every step in the rollout.
        w1, w2, w3 = vlm_scores[0]  # e.g., [0.9, 0.8, 0.7]

        # --- 4. EXTRACT RAW REWARD COMPONENTS FROM BUFFER ---
        c1_batch = np.array([info[0].get("safety_reward", 0) for info in infos_raw]).reshape(-1, 1)
        c2_batch = np.array([info[0].get("progress_reward", 0) for info in infos_raw]).reshape(-1, 1)
        c3_batch = np.array([info[0].get("smoothness_reward", 0) for info in infos_raw]).reshape(-1, 1)
        col_batch = np.array([info[0].get("collision_penalty", 0) for info in infos_raw]).reshape(-1, 1)

        # --- 5. CALCULATE THE FINAL REWARD ---
        # Apply the single set of VLM weights to all timesteps
        weighted_base_rewards = (w1 * c1_batch) + (w2 * c2_batch) + (w3 * c3_batch) + col_batch
        
        p = self.config.clip_reward_params.get('p', 0.1)
        final_rewards = weighted_base_rewards + p * r_synthetic
        
        # --- 6. UPDATE BUFFER AND LOGGING ---
        print("--- Reward Calculation Summary ---")
        print(f"VLM Scores: Safety(w1)={w1:.2f}, Comfort(w2)={w2:.2f}, Efficiency(w3)={w3:.2f}")
        print("Weighted Base Rewards (first 5):", list(np.round(weighted_base_rewards.flatten()[:5], 4)))
        print("Synthetic Rewards (first 5):  ", list(np.round(r_synthetic.flatten()[:5], 4)))
        print("Final Rewards (first 5):      ", list(np.round(final_rewards.flatten()[:5], 4)))
        
        self.rollout_buffer.clear_render_arrays()
        self.rollout_buffer.rewards = final_rewards


    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: CLIPRolloutBuffer,
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

            assert isinstance(infos, list) and len(infos) == env.num_envs, \
                f"Expected infos to be a list of length {env.num_envs}, got {type(infos)} with length {len(infos)}"
            info = infos[0]  # Single environment
            assert isinstance(info, dict), f"Expected infos[0] to be a dict, got {type(info)}"
            render_arrays = info.get("render_arrays", new_obs)
            assert isinstance(render_arrays, np.ndarray) and render_arrays.shape[0] == 3 and render_arrays.shape[1:] == (384, 384), \
                f"Expected render_arrays shape [3, 384, 384], got {render_arrays.shape}"
            speeds = info.get("speed_ms", 0.0)

            # This per-step synthetic reward calculation is now only for immediate logging/debugging
            if not self.inference_only:
                frame = torch.stack([
                    self.clip_preprocess(Image.fromarray(np.transpose(render_arrays, (1, 2, 0)).astype(np.uint8)))
                ]).to(self.device)
                r_synthetic_step = compute_rewards(
                    model=self.reward_model,
                    frames=frame,
                    batch_size=1,
                ).numpy().reshape(-1, 1)
                r_synthetic_step = np.clip(r_synthetic_step, a_min=0.0, a_max=1.0)
                info["synthetic_reward_step"] = float(r_synthetic_step[0])
                p = self.config.clip_reward_params.get('p', 0.1)
                info["total_reward_estimate"] = float(rewards[0] + p * r_synthetic_step[0])
                # print(f"Step {self.num_timesteps}: synthetic_reward_step={info['synthetic_reward_step']:.4f}, total_reward_estimate={info['total_reward_estimate']:.4f}") # Optional: uncomment for verbose step-by-step logging

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
                infos=infos, # Store the list of infos
                render_arrays=render_arrays,
                speeds=speeds,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        if not self.inference_only:
            self._compute_clip_rewards()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def _log(self) -> None:
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_gt_rew_mean",
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
        if self.ep_clip_info_buffer is None or reset_num_timesteps:
            self.ep_clip_info_buffer = deque(maxlen=100)
        return total_timesteps, callback

    def learn(self: SelfCLIPRewardedPPO, *args, **kwargs) -> SelfCLIPRewardedPPO:
        assert not self.inference_only
        return super().learn(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        super().save(*args, exclude=["reward_model", "vlm_scorer", "worker_frames_tensor"], **kwargs)

    @classmethod
    def load(
            cls: Type[SelfCLIPRewardedPPO],
            path: Union[str, pathlib.Path],
            *,
            env: Optional[VecEnv] = None,
            load_clip: bool = True,
            device: Union[torch.device, str] = "cuda:0",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfCLIPRewardedPPO:
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            if (
                    "net_arch" in data["policy_kwargs"]
                    and len(data["policy_kwargs"]["net_arch"]) > 0
            ):
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(
                        saved_net_arch[0], dict
                ):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
                "policy_kwargs" in kwargs
                and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, "
                f"specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify "
                "new environments."
            )

        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            env = cls._wrap_env(env, data["verbose"])
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
            if force_reset and data is not None:
                data["_last_obs"] = None
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            if "env" in data:
                env = data["env"]

        if "config" not in data:
            data["config"] = Box(default_box=True)
        if not hasattr(data["config"], "action_noise"):
            data["config"].action_noise = None

        data["config"].algorithm_params.device = device
        model = cls(
            env=env,
            config=data["config"],
            inference_only=not load_clip,
        )

        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for "
                    f"more info). Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        if pytorch_variables is not None:
            for name in pytorch_variables:
                if pytorch_variables[name] is None:
                    continue
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        if model.use_sde:
            model.policy.reset_noise()

        if load_clip:
            model._load_modules()
        return model
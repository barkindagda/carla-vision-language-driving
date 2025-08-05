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

    def _compute_clip_rewards(self) -> None:
        assert self.env is not None
        assert self.ep_info_buffer is not None
        ep_info_buffer_maxlen = self.ep_info_buffer.maxlen
        assert ep_info_buffer_maxlen is not None

        frames = self.rollout_buffer.render_arrays  # List of [3, 384, 384]
        if len(frames) == 0:
            return

        # Validate frame shape
        for arr in frames:
            assert arr.shape[0] == 3 and len(arr.shape) == 3 and arr.shape[1:] == (384, 384), \
                f"Expected frame shape [3, 384, 384], got {arr.shape}"

        frames = torch.stack([
            self.clip_preprocess(Image.fromarray(np.transpose(arr, (1, 2, 0)).astype(np.uint8)))
            for arr in frames
        ]).to(self.device)

        r_synthetic = compute_rewards(
            model=self.reward_model,
            frames=frames,
            batch_size=self.config.clip_reward_params.batch_size,
        )
        r_synthetic = r_synthetic.numpy().reshape(-1, 1)

        thre_min, thre_max = 0.0, 1.0
        r_synthetic = np.clip(r_synthetic, a_min=thre_min, a_max=thre_max)
        r_synthetic = (r_synthetic - thre_min) / (thre_max - thre_min)

        base_rewards = np.array(self.rollout_buffer.base_rewards).reshape(-1, 1)
        assert base_rewards.shape[0] == r_synthetic.shape[0], \
            f"Shape mismatch: base_rewards {base_rewards.shape}, r_synthetic {r_synthetic.shape}"

        p = self.config.clip_reward_params.get('p', 0.1)
        rewards = base_rewards + p * r_synthetic

        # Update infos for training
        for i, reward in enumerate(r_synthetic.flatten()):
            idx = i  # Since we process all frames, idx aligns with step
            if idx < len(self.rollout_buffer.infos):
                if not isinstance(self.rollout_buffer.infos[idx], dict):
                    self.rollout_buffer.infos[idx] = {}
                self.rollout_buffer.infos[idx]['synthetic_reward'] = float(reward)
                self.rollout_buffer.infos[idx]['total_reward'] = float(base_rewards[i] + p * reward)

        print("R_synthetic ...")
        print(list(np.round(r_synthetic.flatten(), 4)))
        print("Base rewards (R_original) ...")
        print(list(np.round(base_rewards.flatten(), 4)))
        print("Final rewards (R_new) ...")
        print(list(np.round(rewards.flatten(), 4)))

        speeds = np.array(self.rollout_buffer.speeds)
        print("Speeds (m/s) ...")
        print(list(np.round(speeds.flatten(), 4)))

        self.rollout_buffer.clear_render_arrays()
        self.rollout_buffer.rewards = rewards

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

            # Validate infos and extract render_arrays, speeds
            assert isinstance(infos, list) and len(infos) == env.num_envs, \
                f"Expected infos to be a list of length {env.num_envs}, got {type(infos)} with length {len(infos)}"
            info = infos[0]  # Single environment
            assert isinstance(info, dict), f"Expected infos[0] to be a dict, got {type(info)}"
            render_arrays = info.get("render_arrays", new_obs)
            assert isinstance(render_arrays, np.ndarray) and render_arrays.shape[0] == 3 and render_arrays.shape[1:] == (384, 384), \
                f"Expected render_arrays shape [3, 384, 384], got {render_arrays.shape}"
            speeds = info.get("speed_ms", 0.0)

            # Compute synthetic reward for this step
            if not self.inference_only:
                frame = torch.stack([
                    self.clip_preprocess(Image.fromarray(np.transpose(render_arrays, (1, 2, 0)).astype(np.uint8)))
                ]).to(self.device)
                r_synthetic = compute_rewards(
                    model=self.reward_model,
                    frames=frame,
                    batch_size=1,
                ).numpy().reshape(-1, 1)
                r_synthetic = np.clip(r_synthetic, a_min=0.0, a_max=1.0)
                r_synthetic = (r_synthetic - 0.0) / (1.0 - 0.0)
                info["synthetic_reward"] = float(r_synthetic[0])
                p = self.config.clip_reward_params.get('p', 0.1)
                info["total_reward"] = float(rewards[0] + p * r_synthetic[0])
                print(f"Step {self.num_timesteps}: synthetic_reward={info['synthetic_reward']:.4f}, total_reward={info['total_reward']:.4f}")  # Debug

            callback.update_locals(locals())  # Moved after info update
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
                infos=[info],
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
        super().save(*args, exclude=["reward_model", "worker_frames_tensor"], **kwargs)

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
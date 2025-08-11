import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Optional, Tuple, TypeVar, Type, Union, Dict, Any

import numpy as np
from gymnasium import spaces
import gymnasium as gym
import torch
import torch as th
from box import Box
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean, check_for_correct_spaces, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from Models.vlm_controller import VLMScorer
from Models.CLIP.rollout_buffer import VLMRolloutBuffer

SelfVLMRewardedPPO = TypeVar("SelfVLMRewardedPPO", bound="VLMRewardedPPO")

class VLMRewardedPPO(PPO):
    rollout_buffer: VLMRolloutBuffer

    def __init__(
            self,
            *,
            env: VecEnv,
            config: Box,
            inference_only: bool = False,
    ):
        """
        PPO with VLM-based potential shaping for safety, comfort, and efficiency.
        Args:
            env: Vectorized environment.
            config: Configuration with algorithm_params and vlm_params.
            inference_only: If True, skip VLM loading for inference.
        """
        self.config = config
        self.vlm_scorer = None
        self.ep_vlm_info_buffer = None  # type: Optional[deque]

        super().__init__(
            env=env,
            policy='MultiInputPolicy',
            seed=config.seed,
            verbose=1,
            **self.config.algorithm_params,
        )

        self.inference_only = inference_only
        if not self.inference_only:
            self._setup_model()
            self._load_modules()

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = VLMRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            segment_length=self.config.vlm_params.get("segment_length", 8),
            beta=self.config.vlm_params.get("beta", 0.2),
            kappa=self.config.vlm_params.get("kappa", 0.5),
            smooth_alpha=self.config.vlm_params.get("smooth_alpha", 0.8),
            weights=self.config.vlm_params.get("weights", {"safety": 0.5, "comfort": 0.3, "efficiency": 0.2}),
        )

    def _load_modules(self):
        print("Loading VLMScorer...")
        self.vlm_scorer = VLMScorer(
            model_name=self.config.vlm_params.get("model_name", "DAMO-NLP-SG/VideoLLaMA3-2B-Image"),
            device=self.config.vlm_params.get("device", "cuda"),
            batch_size=self.config.vlm_params.get("batch_size", 32),
            max_new_tokens=self.config.vlm_params.get("max_new_tokens", 512),
            output_dir=self.config.vlm_params.get("output_dir", "./vlm_outputs"),
            verbose=self.config.vlm_params.get("verbose", True),
        )

    def _compute_vlm_potentials(self):
        assert self.vlm_scorer is not None, "VLMScorer not initialized"
        assert self.rollout_buffer is not None, "Rollout buffer not initialized"

        print(f"Computing VLM potentials for {self.rollout_buffer.pos} steps...")
        print(f"Render arrays shape: {np.array(self.rollout_buffer.render_arrays).shape}")
        self.rollout_buffer.compute_potentials_and_shaped_rewards(self.vlm_scorer)

        for i in range(self.rollout_buffer.pos):
            info = self.rollout_buffer.infos[i]
            if isinstance(info, dict):
                info["potentials"] = {
                    "safety": float(self.rollout_buffer.potentials[i, 0, 0]),
                    "comfort": float(self.rollout_buffer.potentials[i, 0, 1]),
                    "efficiency": float(self.rollout_buffer.potentials[i, 0, 2]),
                }
                info["base_reward"] = float(self.rollout_buffer.base_rewards[i, 0])
                info["shaped_reward"] = float(self.rollout_buffer.rewards[i, 0])
                print(f"Step {i}: potentials={info['potentials']}, base_reward={info['base_reward']:.4f}, shaped_reward={info['shaped_reward']:.4f}")

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
        print("Starting rollout collection...")

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
            n_steps += 1

            # Validate infos and extract render_arrays, speeds
            assert isinstance(infos, list) and len(infos) == env.num_envs, \
                f"Expected infos to be a list of length {env.num_envs}, got {type(infos)} with length {len(infos)}"
            info = infos[0]
            assert isinstance(info, dict), f"Expected infos[0] to be a dict, got {type(info)}"
            render_arrays = info.get("render_arrays", new_obs)
            # Handle CARLA frame shapes
            if isinstance(render_arrays, torch.Tensor):
                render_arrays = render_arrays.cpu().numpy()
            if render_arrays.shape != (3, 384, 384):
                if len(render_arrays.shape) == 3 and render_arrays.shape[-1] == 3:
                    render_arrays = np.array(Image.fromarray(render_arrays).resize((384, 384))).transpose(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected render_arrays shape: {render_arrays.shape}, expected [3, 384, 384] or [H, W, 3]")
            assert isinstance(render_arrays, np.ndarray) and render_arrays.shape == (3, 384, 384), \
                f"Expected render_arrays shape [3, 384, 384], got {render_arrays.shape}"
            speeds = info.get("speed_ms", 0.0)

            print(f"Step {n_steps}: render_arrays shape={render_arrays.shape}, type={type(render_arrays)}")

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                infos=infos,
                render_arrays=render_arrays,
                speeds=speeds,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        if not self.inference_only:
            self._compute_vlm_potentials()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        print("Rollout collection completed.")

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
        if self.ep_vlm_info_buffer is None or reset_num_timesteps:
            self.ep_vlm_info_buffer = deque(maxlen=100)
        return total_timesteps, callback

    def learn(self: SelfVLMRewardedPPO, *args, **kwargs) -> SelfVLMRewardedPPO:
        assert not self.inference_only
        return super().learn(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        super().save(*args, exclude=["vlm_scorer"], **kwargs)

    @classmethod
    def load(
            cls: Type[SelfVLMRewardedPPO],
            path: Union[str, pathlib.Path],
            *,
            env: Optional[VecEnv] = None,
            load_vlm: bool = True,
            device: Union[torch.device, str] = "cuda:0",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfVLMRewardedPPO:
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
            inference_only=not load_vlm,
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

        if load_vlm:
            model._load_modules()
        return model
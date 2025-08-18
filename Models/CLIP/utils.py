import cv2
import math
import json
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

def write_json(data, path):
    config_dict = {}
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            if isinstance(v, str) and v.isnumeric():
                config_dict[k] = int(v)
            elif isinstance(v, dict):
                config_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    config_dict[k][k_inner] = v_inner.__str__()
                config_dict[k] = str(config_dict[k])
            else:
                config_dict[k] = v.__str__()
        json.dump(config_dict, f, indent=4)

class VideoRecorder:
    def __init__(self, filename, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, int(fps), (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def add_frame_with_reward(self, frame, reward):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        reward_text = f"Reward: {reward:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(reward_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        position = (frame.shape[1] - text_width - 10, frame.shape[0] - 10)
        cv2.putText(frame, reward_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()

class HParamCallback(BaseCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        hparam_dict = {}
        for k, v in self.config.items():
            if isinstance(v, str) and v.isnumeric():
                hparam_dict[k] = int(v)
            elif isinstance(v, dict):
                hparam_dict[k] = dict()
                for k_inner, v_inner in v.items():
                    hparam_dict[k][k_inner] = v_inner.__str__()
                hparam_dict[k] = str(hparam_dict[k])
            else:
                hparam_dict[k] = v.__str__()
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0,
            "train/policy_loss": 0,
            "train/entropy_loss": 0,
        }
        self.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self) -> bool:
        return True

class TensorboardCallback(BaseCallback):
    """
    Logs per-step custom metrics from the environment's info dictionary.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        for key in ['base_reward', 'synthetic_reward', 'shaping_term', 'total_reward']:
            if key in info and isinstance(info[key], (int, float)):
                self.logger.record(f"custom/{key}", info[key])
        return True

class PostRolloutLogCallback(BaseCallback):
    """
    Logs the mean of reward components and episode statistics from the rollout buffer after a rollout.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.model.rollout_buffer
        if not hasattr(rollout_buffer, 'infos'):
            return

        base_rewards = [info.get('base_reward', 0) for info in rollout_buffer.infos if isinstance(info, dict)]
        shaping_terms = [info.get('shaping_term', 0) for info in rollout_buffer.infos if isinstance(info, dict)]
        synthetic_rewards = [info.get('synthetic_reward', 0) for info in rollout_buffer.infos if isinstance(info, dict)]
        total_rewards = [info.get('total_reward', 0) for info in rollout_buffer.infos if isinstance(info, dict)]

        if base_rewards:
            self.logger.record("rollout/mean_base_reward", np.mean(base_rewards))
        if shaping_terms:
            self.logger.record("rollout/mean_shaping_term", np.mean(shaping_terms))
        if synthetic_rewards:
            self.logger.record("rollout/mean_synthetic_reward", np.mean(synthetic_rewards))
        if total_rewards:
            self.logger.record("rollout/mean_total_reward", np.mean(total_rewards))

        # Log episode statistics
        episode_starts = np.where(rollout_buffer.episode_starts[:, 0])[0]
        episode_starts = np.concatenate([episode_starts, [rollout_buffer.pos]])
        episode_rewards = []
        episode_lengths = []
        for i in range(len(episode_starts) - 1):
            start, end = episode_starts[i], episode_starts[i + 1]
            ep_reward = sum(rollout_buffer.infos[t].get('total_reward', 0) for t in range(start, end))
            ep_length = end - start
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
        if rollout_buffer.pos > episode_starts[-1]:  # Handle partial episode
            start = episode_starts[-1]
            ep_reward = sum(rollout_buffer.infos[t].get('total_reward', 0) for t in range(start, rollout_buffer.pos))
            ep_length = rollout_buffer.pos - start
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
        if episode_rewards:
            self.logger.record("rollout/num_episodes", len(episode_rewards))
            self.logger.record("rollout/ep_total_reward_mean", np.mean(episode_rewards))
            self.logger.record("rollout/ep_len_mean", np.mean(episode_lengths))

    def _on_step(self) -> bool:
        return True

class VideoRecorderCallback(BaseCallback):
    def __init__(self, video_path, frame_size, video_length=-1, fps=30, skip_frame=1, verbose=0):
        super().__init__(verbose)
        self.video_recorder = VideoRecorder(video_path, frame_size, fps)
        self.max_length = video_length
        self.skip_frame = skip_frame

    def _on_step(self) -> bool:
        if self.max_length != -1 and self.num_timesteps > self.max_length:
            self.video_recorder.release()
            return False
        if self.num_timesteps % self.skip_frame != 0:
            return True
        display = self.training_env.unwrapped.envs[0].env.display
        frame = np.array(pygame.surfarray.array3d(display), dtype=np.uint8).transpose([1, 0, 2])
        reward = self.locals['infos'][0].get('total_reward', self.locals['rewards'][0])
        self.video_recorder.add_frame_with_reward(frame, reward)
        return True

    def _on_training_end(self) -> None:
        self.video_recorder.release()

def lr_schedule(initial_value: float, end_value: float, rate: float):
    def func(progress_remaining: float) -> float:
        if progress_remaining <= 0:
            return end_value
        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))
    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    return func

class HistoryWrapperObsDict(gym.Wrapper):
    def __init__(self, env: gym.Env, horizon: int = 2, obs_key: str = 'vae_latent'):
        self.obs_key = obs_key
        assert isinstance(env.observation_space.spaces[obs_key], gym.spaces.Box)
        print("Wrapping the env with HistoryWrapperObsDict.")
        wrapped_obs_space = env.observation_space.spaces[self.obs_key]
        wrapped_action_space = env.action_space
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)
        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)
        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))
        env.observation_space.spaces[obs_key] = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)
        super().__init__(env)
        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, **kwargs):
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs_dict = self.env.reset(**kwargs)
        obs = obs_dict[self.obs_key]
        self.obs_history[..., -obs.shape[-1]:] = obs
        obs_dict[self.obs_key] = self._create_obs_from_history()
        return obs_dict, {}

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = obs_dict[self.obs_key]
        last_ax_size = obs.shape[-1]
        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs
        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action
        obs_dict[self.obs_key] = self._create_obs_from_history()
        return obs_dict, reward, done, False, info

class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        print("Wrapping the env with FrameSkip.")
        self._skip = skip

    def step(self, action: np.ndarray):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
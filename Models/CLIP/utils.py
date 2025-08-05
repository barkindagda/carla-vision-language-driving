import cv2
import math
import json
import gym
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
        }
        self.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self) -> bool:
        return True

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_speeds = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]
        self.episode_rewards.append(reward)
        if 'speed_ms' in info:
            self.episode_speeds.append(info['speed_ms'])

        # Log per-step metrics
        if 'speed_ms' in info:
            self.logger.record("custom/speed_ms", info['speed_ms'])
        if 'reward' in info:
            self.logger.record("custom/step_reward", info['reward'])
        if 'safety_reward' in info:
            self.logger.record("custom/safety_reward", info['safety_reward'])
        if 'progress_reward' in info:
            self.logger.record("custom/progress_reward", info['progress_reward'])
        if 'smoothness_reward' in info:
            self.logger.record("custom/smoothness_reward", info['smoothness_reward'])
        if 'collision_penalty' in info:
            self.logger.record("custom/collision_penalty", info['collision_penalty'])
        # Log synthetic_reward, default to 0.0 if missing
        self.logger.record("custom/synthetic_reward", info.get('synthetic_reward', 0.0))
        if 'total_reward' in info:
            self.logger.record("custom/total_reward", info['total_reward'])
        if 'pedestrian_distance' in info:
            self.logger.record("custom/pedestrian_distance", info['pedestrian_distance'])
        if 'distance_to_goal' in info:
            self.logger.record("custom/distance_to_goal", info['distance_to_goal'])

        # Log episode metrics when done
        if done:
            self.episode_count += 1
            episode_length = len(self.episode_rewards)
            total_reward = sum(self.episode_rewards)
            mean_reward = total_reward / episode_length if episode_length > 0 else 0
            avg_speed = np.mean(self.episode_speeds) if self.episode_speeds else 0
            self.logger.record("custom/episode_count", self.episode_count)
            self.logger.record("custom/total_reward", total_reward)
            self.logger.record("custom/mean_reward", mean_reward)
            self.logger.record("custom/episode_length", episode_length)
            self.logger.record("custom/avg_speed", avg_speed)
            self.logger.record("custom/collision_detected", 1 if info.get('collision_detected', False) else 0)
            self.logger.record("custom/successful_ep", info.get('successful_ep', 0))
            self.logger.record("custom/collision_ep", info.get('collision_ep', 0))
            self.logger.record("custom/stall_ep", info.get('stall_ep', 0))
            self.logger.record("custom/lane_ep", info.get('lane_ep', 0))
            self.logger.record("time/num_timesteps", self.num_timesteps)
            self.episode_rewards = []
            self.episode_speeds = []

        self.logger.dump(self.num_timesteps)
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
        reward = self.locals['rewards'][0]
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
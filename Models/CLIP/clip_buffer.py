import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class CLIPRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores rendered frames for CLIP reward computation.
    Compatible with Box observation spaces (e.g., 3x384x384 RGB images).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device = torch.device("cpu"),
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.render_arrays = None
        self.base_rewards = None
        self.speeds = None
        self.infos = None
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.render_arrays = np.zeros((self.buffer_size, *self.observation_space.shape), dtype=np.uint8)
        self.base_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.speeds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = [{} for _ in range(self.buffer_size)]

    def add(self, obs, action, reward, episode_start, value, log_prob, infos, render_arrays, speeds):
        """
        Add a new transition to the buffer, including render_arrays, base_rewards, speeds, and infos.
        """
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.base_rewards[self.pos] = np.array(reward).copy()  # Store original rewards
        self.rewards[self.pos] = np.array(reward).copy()  # Will be updated with CLIP rewards
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.render_arrays[self.pos] = np.array(render_arrays).copy()
        self.speeds[self.pos] = np.array(speeds).copy()
        self.infos[self.pos] = infos.copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def clear_render_arrays(self):
        """
        Clear render_arrays to free memory after CLIP reward computation.
        """
        self.render_arrays = np.zeros_like(self.render_arrays)


class CLIPReplayBuffer(CLIPRolloutBuffer):
    """
    Replay buffer for CLIP reward computation, extending CLIPRolloutBuffer.
    Currently a placeholder for future extensions (e.g., prioritized replay).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device = torch.device("cpu"),
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

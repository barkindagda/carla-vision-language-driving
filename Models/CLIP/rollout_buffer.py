import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from tqdm.auto import tqdm
class VLMRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer for potential-based shaping rewards using a VLM.
    Computes potentials for safety, comfort, and efficiency.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device = torch.device("cuda"),
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        segment_length: int = 3,
        beta: float = 0.2,
        kappa: float = 0.5,
        smooth_alpha: float = 0.8,
        weights: dict = None
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.segment_length = segment_length
        self.beta = beta
        self.kappa = kappa
        self.smooth_alpha = smooth_alpha
        self.weights = weights or {"safety": 0.5, "comfort": 0.3, "efficiency": 0.2}
        self.render_arrays = None
        self.base_rewards = None
        self.speeds = None
        self.infos = None
        self.potentials = None
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.render_arrays = np.zeros((self.buffer_size, self.n_envs, *self.observation_space.shape), dtype=np.uint8)
        self.base_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.speeds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = [{} for _ in range(self.buffer_size)]
        self.potentials = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32)  # [safety, comfort, efficiency]

    def add(self, obs, action, reward, episode_start, value, log_prob, infos, render_arrays, speeds):
        """
        Add a new transition to the buffer.
        """
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.base_rewards[self.pos] = np.array(reward).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.render_arrays[self.pos, 0] = np.array(render_arrays).copy()  # Assign to env 0
        self.speeds[self.pos, 0] = np.array(speeds).copy()  # Consistent for speeds (already works, but explicit)
        self.infos[self.pos] = infos[0].copy()  # Change to dict (not list); update self.infos = [{} for _ in range(self.buffer_size)]

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_potentials_and_shaped_rewards(self, vlm_scorer):
        """
        Compute potentials for safety, comfort, efficiency and update rewards.
        Args:
            vlm_scorer: Instance of VLMScorer.
        """
        self.potentials = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32)

        for env_idx in range(self.n_envs):
            episode_starts = np.where(self.episode_starts[:, env_idx])[0]
            episode_starts = np.concatenate([[0], episode_starts, [self.buffer_size]])

            for ep_start, ep_end in zip(episode_starts[:-1], episode_starts[1:]):
                segments = []
                segment_indices = []
                for seg_start in range(ep_start, ep_end, self.segment_length):
                    seg_end = min(seg_start + self.segment_length, ep_end)
                    if seg_end - seg_start < self.segment_length:
                        continue
                    segments.append({
                        "frames": self.render_arrays[seg_start:seg_end, env_idx],
                        "speeds": self.speeds[seg_start:seg_end, env_idx],
                        "infos": [self.infos[t] for t in range(seg_start, seg_end)],  # List of dicts (no nesting)
                        "start_idx": seg_start,
                        "end_idx": seg_end
                    })
                    segment_indices.append((seg_start, seg_end))

                # MODIFICATION: Add tqdm progress bar here
                num_batches = (len(segments) + vlm_scorer.batch_size - 1) // vlm_scorer.batch_size
                batch_iterator = range(0, len(segments), vlm_scorer.batch_size)
                
                for batch_start in tqdm(batch_iterator, total=num_batches, desc="VLM Scoring Segments"):
                    batch_segments = segments[batch_start:batch_start + vlm_scorer.batch_size]
                    if not batch_segments:
                        continue

                    frames_batch = np.stack([seg["frames"] for seg in batch_segments])
                    speeds_batch = np.stack([seg["speeds"] for seg in batch_segments])
                    infos_batch = [seg["infos"] for seg in batch_segments]


                    scores = vlm_scorer.score_segment_batch(frames_batch, speeds_batch, infos_batch)  # [batch_size, 3]

                    for i, (seg_start, seg_end) in enumerate(segment_indices[batch_start:batch_start + len(batch_segments)]):
                        potentials = self.kappa * scores[i]  # [safety, comfort, efficiency]
                        self.potentials[seg_start:seg_end, env_idx] = potentials

                if self.smooth_alpha > 0:
                    for t in range(ep_start + 1, ep_end):
                        for obj_idx in range(3):  # Safety, comfort, efficiency
                            self.potentials[t, env_idx, obj_idx] = (
                                self.smooth_alpha * self.potentials[t - 1, env_idx, obj_idx]
                                + (1 - self.smooth_alpha) * self.potentials[t, env_idx, obj_idx]
                            )

        # Compute combined shaping term
        for t in range(self.buffer_size - 1):
            for env_idx in range(self.n_envs):
                if self.episode_starts[t + 1, env_idx]:
                    continue
                shaping_term = 0.0
                for obj_idx, obj in enumerate(["safety", "comfort", "efficiency"]):
                    shaping_term += self.weights[obj] * (
                        self.gamma * self.potentials[t + 1, env_idx, obj_idx]
                        - self.potentials[t, env_idx, obj_idx]
                    )
                self.rewards[t, env_idx] = self.base_rewards[t, env_idx] + self.beta * shaping_term

        for env_idx in range(self.n_envs):
            episode_ends = np.where(self.episode_starts[1:, env_idx])[0]
            if len(episode_ends) == 0:
                episode_ends = [self.buffer_size - 1]
            for end_idx in episode_ends:
                self.potentials[end_idx, env_idx] = 0

    def clear_render_arrays(self):
        """
        Clear render_arrays to free memory.
        """
        self.render_arrays = np.zeros_like(self.render_arrays)
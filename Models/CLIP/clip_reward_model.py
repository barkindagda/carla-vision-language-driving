from typing import List

import open_clip
import torch
import torch.nn as nn
from torch import Tensor

from Models.CLIP.transform import image_transform


class CLIPEmbed(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        if isinstance(clip_model.visual.image_size, int):
            image_size = clip_model.visual.image_size
        else:
            image_size = clip_model.visual.image_size[0]
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.transform(x)  # [batch, 3, 224, 224]
            x = self.clip_model.encode_image(x, normalize=True)  # [batch, 1024]
        return x


class CLIPReward(nn.Module):
    def __init__(
            self,
            *,
            model: CLIPEmbed,
            alpha: float,
            target_prompts: torch.Tensor,
            baseline_prompts: torch.Tensor,
    ) -> None:
        super().__init__()
        self.clip_embed_module = model
        self.alpha = alpha
        targets = self.embed_prompts(target_prompts)
        self.register_buffer("targets", targets)
        if len(baseline_prompts) > 0:
            baselines = self.embed_prompts(baseline_prompts)
            self.register_buffer("baselines", baselines)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = x @ self.targets.T
        if hasattr(self, "baselines"):
            z = y[:, 0] - (x @ self.baselines.T)[:, 0]
        else:
            z = y[:, 0]
        return self.alpha * z

    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return open_clip.tokenize(x)

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.clip_embed_module.clip_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.clip_embed_module.forward(x)


def compute_rewards(
        model: CLIPReward,
        frames: torch.Tensor,
        batch_size: int,
) -> Tensor:
    n_samples = len(frames)
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i: i + batch_size].to(next(model.parameters()).device)
            with torch.no_grad():
                embeddings = model.clip_embed_module(frames_batch)
                rewards_batch = model(embeddings)
            rewards[i: i + batch_size] = rewards_batch
    return rewards
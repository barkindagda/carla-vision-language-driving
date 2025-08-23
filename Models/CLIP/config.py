import torch as th
from box import Box
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from Models.CLIP.utils import lr_schedule

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch.nn as nn
import gymnasium as gym
import torch

class CustomCNN(nn.Module):
    def __init__(self, input_shape, features_dim=1):
        super(CustomCNN, self).__init__()
        n_input_channels = input_shape[0]

        if n_input_channels == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),  # (16, 190, 190)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 94, 94)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (64, 46, 46)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),  # (128, 22, 22)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 20, 20)
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).view(-1).shape[0]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x


class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0

        if isinstance(observation_space, gym.spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if key == "rgb_camera":
                    extractors[key] = CustomCNN(subspace.shape, features_dim=features_dim)
                    total_concat_size += features_dim
                else:
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)
        else:
            extractors["default"] = CustomCNN(observation_space.shape, features_dim=features_dim)
            total_concat_size = features_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        if isinstance(observations, dict):
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key]))
        else:
            encoded_tensor_list.append(self.extractors["default"](observations))
        return torch.cat(encoded_tensor_list, dim=1)

# --- NEW: Add parameters for the VLM Scorer ---
vlm_params = {
    "vlm_carla": dict(
        model_name="DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        batch_size=8,
        max_new_tokens=32,
        clip_size=3,  # Number of frames per short clip
        step_size=3,  # Step size for the rolling clip window
    ),
}

algorithm_params = {
    "PPO": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=[dict(pi=[500, 300], vf=[500, 300])],
            features_extractor_class=CustomMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
    ),
}

reward_params = {
    "reward_carla": dict(
        target_speed=20.0,
    ),
    "reward_clg": dict(
        pretrained_model="ViT-B-32",
        batch_size=1,
        alpha=0.5,
        target_prompts=["The vehicle is stopped, and the pedestrian is safely crossing the road."],
        baseline_prompts=["The vehicle is moving toward a pedestrian in close proximity there is a collision."],
        p=0.1,  # Weight for R_synthetic
    ),
}

_CONFIG_carla_ppo = {
    "algorithm": "CLIP-PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": ["rgb_camera"],
    "reward_fn": "reward_carla",
    "reward_params": reward_params["reward_carla"],
    "clip_reward_params": reward_params["reward_clg"],
    "vlm_params": vlm_params["vlm_carla"], # --- NEW: Add the VLM parameters to the config ---
    "vlm_reward_type": "VLM-RL",
    "obs_res": (384, 384),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "action_space_type": "continuous",
    "use_rgb_bev": False,
    "use_seg_bev": False,
}

CONFIGS = {
    "carla_ppo": _CONFIG_carla_ppo,
}

CONFIG = None

def set_config(config_name):
    global CONFIG
    CONFIG = Box(CONFIGS[config_name], default_box=True)
    return CONFIG
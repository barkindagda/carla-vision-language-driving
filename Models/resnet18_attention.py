import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import os

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Models.feature_extractors.resnet18 import ResNetUNet

class ResnetAttention(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor that uses a pre-trained segmentation
    model to create a pedestrian-aware feature embedding.

    :param observation_space: The observation space of the environment.
    :param features_dim: The number of features to extract.
    :param resnet18_unet_weights_path: Path to the pre-trained ResNet18-UNet weights.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int, resnet18_unet_weights_path: str):
        super().__init__(observation_space, features_dim)

        if not os.path.exists(resnet18_unet_weights_path):
            raise FileNotFoundError(f"ResNet18-UNet weights not found at path: {resnet18_unet_weights_path}")

        self.seg_model = ResNetUNet(n_class=28)
        
        # <<< CORRECTION: Load the model to 'cpu' first.
        # SB3 will automatically move it to the correct device (e.g., cuda) later.
        self.seg_model.load_state_dict(torch.load(resnet18_unet_weights_path, map_location='cpu'))
        
        # The .to(self.device) call will be handled automatically by the parent PPO model.
        self.seg_model.eval()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pedestrian_class_index = 11
        self.linear_projection = nn.Linear(512, features_dim)

        # Corrected the print statement to match the class name
        print("✅ ResnetAttention extractor initialized successfully.")
        print(f"   - UNet weights loaded from: {resnet18_unet_weights_path}")
        print(f"   - Output features_dim will be: {features_dim}")


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            seg_logits = self.seg_model(observations)
            pred_labels = torch.argmax(seg_logits, dim=1)
            pedestrian_mask = (pred_labels == self.pedestrian_class_index).float().unsqueeze(1)

            layer0 = self.seg_model.layer0(observations)
            layer1 = self.seg_model.layer1(layer0)
            layer2 = self.seg_model.layer2(layer1)
            layer3 = self.seg_model.layer3(layer2)
            encoder_feature_map = self.seg_model.layer4(layer3)

            image_embedding = torch.flatten(self.pool(encoder_feature_map), 1)

            attention_map = F.interpolate(pedestrian_mask, size=encoder_feature_map.shape[2:], mode='nearest')
            attended_feature_map = encoder_feature_map * attention_map
            attended_embedding = torch.flatten(self.pool(attended_feature_map), 1)

            output_embedding_512 = attended_embedding + image_embedding

        final_features = self.linear_projection(output_embedding_512)

        return final_features
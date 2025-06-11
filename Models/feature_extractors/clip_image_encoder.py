import gymnasium as gym
import torch
import torch.nn as nn
from torchvision import transforms as T_vis
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation, CLIPVisionModel, CLIPImageProcessor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import os
import glob
from typing import Optional, List

# 👇 ADD THIS IMPORT FOR VIRIDIS COLORMAP
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For accessing colormaps

class CLIPSegAttentionGenerator:
    """
    Generates attention maps (heatmaps) using a CLIPSeg model
    for given images and text prompts.
    Optionally saves these heatmaps to a specified directory using Viridis colormap.
    """
    def __init__(self, 
                 model_name: str = 'CIDAS/clipseg-rd64-refined', 
                 device: Optional[str] = None,
                 heatmap_save_dir: Optional[str] = None,
                 max_heatmaps_to_keep: int = 1000,
                 save_from_logits: bool = True):
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        self.heatmap_save_dir = heatmap_save_dir
        self.max_heatmaps_to_keep = max_heatmaps_to_keep
        self.save_from_logits = save_from_logits
        self.saved_heatmap_counter = 0

        if self.heatmap_save_dir:
            os.makedirs(self.heatmap_save_dir, exist_ok=True)
            print(f"CLIPSegAttentionGenerator: Heatmaps (Viridis) will be saved to {self.heatmap_save_dir}")

    def _save_heatmap_image(self, heatmap_tensor: torch.Tensor, filename_prefix: str = "heatmap"):
        """
        Saves a heatmap tensor as an image using the Viridis colormap.
        Manages the number of saved images in the directory.
        heatmap_tensor: The 2D heatmap tensor (logits or probabilities) on CPU.
        """
        if not self.heatmap_save_dir:
            return

        heatmap_numpy = heatmap_tensor.cpu().numpy()

        # Normalize for applying colormap
        heatmap_normalized = (heatmap_numpy - heatmap_numpy.min()) / \
                             (heatmap_numpy.max() - heatmap_numpy.min() + 1e-6)
        
        # 👇 APPLY VIRIDIS COLORMAP
        # Get the Viridis colormap
        viridis_cmap = cm.get_cmap('viridis') 
        # Apply the colormap. This returns an RGBA image (H, W, 4)
        colored_heatmap_rgba = viridis_cmap(heatmap_normalized)
        # Convert to RGB by discarding the alpha channel and scaling to 0-255
        colored_heatmap_rgb_uint8 = (colored_heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        
        heatmap_pil = Image.fromarray(colored_heatmap_rgb_uint8, mode='RGB')

        filepath = os.path.join(self.heatmap_save_dir, f"{filename_prefix}_{self.saved_heatmap_counter:06d}.png")
        self.saved_heatmap_counter += 1

        try:
            heatmap_pil.save(filepath)
        except Exception as e:
            print(f"Warning: Could not save Viridis heatmap {filepath}: {e}")
            return

        # Cleanup logic remains the same
        try:
            files = glob.glob(os.path.join(self.heatmap_save_dir, f"{filename_prefix}_*.png"))
            if len(files) > self.max_heatmaps_to_keep:
                files.sort(key=os.path.getmtime)
                for old_file in files[:-self.max_heatmaps_to_keep]:
                    try:
                        os.remove(old_file)
                    except OSError:
                        pass 
        except Exception as e:
            print(f"Warning: Error during heatmap cleanup: {e}")

    def get_attention_map(self, pil_image: Image.Image, text_prompt: str) -> torch.Tensor:
        """
        Generates an attention map. If heatmap_save_dir is set, saves the map.
        Input:
            pil_image (PIL.Image.Image): The input image.
            text_prompt (str): Text prompt for attention.
        Returns:
            torch.Tensor: 2D tensor (heatmap probabilities) on CPU. Shape: [height, width].
        """
        inputs = self.processor(
            text=[text_prompt],
            images=[pil_image],
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.squeeze(0)
        
        if self.heatmap_save_dir:
            # Sanitize text_prompt for filename
            safe_prompt = "".join(c if c.isalnum() else "_" for c in text_prompt)
            if self.save_from_logits:
                self._save_heatmap_image(logits.cpu(), filename_prefix=f"heatmap_logits_prompt_{safe_prompt}")
            else:
                heatmap_probs_for_save = torch.sigmoid(logits).cpu()
                self._save_heatmap_image(heatmap_probs_for_save, filename_prefix=f"heatmap_probs_prompt_{safe_prompt}")

        heatmap_probabilities = torch.sigmoid(logits).cpu()
        return heatmap_probabilities


class CLIPAndSegFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that combines:
    1. Spatial features from CLIPVisionModel.
    2. A pedestrian attention map from CLIPSegAttentionGenerator.
    """
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 clip_vision_model_name: str = "openai/clip-vit-base-patch32",
                 clipseg_model_name: str = "CIDAS/clipseg-rd64-refined",
                 pedestrian_prompt: str = "human, pedestrian",
                 device: Optional[str] = None,
                 extractor_heatmap_save_dir: Optional[str] = None, 
                 max_heatmaps_to_keep: int = 1000, # This should be passed to CLIPSegAttentionGenerator
                 extractor_save_from_logits: bool = True):
        
        super().__init__(observation_space, features_dim)
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.pedestrian_prompt = pedestrian_prompt

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_vision_model_name)
        self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_vision_model_name).to(self.device)
        self.clip_vision_model.eval()

        cfg = self.clip_vision_model.config
        self.num_patches_side = cfg.image_size // cfg.patch_size
        self.clip_spatial_embed_dim = cfg.hidden_size

        self.seg_attention_generator = CLIPSegAttentionGenerator(
            model_name=clipseg_model_name,
            device=self.device,
            heatmap_save_dir=extractor_heatmap_save_dir,
            max_heatmaps_to_keep=max_heatmaps_to_keep, # Pass the argument here
            save_from_logits=extractor_save_from_logits
        )

        combined_channels = self.clip_spatial_embed_dim + 1 

        self.processing_convs = nn.Sequential(
            nn.Conv2d(combined_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(self.device)
        
        self.final_fc = nn.Linear(512, features_dim).to(self.device)

    def _tensor_to_pil_list(self, tensor_batch: torch.Tensor) -> List[Image.Image]:
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)
        
        pil_images = []
        for i in range(tensor_batch.shape[0]):
            img_tensor_chw = tensor_batch[i].cpu()
            
            if img_tensor_chw.dtype == torch.float32 and img_tensor_chw.max() <= 1.0 and img_tensor_chw.min() >= 0.0:
                img_tensor_chw = (img_tensor_chw * 255).byte()
            elif img_tensor_chw.dtype != torch.uint8:
                img_tensor_chw = ((img_tensor_chw - img_tensor_chw.min()) / 
                                  (img_tensor_chw.max() - img_tensor_chw.min() + 1e-6) * 255).byte()

            if img_tensor_chw.shape[0] == 1:
                pil_img = T_vis.ToPILImage(mode='L')(img_tensor_chw)
            elif img_tensor_chw.shape[0] == 3:
                pil_img = T_vis.ToPILImage(mode='RGB')(img_tensor_chw)
            else:
                raise ValueError(f"Unsupported channel size for ToPILImage: {img_tensor_chw.shape[0]}")

            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            pil_images.append(pil_img)
        return pil_images

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        pil_images_list = self._tensor_to_pil_list(observations)
        
        clip_vision_inputs = self.clip_image_processor(
            images=pil_images_list,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            vision_outputs = self.clip_vision_model(**clip_vision_inputs)
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :] 
        
        spatial_clip_features = patch_embeddings.permute(0, 2, 1).reshape(
            batch_size,
            self.clip_spatial_embed_dim,
            self.num_patches_side,
            self.num_patches_side
        )

        attention_maps_probs_list = []
        for i in range(batch_size):
            attn_map_probs = self.seg_attention_generator.get_attention_map(
                pil_image=pil_images_list[i], 
                text_prompt=self.pedestrian_prompt
            )
            attention_maps_probs_list.append(attn_map_probs)
        
        batched_attention_maps_probs = torch.stack(attention_maps_probs_list).to(self.device)
        if batched_attention_maps_probs.ndim == 3:
            batched_attention_maps_probs = batched_attention_maps_probs.unsqueeze(1)

        resized_attention_map = T_vis.functional.resize(
            batched_attention_maps_probs,
            (self.num_patches_side, self.num_patches_side),
            interpolation=T_vis.InterpolationMode.BILINEAR,
            antialias=True 
        )

        combined_features = torch.cat((spatial_clip_features, resized_attention_map), dim=1)
        processed_features = self.processing_convs(combined_features)
        flattened = torch.flatten(processed_features, start_dim=1)
        final_embedding = self.final_fc(flattened)
        
        return final_embedding
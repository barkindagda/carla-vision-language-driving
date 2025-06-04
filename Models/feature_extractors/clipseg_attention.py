import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import argparse

class CLIPSegAttention:
    def __init__(self, model_name='CIDAS/clipseg-rd64-refined', device=None)
        if device is None:
            self.device='cuda' if torch.cuda.is_available() else "cpu"
        else:
            self.device=device

        self.model_name=model_name
        self.processor=AutoProcessor.from_pretrained(self.model_name)
        self.model=CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def get_attention_map(self, pil_image:Image.Image, text_prompt):
        """
        Input:
            The input image in PIL format
            text_prompt should be in string format to guide the attention

        Returns:
            torch.Tensor: 2D tensor (heatmap) of probabilities on the cpu
            shape is [height, width]
        """
        inputs=self.processor(text=[text_prompt], images=[pil_image], padding="max_length", return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs=self.model(**inputs)
        
        heatmap_logits=torch.sigmoid(logits).cpu()

        return heatmap_logits
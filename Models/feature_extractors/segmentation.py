import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from resnet18 import ResNetUNet
import helper

# Configuration
INPUT_FOLDER = '/home/server00/BARKIN/carla-vision-language-driving/Models/feature_extractors/pedestrian_images'  # Replace with your input folder path
OUTPUT_FOLDER = './output'  # Replace with your output folder path
MODEL_PATH = '/home/server00/BARKIN/carla-vision-language-driving/Models/feature_extractors/resne18unet_weights.pt'  # Path to pretrained ResNet18 U-Net weights
NUM_CLASSES = 28
BATCH_SIZE = 3

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
TRANS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet
])

# Reverse transform for visualization
def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

# Load and process images
def process_image_folder(input_folder, output_folder, model_path, num_classes=14):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    model = ResNetUNet(n_class=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Collect image paths
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process images in batches
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        input_images = []
        input_images_rgb = []
        
        # Load and preprocess batch
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = TRANS(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
                input_images.append(input_tensor)
                input_images_rgb.append(reverse_transform(input_tensor.squeeze(0)))
            except Exception as e:
                print(f'Error loading {img_path}: {e}')
                continue
        
        if not input_images:
            continue
        
        # Concatenate batch
        input_batch = torch.cat(input_images, dim=0).to(device)  # Shape: [batch_size, 3, 224, 224]
        
        # Generate predictions
        with torch.no_grad():
            pred = model(input_batch)  # Shape: [batch_size, num_classes, 224, 224]
            pred = torch.sigmoid(pred)  # Apply sigmoid for probabilities
            pred = pred.data.cpu().numpy()  # Shape: [batch_size, num_classes, 224, 224]
        
        # Convert predictions to colored masks
        pred_rgb = [helper.masks_to_colorimg(x) for x in pred]
        
        # Visualize input images and predictions
        helper.plot_side_by_side([input_images_rgb, pred_rgb])
        
        # Save segmentation maps
        for j, img_path in enumerate(batch_paths):
            output_path = os.path.join(output_folder, f'seg_{os.path.basename(img_path)}')
            Image.fromarray(pred_rgb[j]).save(output_path)
            print(f'Saved segmentation map for {os.path.basename(img_path)} to {output_path}')

if __name__ == '__main__':
    process_image_folder(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH, num_classes=NUM_CLASSES)
    print("Done")
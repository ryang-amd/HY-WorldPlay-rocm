#!/usr/bin/env python3
"""
Extract the first frame from vKITTI latent file by decoding with VAE.
This gives us a driving scene image for evaluation.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/ruijyang/workrepo/HY-WorldPlay-rocm')

from diffusers import AutoencoderKLHunyuanVideo
from PIL import Image
import numpy as np

def main():
    # Load a latent file
    latent_path = "/data/ruijyang/datasets/vkitti_training_data_full/latents/Scene01_15-deg-left_Camera_0.pt"
    output_path = "/home/ruijyang/workrepo/HY-WorldPlay-rocm/assets/img/vkitti_driving.png"
    
    print(f"Loading latent from: {latent_path}")
    latent_data = torch.load(latent_path, map_location='cpu')
    
    # The data is a dictionary with 'image_cond' being the first frame latent
    print(f"Keys: {latent_data.keys()}")
    
    # Use image_cond which is the first frame: [1, 32, 1, 30, 52]
    latent = latent_data['image_cond']
    print(f"image_cond shape: {latent.shape}")
    
    # Load VAE
    model_path = "/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5"
    print(f"Loading VAE from: {model_path}")
    
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float16
    ).to("cuda")
    
    print(f"Decoding latent shape: {latent.shape}")
    
    # Decode
    latent = latent.to("cuda", dtype=torch.float16)
    
    # Scale latent (HunyuanVideo uses specific scaling)
    latent = latent / 0.476986
    
    with torch.no_grad():
        # Decode - output is [B, C, T, H, W]
        # Need to handle the temporal dimension for VAE
        # For single frame, we might need to expand or handle differently
        try:
            decoded = vae.decode(latent).sample
        except Exception as e:
            print(f"Direct decode failed: {e}")
            # Try with expanded temporal dim
            latent_expanded = latent.repeat(1, 1, 4, 1, 1)  # Expand to 4 frames
            decoded = vae.decode(latent_expanded).sample
            decoded = decoded[:, :, :1, :, :]  # Take first frame output
    
    print(f"Decoded shape: {decoded.shape}")
    
    # Convert to image
    # Take first frame: [B, C, T, H, W] -> [C, H, W]
    if decoded.dim() == 5:
        frame = decoded[0, :, 0, :, :]  # First frame
    else:
        frame = decoded[0]
    frame = frame.cpu().float()
    
    # Normalize to [0, 255]
    frame = (frame + 1) / 2 * 255
    frame = frame.clamp(0, 255).byte()
    frame = frame.permute(1, 2, 0).numpy()  # CHW -> HWC
    
    # Save as image
    img = Image.fromarray(frame)
    img.save(output_path)
    print(f"Saved driving scene image to: {output_path}")
    print(f"Image size: {img.size}")

if __name__ == "__main__":
    main()

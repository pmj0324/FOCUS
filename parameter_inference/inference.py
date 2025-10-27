"""
Parameter inference utilities.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from models import SimpleUNet
from diffusion import GaussianDiffusion
from utils import visualize_samples


def load_model(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: checkpoint file path
        device: device
        
    Returns:
        model, diffusion, checkpoint
    """
    # Create model
    model = SimpleUNet(
        in_channels=1,
        out_channels=1,
        cond_dim=6,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Diffusion process
    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model, diffusion, checkpoint


@torch.no_grad()
def generate_samples(
    model,
    diffusion,
    params,
    num_samples=16,
    cfg_scale=2.0,
    ddim_steps=50,
    device='cuda'
):
    """
    Generate conditional samples
    
    Args:
        model: UNet model
        diffusion: GaussianDiffusion
        params: (N, 6) parameters or single (6,) parameter
        num_samples: number of samples to generate (if params is single)
        cfg_scale: CFG scale
        ddim_steps: DDIM sampling steps
        device: device
        
    Returns:
        samples: (N, 1, 256, 256) generated samples
    """
    model.eval()
    
    # Handle parameters
    if params.ndim == 1:
        # Single parameter -> repeat
        params = params.unsqueeze(0).repeat(num_samples, 1)
    
    params = params.to(device)
    batch_size = params.shape[0]
    
    # DDIM sampling
    print(f"Generating {batch_size} samples with CFG scale {cfg_scale}...")
    samples = diffusion.ddim_sample(
        model,
        shape=(batch_size, 1, 256, 256),
        cond=params,
        ddim_timesteps=ddim_steps,
        cfg_scale=cfg_scale,
        progress=True
    )
    
    print("✓ Generation complete!")
    
    return samples


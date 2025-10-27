"""
Run inference for experiments from tasks folder.
"""
import torch
import yaml
import numpy as np
from pathlib import Path
import argparse

from parameter_inference import load_model, generate_samples
from utils import visualize_samples, check_gaussian_distribution


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt', help='Checkpoint file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    exp_dir = Path(args.exp_dir)
    
    # Device
    device = config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint_path = exp_dir / 'checkpoints' / args.checkpoint
    model, diffusion, checkpoint = load_model(checkpoint_path, device)
    
    print("\n" + "=" * 80)
    print("Generate Samples")
    print("=" * 80)
    
    # Load test parameters
    data_config = config['data']
    stats_path = Path(data_config['processed_dir']) / 'normalization_stats.npy'
    
    # Load normalized parameters
    params_normalized = np.load(data_config['params_path'])
    
    # Randomly select a few
    num_test = 8
    indices = np.random.choice(len(params_normalized), num_test, replace=False)
    test_params = torch.FloatTensor(params_normalized[indices])
    
    # Generate samples (multiple CFG scales)
    cfg_scales = [1.0, 2.0, 4.0]
    
    for cfg_scale in cfg_scales:
        print(f"\nCFG scale: {cfg_scale}")
        
        samples = generate_samples(
            model, diffusion, test_params,
            cfg_scale=cfg_scale,
            ddim_steps=config['sampling']['ddim_timesteps'],
            device=device
        )
        
        # Visualize
        visualize_samples(
            samples,
            params=test_params.numpy(),
            save_path=exp_dir / 'figs' / f"samples_cfg{cfg_scale}.png",
            num_display=num_test
        )
    
    print("\n" + "=" * 80)
    print("Gaussian Distribution Check")
    print("=" * 80)
    
    # Check from pure noise
    noise = torch.randn(1, 1, 256, 256).to(device)
    
    check_gaussian_distribution(
        noise,
        title="Initial Noise Distribution",
        save_path=exp_dir / 'figs' / "noise_gaussian_check.png"
    )
    
    print(f"\n✓ Results saved to {exp_dir / 'figs'}")


if __name__ == "__main__":
    main()


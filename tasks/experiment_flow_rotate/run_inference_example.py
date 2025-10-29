"""
Simple example script for parameter inference.

Usage:
    python run_inference_example.py --test_idx 100
    python run_inference_example.py --test_idx 100 --method mle --iterations 500
    python run_inference_example.py --test_idx 100 --method mcmc
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from parameter_inference import ParameterInference


def main():
    parser = argparse.ArgumentParser(description='Flow Matching Parameter Inference Example')
    parser.add_argument('--test_idx', type=int, default=None,
                        help='Index of test sample to use (random if not specified)')
    parser.add_argument('--method', type=str, default='mle',
                        choices=['mle', 'mcmc', 'grid'],
                        help='Inference method to use')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Number of MLE iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for MLE')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of MCMC samples')
    parser.add_argument('--grid_points', type=int, default=20,
                        help='Number of grid points per dimension for grid search')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Paths
    exp_dir = Path(__file__).parent
    checkpoint_path = exp_dir / 'checkpoints' / 'checkpoint_best.pt'
    config_path = exp_dir / 'config.yaml'
    
    print("="*80)
    print("Flow Matching Parameter Inference")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Initialize
    device = args.device if torch.cuda.is_available() else 'cpu'
    inferencer = ParameterInference(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        device=device
    )
    
    # Load test data
    data_dir = Path(inferencer.config['data']['maps_path']).parent
    maps = np.load(data_dir / 'maps_normalized.npy')
    params = np.load(data_dir / 'params_normalized.npy')
    
    # Select test sample
    if args.test_idx is None:
        test_idx = np.random.randint(0, len(maps))
    else:
        test_idx = args.test_idx
        if test_idx >= len(maps):
            print(f"Warning: test_idx {test_idx} out of range, using random index")
            test_idx = np.random.randint(0, len(maps))
    
    x_obs = torch.tensor(maps[test_idx:test_idx+1], device=device, dtype=torch.float32)
    
    # Ensure x_obs has channel dimension
    if x_obs.ndim == 3:
        x_obs = x_obs.unsqueeze(1)
    
    params_true = torch.tensor(params[test_idx:test_idx+1], device=device, dtype=torch.float32)
    
    print(f"\nTest sample index: {test_idx}")
    print(f"True parameters (normalized): {params_true.cpu().numpy()[0]}")
    params_true_denorm = inferencer.denormalize_params(params_true)
    print(f"True parameters (denormalized): {params_true_denorm.cpu().numpy()[0]}")
    
    # Create output directory
    output_dir = exp_dir / 'inference_results'
    output_dir.mkdir(exist_ok=True)
    
    # Run inference based on method
    if args.method == 'mle':
        print(f"\nRunning MLE with {args.iterations} iterations...")
        params_est, losses = inferencer.infer_mle(
            x_obs,
            num_iterations=args.iterations,
            lr=args.lr,
            method='reconstruction'
        )
        
        # Visualize results
        inferencer.visualize_results(
            x_obs,
            params_est,
            params_true_denorm,
            save_path=str(output_dir / f'inference_mle_idx{test_idx}.png')
        )
        
        # Plot loss history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Negative Log-Likelihood', fontsize=12)
        plt.title('MLE Optimization History', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'mle_loss_idx{test_idx}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Loss history saved to {output_dir / f'mle_loss_idx{test_idx}.png'}")
        plt.close()
        
    elif args.method == 'mcmc':
        print(f"\nRunning MCMC with {args.num_samples} samples...")
        samples, acc_rate = inferencer.infer_mcmc(
            x_obs,
            num_samples=args.num_samples,
            burn_in=args.num_samples // 10,
            proposal_std=0.05
        )
        
        # Use posterior mean as estimate
        params_est = torch.tensor(samples.mean(axis=0), device=device).unsqueeze(0)
        
        # Visualize results
        inferencer.visualize_results(
            x_obs,
            params_est,
            params_true_denorm,
            save_path=str(output_dir / f'inference_mcmc_idx{test_idx}.png')
        )
        
        # Plot posterior distributions
        import matplotlib.pyplot as plt
        param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(6):
            axes[i].hist(samples[:, i], bins=50, density=True, alpha=0.7, label='Posterior')
            axes[i].axvline(params_true_denorm[0, i].cpu().numpy(), 
                           color='r', linestyle='--', linewidth=2, label='True')
            axes[i].axvline(samples[:, i].mean(), 
                           color='g', linestyle='-', linewidth=2, label='Mean')
            axes[i].set_xlabel(param_names[i], fontsize=12)
            axes[i].set_ylabel('Density', fontsize=12)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'mcmc_posterior_idx{test_idx}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Posterior distributions saved to {output_dir / f'mcmc_posterior_idx{test_idx}.png'}")
        plt.close()
        
    elif args.method == 'grid':
        print(f"\nRunning grid search with {args.grid_points}x{args.grid_points} points...")
        params_est, likelihoods = inferencer.infer_grid_search(
            x_obs,
            grid_points=args.grid_points
        )
        
        # Visualize results
        inferencer.visualize_results(
            x_obs,
            params_est,
            params_true_denorm,
            save_path=str(output_dir / f'inference_grid_idx{test_idx}.png')
        )
        
        # Plot likelihood surface
        import matplotlib.pyplot as plt
        param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w']
        
        plt.figure(figsize=(10, 8))
        plt.imshow(likelihoods.T, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Log Likelihood')
        plt.xlabel(f'{param_names[0]} (normalized)', fontsize=12)
        plt.ylabel(f'{param_names[1]} (normalized)', fontsize=12)
        plt.title('Likelihood Surface (2D slice)', fontsize=14, fontweight='bold')
        plt.savefig(output_dir / f'grid_likelihood_idx{test_idx}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Likelihood surface saved to {output_dir / f'grid_likelihood_idx{test_idx}.png'}")
        plt.close()
    
    print("\n" + "="*80)
    print("Parameter Inference Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    # Print comparison
    print("\nParameter Comparison:")
    print("-"*80)
    param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w']
    print(f"{'Parameter':<10} {'True':<12} {'Estimated':<12} {'Error':<12}")
    print("-"*80)
    for i, name in enumerate(param_names):
        true_val = params_true_denorm[0, i].cpu().numpy()
        est_val = params_est[0, i].cpu().numpy()
        error = abs(est_val - true_val)
        print(f"{name:<10} {true_val:<12.6f} {est_val:<12.6f} {error:<12.6f}")
    print("-"*80)


if __name__ == '__main__':
    main()


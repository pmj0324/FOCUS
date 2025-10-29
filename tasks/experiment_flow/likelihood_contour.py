"""
Likelihood Contour Visualization for Parameter Inference

이 스크립트는 관측된 데이터에 대해 파라미터 공간에서 likelihood contour를 그립니다.
이를 통해 어떤 파라미터 조합이 데이터를 가장 잘 설명하는지 시각적으로 확인할 수 있습니다.

Usage:
    python likelihood_contour.py --test_idx 1234
    python likelihood_contour.py --test_idx 1234 --param1 0 --param2 1 --grid_points 30
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from parameter_inference import ParameterInference


def compute_likelihood_grid_2d(
    inferencer: ParameterInference,
    x_obs: torch.Tensor,
    param_idx1: int = 0,
    param_idx2: int = 1,
    grid_points: int = 30,
    fixed_params: torch.Tensor = None
):
    """
    2D 파라미터 공간에서 likelihood를 계산합니다.
    
    Args:
        inferencer: ParameterInference 인스턴스
        x_obs: 관측 데이터 [1, 1, 256, 256]
        param_idx1: 첫 번째 파라미터 인덱스 (0-5)
        param_idx2: 두 번째 파라미터 인덱스 (0-5)
        grid_points: 각 차원당 grid point 수
        fixed_params: 고정할 파라미터 값 (None이면 0.5 사용)
        
    Returns:
        grid_0, grid_1: 파라미터 grid
        likelihoods: likelihood 값들
    """
    device = inferencer.device
    
    # Grid 생성
    range_0 = torch.linspace(0, 1, grid_points, device=device)
    range_1 = torch.linspace(0, 1, grid_points, device=device)
    grid_0, grid_1 = torch.meshgrid(range_0, range_1, indexing='ij')
    
    # 고정 파라미터 설정
    if fixed_params is None:
        fixed_params = torch.ones(1, 6, device=device) * 0.5
    
    # Likelihood 계산
    likelihoods = np.zeros((grid_points, grid_points))
    
    print(f"Computing likelihood on {grid_points}x{grid_points} grid...")
    print(f"Parameters: {get_param_name(param_idx1)} vs {get_param_name(param_idx2)}")
    
    for i in tqdm(range(grid_points)):
        for j in range(grid_points):
            params = fixed_params.clone()
            params[0, param_idx1] = grid_0[i, j]
            params[0, param_idx2] = grid_1[i, j]
            
            # Compute negative log-likelihood
            nll = inferencer.likelihood.compute_reconstruction_error(
                x_obs, params, num_samples=1
            )
            
            # Convert to log-likelihood
            likelihoods[i, j] = -nll.item()
    
    return grid_0.cpu().numpy(), grid_1.cpu().numpy(), likelihoods


def get_param_name(idx):
    """파라미터 인덱스를 이름으로 변환"""
    names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w']
    return names[idx]


def plot_likelihood_contour(
    grid_0, grid_1, likelihoods,
    param_idx1, param_idx2,
    inferencer,
    true_params=None,
    mle_params=None,
    save_path=None
):
    """
    Likelihood contour를 그립니다.
    
    Args:
        grid_0, grid_1: 파라미터 grid (normalized)
        likelihoods: Log-likelihood 값들
        param_idx1, param_idx2: 파라미터 인덱스
        inferencer: ParameterInference 인스턴스
        true_params: 실제 파라미터 (있다면) [1, 6]
        mle_params: MLE 추정 파라미터 (있다면) [1, 6]
        save_path: 저장 경로
    """
    param_name1 = get_param_name(param_idx1)
    param_name2 = get_param_name(param_idx2)
    
    # Denormalize grid for plotting
    denorm_0 = grid_0 * (inferencer.param_max[param_idx1].cpu().numpy() - 
                         inferencer.param_min[param_idx1].cpu().numpy()) + \
               inferencer.param_min[param_idx1].cpu().numpy()
    denorm_1 = grid_1 * (inferencer.param_max[param_idx2].cpu().numpy() - 
                         inferencer.param_min[param_idx2].cpu().numpy()) + \
               inferencer.param_min[param_idx2].cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contour
    levels = 20
    contour = ax.contourf(denorm_0, denorm_1, likelihoods, 
                          levels=levels, cmap='viridis', alpha=0.8)
    contour_lines = ax.contour(denorm_0, denorm_1, likelihoods, 
                               levels=levels, colors='white', 
                               alpha=0.3, linewidths=0.5)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Log-Likelihood', fontsize=14, fontweight='bold')
    
    # Plot true parameters if available
    if true_params is not None:
        true_denorm = inferencer.denormalize_params(true_params)
        ax.plot(true_denorm[0, param_idx1].cpu().numpy(),
                true_denorm[0, param_idx2].cpu().numpy(),
                'r*', markersize=20, label='True Parameters', 
                markeredgecolor='white', markeredgewidth=2)
    
    # Plot MLE result if available
    if mle_params is not None:
        ax.plot(mle_params[0, param_idx1].cpu().numpy(),
                mle_params[0, param_idx2].cpu().numpy(),
                'go', markersize=15, label='MLE Estimate',
                markeredgecolor='white', markeredgewidth=2)
    
    # Find and mark maximum likelihood point
    max_idx = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
    max_point = (denorm_0[max_idx], denorm_1[max_idx])
    ax.plot(max_point[0], max_point[1], 'y^', markersize=15, 
            label='Grid Maximum', markeredgecolor='white', markeredgewidth=2)
    
    # Labels and title
    ax.set_xlabel(param_name1, fontsize=14, fontweight='bold')
    ax.set_ylabel(param_name2, fontsize=14, fontweight='bold')
    ax.set_title(f'Likelihood Contour: {param_name1} vs {param_name2}',
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    if true_params is not None or mle_params is not None:
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Contour plot saved to {save_path}")
    
    return fig, ax


def plot_corner_likelihood(
    inferencer: ParameterInference,
    x_obs: torch.Tensor,
    grid_points: int = 20,
    param_pairs=None,
    true_params=None,
    save_path=None
):
    """
    Corner plot 스타일로 여러 파라미터 쌍의 likelihood contour를 그립니다.
    
    Args:
        inferencer: ParameterInference 인스턴스
        x_obs: 관측 데이터
        grid_points: Grid 해상도
        param_pairs: 그릴 파라미터 쌍 리스트 [(i,j), ...] (None이면 자동)
        true_params: 실제 파라미터
        save_path: 저장 경로
    """
    if param_pairs is None:
        # Default: 주요 파라미터 쌍들
        param_pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
    
    n_pairs = len(param_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w']
    
    for idx, (param_idx1, param_idx2) in enumerate(param_pairs):
        ax = axes[idx]
        
        # Compute likelihood grid
        grid_0, grid_1, likelihoods = compute_likelihood_grid_2d(
            inferencer, x_obs, param_idx1, param_idx2, grid_points
        )
        
        # Denormalize
        denorm_0 = grid_0 * (inferencer.param_max[param_idx1].cpu().numpy() - 
                             inferencer.param_min[param_idx1].cpu().numpy()) + \
                   inferencer.param_min[param_idx1].cpu().numpy()
        denorm_1 = grid_1 * (inferencer.param_max[param_idx2].cpu().numpy() - 
                             inferencer.param_min[param_idx2].cpu().numpy()) + \
                   inferencer.param_min[param_idx2].cpu().numpy()
        
        # Plot
        contour = ax.contourf(denorm_0, denorm_1, likelihoods, 
                             levels=15, cmap='viridis', alpha=0.8)
        ax.contour(denorm_0, denorm_1, likelihoods, 
                  levels=15, colors='white', alpha=0.3, linewidths=0.5)
        
        # True parameters
        if true_params is not None:
            true_denorm = inferencer.denormalize_params(true_params)
            ax.plot(true_denorm[0, param_idx1].cpu().numpy(),
                   true_denorm[0, param_idx2].cpu().numpy(),
                   'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
        
        # Maximum
        max_idx = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
        ax.plot(denorm_0[max_idx], denorm_1[max_idx], 
               'y^', markersize=12, markeredgecolor='white', markeredgewidth=1.5)
        
        # Labels
        ax.set_xlabel(param_names[param_idx1], fontsize=12, fontweight='bold')
        ax.set_ylabel(param_names[param_idx2], fontsize=12, fontweight='bold')
        ax.set_title(f'{param_names[param_idx1]} vs {param_names[param_idx2]}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        plt.colorbar(contour, ax=ax, label='Log-Likelihood')
    
    # Remove extra subplots
    for idx in range(n_pairs, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Corner plot saved to {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Likelihood Contour Visualization')
    parser.add_argument('--test_idx', type=int, default=None,
                       help='Test sample index')
    parser.add_argument('--param1', type=int, default=0,
                       help='First parameter index (0-5)')
    parser.add_argument('--param2', type=int, default=1,
                       help='Second parameter index (0-5)')
    parser.add_argument('--grid_points', type=int, default=30,
                       help='Number of grid points per dimension')
    parser.add_argument('--corner', action='store_true',
                       help='Generate corner plot with multiple parameter pairs')
    parser.add_argument('--run_mle', action='store_true',
                       help='Run MLE and show result on contour')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    exp_dir = Path(__file__).parent
    checkpoint_path = exp_dir / 'checkpoints' / 'checkpoint_best.pt'
    config_path = exp_dir / 'config.yaml'
    
    print("="*80)
    print("Likelihood Contour Visualization")
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
            print(f"Warning: test_idx {test_idx} out of range")
            test_idx = np.random.randint(0, len(maps))
    
    x_obs = torch.tensor(maps[test_idx:test_idx+1], device=device, dtype=torch.float32)
    if x_obs.ndim == 3:
        x_obs = x_obs.unsqueeze(1)
    
    params_true = torch.tensor(params[test_idx:test_idx+1], device=device, dtype=torch.float32)
    params_true_denorm = inferencer.denormalize_params(params_true)
    
    print(f"\nTest sample: {test_idx}")
    print(f"True parameters (denormalized):")
    param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w']
    for i, name in enumerate(param_names):
        print(f"  {name}: {params_true_denorm[0, i].cpu().numpy():.6f}")
    
    # Output directory
    output_dir = exp_dir / 'inference_results'
    output_dir.mkdir(exist_ok=True)
    
    # Run MLE if requested
    mle_params = None
    if args.run_mle:
        print("\nRunning MLE...")
        mle_params, _ = inferencer.infer_mle(
            x_obs,
            num_iterations=100,
            lr=0.01,
            method='reconstruction'
        )
        print(f"MLE result:")
        for i, name in enumerate(param_names):
            print(f"  {name}: {mle_params[0, i].cpu().numpy():.6f}")
    
    if args.corner:
        # Corner plot
        print("\nGenerating corner plot...")
        param_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]
        fig = plot_corner_likelihood(
            inferencer,
            x_obs,
            grid_points=args.grid_points,
            param_pairs=param_pairs,
            true_params=params_true,
            save_path=output_dir / f'likelihood_corner_idx{test_idx}.png'
        )
        plt.show()
    else:
        # Single 2D contour
        print(f"\nGenerating 2D likelihood contour...")
        print(f"Parameters: {get_param_name(args.param1)} vs {get_param_name(args.param2)}")
        
        grid_0, grid_1, likelihoods = compute_likelihood_grid_2d(
            inferencer, x_obs,
            args.param1, args.param2,
            args.grid_points
        )
        
        fig, ax = plot_likelihood_contour(
            grid_0, grid_1, likelihoods,
            args.param1, args.param2,
            inferencer,
            true_params=params_true,
            mle_params=mle_params,
            save_path=output_dir / f'likelihood_contour_{args.param1}_{args.param2}_idx{test_idx}.png'
        )
        
        plt.show()
        
        # Print statistics
        print(f"\nLikelihood Statistics:")
        print(f"  Maximum log-likelihood: {likelihoods.max():.6f}")
        print(f"  Minimum log-likelihood: {likelihoods.min():.6f}")
        print(f"  Range: {likelihoods.max() - likelihoods.min():.6f}")
    
    print("\n" + "="*80)
    print("Likelihood Contour Visualization Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()


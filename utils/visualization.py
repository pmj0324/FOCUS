"""
Visualization utilities.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def check_gaussian_distribution(samples, title="Distribution Check", save_path=None):
    """
    Check if samples follow Gaussian distribution
    
    Args:
        samples: (N, ...) tensor or numpy array
        title: plot title
        save_path: save path for plot
        
    Returns:
        stats_dict: dictionary with statistical tests
    """
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    
    samples_flat = samples.flatten()
    
    # Statistics
    mean = samples_flat.mean()
    std = samples_flat.std()
    
    # Normality test (Shapiro-Wilk for small samples)
    if len(samples_flat) < 5000:
        stat, p_value = stats.shapiro(samples_flat[:5000])
        test_name = "Shapiro-Wilk"
    else:
        subsample = np.random.choice(samples_flat, 5000, replace=False)
        stat, p_value = stats.shapiro(subsample)
        test_name = "Shapiro-Wilk (subsample)"
    
    # Anderson-Darling test
    anderson_result = stats.anderson(samples_flat[:5000] if len(samples_flat) > 5000 else samples_flat)
    
    print(f"\n{title}")
    print("=" * 60)
    print(f"Mean: {mean:.6f}")
    print(f"Std:  {std:.6f}")
    print(f"\n{test_name} test:")
    print(f"  Statistic: {stat:.6f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Gaussian? {'Yes (p > 0.05)' if p_value > 0.05 else 'No (p ≤ 0.05)'}")
    
    print(f"\nAnderson-Darling test:")
    print(f"  Statistic: {anderson_result.statistic:.6f}")
    print(f"  Critical values: {anderson_result.critical_values}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    axes[0].hist(samples_flat, bins=100, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay Gaussian
    x = np.linspace(samples_flat.min(), samples_flat.max(), 100)
    gaussian = stats.norm.pdf(x, mean, std)
    axes[0].plot(x, gaussian, 'r-', linewidth=2, label=f'N({mean:.2f}, {std:.2f}²)')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Histogram vs Gaussian')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(samples_flat, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    # ECDF
    sorted_samples = np.sort(samples_flat)
    ecdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    theoretical_cdf = stats.norm.cdf(sorted_samples, mean, std)
    
    axes[2].plot(sorted_samples, ecdf, label='Empirical CDF')
    axes[2].plot(sorted_samples, theoretical_cdf, label='Theoretical CDF', linestyle='--')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('CDF')
    axes[2].set_title('Empirical vs Theoretical CDF')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'mean': mean,
        'std': std,
        'test_name': test_name,
        'test_statistic': stat,
        'p_value': p_value,
        'anderson_statistic': anderson_result.statistic,
        'anderson_critical_values': anderson_result.critical_values,
        'is_gaussian': p_value > 0.05
    }


def visualize_samples(samples, params=None, save_path=None, num_display=8):
    """
    Visualize generated samples
    
    Args:
        samples: (N, 1, H, W) or (N, H, W) generated samples
        params: (N, 6) optional parameters
        save_path: save path
        num_display: number of samples to display
    """
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    
    if samples.ndim == 4:
        samples = samples[:, 0]  # Remove channel dimension
    
    num_samples = min(len(samples), num_display)
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(num_samples * 2, 4))
    axes = axes.flatten()
    
    for i in range(num_samples):
        im = axes[i].imshow(samples[i], cmap='viridis')
        axes[i].axis('off')
        
        if params is not None:
            param_str = ', '.join([f'{p:.3f}' for p in params[i]])
            axes[i].set_title(f'Sample {i+1}\n[{param_str}]', fontsize=8)
        else:
            axes[i].set_title(f'Sample {i+1}')
        
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


"""
Power spectrum utilities.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_power_spectrum(field, box_size=75.0):
    """
    Compute 2D power spectrum (physically correct implementation)
    
    Args:
        field: (H, W) 2D field
        box_size: box size in Mpc/h
        
    Returns:
        k, Pk: wave numbers and power spectrum
    """
    if torch.is_tensor(field):
        field = field.cpu().numpy()
    
    # 1. DC 제거 (평균 빼기)
    field_mean = field.mean()
    field_centered = field - field_mean
    
    # 2. FFT 및 shift
    fft = np.fft.fft2(field_centered)
    fft_shift = np.fft.fftshift(fft)
    
    # 3. Power spectrum with normalization
    H, W = field.shape
    power_2d = np.abs(fft_shift) ** 2 / (H * W)
    
    # 4. k-grid 생성 (물리적 단위)
    kx_freq = np.fft.fftfreq(W, d=box_size/W) * 2 * np.pi
    ky_freq = np.fft.fftfreq(H, d=box_size/H) * 2 * np.pi
    kx_shift = np.fft.fftshift(kx_freq)
    ky_shift = np.fft.fftshift(ky_freq)
    
    kx_grid, ky_grid = np.meshgrid(kx_shift, ky_shift)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # 5. Radial averaging
    k_flat = k_grid.flatten()
    power_flat = power_2d.flatten()
    
    k_min = k_flat[k_flat > 0].min()
    k_max = k_flat.max()
    
    # Log bins (우주론적 관례)
    n_bins = 30
    k_bins_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_centers = (k_bins_edges[:-1] + k_bins_edges[1:]) / 2
    
    Pk = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (k_flat >= k_bins_edges[i]) & (k_flat < k_bins_edges[i+1])
        if mask.sum() > 0:
            Pk[i] = power_flat[mask].mean()
    
    return k_centers, Pk


def compare_power_spectra(real_samples, generated_samples, save_path=None):
    """
    Compare power spectra of real and generated data
    
    Args:
        real_samples: (N, H, W) real samples
        generated_samples: (N, H, W) generated samples
        save_path: save path
    """
    if torch.is_tensor(real_samples):
        real_samples = real_samples.cpu().numpy()
    if torch.is_tensor(generated_samples):
        generated_samples = generated_samples.cpu().numpy()
    
    # Average power spectrum over multiple samples
    num_samples = min(len(real_samples), len(generated_samples), 10)
    
    real_pks = []
    gen_pks = []
    
    for i in range(num_samples):
        k, pk_real = compute_power_spectrum(real_samples[i])
        _, pk_gen = compute_power_spectrum(generated_samples[i])
        real_pks.append(pk_real)
        gen_pks.append(pk_gen)
    
    real_pk_mean = np.mean(real_pks, axis=0)
    real_pk_std = np.std(real_pks, axis=0)
    gen_pk_mean = np.mean(gen_pks, axis=0)
    gen_pk_std = np.std(gen_pks, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(k, real_pk_mean, 'b-', label='Real', linewidth=2)
    plt.fill_between(k, real_pk_mean - real_pk_std, real_pk_mean + real_pk_std, 
                     alpha=0.3, color='blue')
    
    plt.plot(k, gen_pk_mean, 'r--', label='Generated', linewidth=2)
    plt.fill_between(k, gen_pk_mean - gen_pk_std, gen_pk_mean + gen_pk_std, 
                     alpha=0.3, color='red')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k (Mpc/h)$^{-1}$')
    plt.ylabel('P(k)')
    plt.title('Power Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


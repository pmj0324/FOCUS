"""
원본 데이터 시각화 및 파워스펙트럼
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_power_spectrum_2d(field, box_size=75.0):
    """
    2D 파워 스펙트럼 계산 (물리적으로 정확한 구현)
    
    Args:
        field: (H, W) 2D field
        box_size: box size in Mpc/h
        
    Returns:
        k, Pk: wave numbers and power spectrum
    """
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
    n_bins = 50
    k_bins_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_centers = (k_bins_edges[:-1] + k_bins_edges[1:]) / 2
    
    Pk = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (k_flat >= k_bins_edges[i]) & (k_flat < k_bins_edges[i+1])
        if mask.sum() > 0:
            Pk[i] = power_flat[mask].mean()
    
    return k_centers, Pk


def visualize_samples_with_power_spectrum(num_samples=6):
    """
    여러 샘플의 이미지와 파워스펙트럼을 함께 시각화
    """
    print("=" * 80)
    print("원본 Dark Matter Maps 시각화 + 파워스펙트럼")
    print("=" * 80)
    
    # 데이터 로드
    data_path = "/Users/pmj0324/Sicence/cosmo/Data/2D/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
    maps = np.load(data_path)
    
    print(f"데이터 shape: {maps.shape}")
    print(f"데이터 범위: [{maps.min():.2e}, {maps.max():.2e}]\n")
    
    # 랜덤하게 샘플 선택
    np.random.seed(42)
    indices = np.random.choice(len(maps), num_samples, replace=False)
    
    # 시각화
    fig = plt.figure(figsize=(20, num_samples * 3))
    
    for i, idx in enumerate(indices):
        sample = maps[idx]
        
        # 로그 스케일 변환 (시각화용)
        sample_log = np.log10(sample + 1e-10)
        
        # 파워 스펙트럼 계산
        k, Pk = compute_power_spectrum_2d(sample)
        
        # 왼쪽: 이미지
        ax_img = plt.subplot(num_samples, 2, 2*i + 1)
        im = ax_img.imshow(sample_log, cmap='viridis', origin='lower')
        ax_img.set_title(f'Sample {idx} - Dark Matter Density (log scale)', fontsize=12)
        ax_img.set_xlabel('x [pixels]')
        ax_img.set_ylabel('y [pixels]')
        cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        cbar.set_label(r'log$_{10}$(Density)', rotation=270, labelpad=20)
        
        # 통계 정보 추가
        text = f'Min: {sample.min():.2e}\nMax: {sample.max():.2e}\nMean: {sample.mean():.2e}'
        ax_img.text(0.02, 0.98, text, transform=ax_img.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 오른쪽: 파워 스펙트럼
        ax_ps = plt.subplot(num_samples, 2, 2*i + 2)
        ax_ps.loglog(k, Pk, 'b-', linewidth=2, alpha=0.8)
        ax_ps.set_xlabel(r'k [Mpc$^{-1}$h]', fontsize=11)
        ax_ps.set_ylabel(r'P(k) [arbitrary units]', fontsize=11)
        ax_ps.set_title(f'Power Spectrum - Sample {idx}', fontsize=12)
        ax_ps.grid(True, alpha=0.3, which='both')
        
        # k 범위 표시
        ax_ps.text(0.05, 0.95, f'k range: [{k.min():.2e}, {k.max():.2e}]',
                  transform=ax_ps.transAxes, fontsize=9,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = '/Users/pmj0324/Sicence/cosmo/New/original_data_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 저장 완료: {save_path}\n")
    plt.close()


def visualize_average_power_spectrum(num_samples=100):
    """
    여러 샘플의 평균 파워스펙트럼
    """
    print("=" * 80)
    print("평균 파워 스펙트럼 계산 중...")
    print("=" * 80)
    
    # 데이터 로드
    data_path = "/Users/pmj0324/Sicence/cosmo/Data/2D/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
    maps = np.load(data_path)
    
    # 랜덤 샘플 선택
    np.random.seed(42)
    indices = np.random.choice(len(maps), min(num_samples, len(maps)), replace=False)
    
    # 파워 스펙트럼 계산
    power_spectra = []
    
    for i, idx in enumerate(indices):
        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{len(indices)}")
        
        k, Pk = compute_power_spectrum_2d(maps[idx])
        power_spectra.append(Pk)
    
    power_spectra = np.array(power_spectra)
    
    # 평균 및 표준편차
    Pk_mean = power_spectra.mean(axis=0)
    Pk_std = power_spectra.std(axis=0)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 왼쪽: 대표 샘플
    sample = maps[indices[0]]
    sample_log = np.log10(sample + 1e-10)
    
    im = axes[0].imshow(sample_log, cmap='viridis', origin='lower')
    axes[0].set_title('Representative Dark Matter Map (log scale)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x [pixels]', fontsize=12)
    axes[0].set_ylabel('y [pixels]', fontsize=12)
    cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label(r'log$_{10}$(Density)', rotation=270, labelpad=20, fontsize=11)
    
    # 오른쪽: 평균 파워 스펙트럼
    axes[1].loglog(k, Pk_mean, 'b-', linewidth=2.5, label=f'Mean (n={len(indices)})')
    axes[1].fill_between(k, Pk_mean - Pk_std, Pk_mean + Pk_std, 
                         alpha=0.3, color='blue', label='±1σ')
    
    axes[1].set_xlabel(r'k [Mpc$^{-1}$h]', fontsize=13, fontweight='bold')
    axes[1].set_ylabel(r'P(k) [arbitrary units]', fontsize=13, fontweight='bold')
    axes[1].set_title(f'Average Power Spectrum', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=11, loc='best')
    
    # 통계 정보
    info_text = f'Samples: {len(indices)}\n'
    info_text += f'k range: [{k.min():.2e}, {k.max():.2e}]\n'
    info_text += f'Box size: 75 Mpc/h\n'
    info_text += f'Resolution: 256×256'
    
    axes[1].text(0.05, 0.05, info_text,
                transform=axes[1].transAxes, fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    save_path = '/Users/pmj0324/Sicence/cosmo/New/average_power_spectrum.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 저장 완료: {save_path}\n")
    plt.close()
    
    return k, Pk_mean, Pk_std


def main():
    """메인 함수"""
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Dark Matter 데이터 시각화" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # 1. 개별 샘플 시각화
    visualize_samples_with_power_spectrum(num_samples=6)
    
    # 2. 평균 파워 스펙트럼
    k, Pk_mean, Pk_std = visualize_average_power_spectrum(num_samples=100)
    
    print("=" * 80)
    print("완료!")
    print("=" * 80)
    print("생성된 파일:")
    print("  1. original_data_visualization.png - 개별 샘플 (6개)")
    print("  2. average_power_spectrum.png - 평균 파워스펙트럼 (100개)")
    print("=" * 80)


if __name__ == "__main__":
    main()


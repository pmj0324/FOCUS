"""
하나의 샘플에 대한 Forward Diffusion 과정 시각화
- 각 timestep에서 이미지가 어떻게 변하는지
- 각 timestep에서 파워스펙트럼이 어떻게 변하는지
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('/Users/pmj0324/Sicence/cosmo/New')

from diffusion import GaussianDiffusion


def compute_power_spectrum_2d(field, box_size=75.0):
    """2D 파워 스펙트럼 계산"""
    # FFT
    field_mean = field.mean()
    field_centered = field - field_mean
    
    fft = np.fft.fft2(field_centered)
    power = np.abs(fft) ** 2
    
    # Normalize
    H, W = field.shape
    power = power / (H * W)
    
    # Wave numbers
    kx = np.fft.fftfreq(W, d=box_size/W) * 2 * np.pi
    ky = np.fft.fftfreq(H, d=box_size/H) * 2 * np.pi
    
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Bin power spectrum
    k_bins = np.logspace(np.log10(k_grid[k_grid > 0].min()), 
                         np.log10(k_grid.max()), 
                         50)
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    
    Pk = np.zeros(len(k_centers))
    
    for i in range(len(k_centers)):
        mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i+1])
        if mask.sum() > 0:
            Pk[i] = power[mask].mean()
    
    return k_centers, Pk


def visualize_single_event_diffusion():
    """
    하나의 이벤트(샘플)에 대한 Forward Diffusion 과정 시각화
    """
    print("=" * 80)
    print("단일 이벤트 Forward Diffusion 과정 시각화")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # 1. 데이터 로드
    print("1. 데이터 로딩...")
    data_path = "/Users/pmj0324/Sicence/cosmo/Data/2D/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
    maps = np.load(data_path)
    
    # 흥미로운 샘플 선택 (분산이 큰 것)
    variances = [maps[i].std() for i in range(min(100, len(maps)))]
    selected_idx = np.argmax(variances)
    
    sample = maps[selected_idx]
    print(f"   선택된 샘플 인덱스: {selected_idx}")
    print(f"   원본 값 범위: [{sample.min():.2e}, {sample.max():.2e}]")
    
    # 2. 정규화
    print("\n2. 정규화...")
    sample_log = np.log10(sample + 1e-10)
    sample_min, sample_max = sample_log.min(), sample_log.max()
    sample_normalized = 2 * (sample_log - sample_min) / (sample_max - sample_min) - 1
    
    x_start = torch.FloatTensor(sample_normalized).unsqueeze(0).unsqueeze(0).to(device)
    print(f"   정규화 범위: [{sample_normalized.min():.4f}, {sample_normalized.max():.4f}]")
    
    # 3. Forward Diffusion 설정
    print("\n3. Forward Diffusion 적용 중...")
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # 더 많은 timesteps 체크
    timesteps_to_check = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    
    # 데이터 저장
    images = []
    power_spectra = []
    
    for t_val in timesteps_to_check:
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        x_noisy = diffusion.q_sample(x_start, t)
        
        img = x_noisy[0, 0].cpu().numpy()
        images.append(img)
        
        # 파워 스펙트럼 계산
        k, Pk = compute_power_spectrum_2d(img)
        power_spectra.append(Pk)
        
        print(f"   t={t_val:4d}: mean={img.mean():7.4f}, std={img.std():7.4f}")
    
    power_spectra = np.array(power_spectra)
    
    # 4. 시각화 1: 이미지 진화
    print("\n4. 시각화 생성 중...")
    print("   4-1. 이미지 진화...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, (t_val, img) in enumerate(zip(timesteps_to_check, images)):
        ax = axes[i]
        im = ax.imshow(img, cmap='viridis', vmin=-3, vmax=3, origin='lower')
        ax.set_title(f't = {t_val}\nμ={img.mean():.3f}, σ={img.std():.3f}', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('x [pixels]', fontsize=10)
        ax.set_ylabel('y [pixels]', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if t_val == 0:
            ax.add_patch(plt.Rectangle((5, 5), 50, 20, 
                        fill=False, edgecolor='red', linewidth=2))
            ax.text(10, 15, 'Original', color='red', fontsize=10, fontweight='bold')
        elif t_val == 999:
            ax.add_patch(plt.Rectangle((5, 5), 50, 20, 
                        fill=False, edgecolor='lime', linewidth=2))
            ax.text(10, 15, 'Gaussian', color='lime', fontsize=10, fontweight='bold')
    
    plt.suptitle('Forward Diffusion Process: Image Evolution', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_dir = Path('/Users/pmj0324/Sicence/cosmo/New/plots/01_forward_diffusion_process')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / 'image_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 저장: {save_path}")
    plt.close()
    
    # 5. 시각화 2: 파워 스펙트럼 진화
    print("   4-2. 파워 스펙트럼 진화...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, (t_val, Pk) in enumerate(zip(timesteps_to_check, power_spectra)):
        ax = axes[i]
        ax.loglog(k, Pk, 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel(r'k [Mpc$^{-1}$h]', fontsize=11)
        ax.set_ylabel(r'P(k)', fontsize=11)
        ax.set_title(f't = {t_val}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([1e-8, 1e2])
        
        if t_val == 0:
            ax.text(0.05, 0.95, 'Original\nStructure', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        elif t_val == 999:
            ax.text(0.05, 0.95, 'White\nNoise', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
    
    plt.suptitle('Forward Diffusion Process: Power Spectrum Evolution', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = output_dir / 'power_spectrum_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 저장: {save_path}")
    plt.close()
    
    # 6. 시각화 3: 결합 (이미지 + 파워스펙트럼)
    print("   4-3. 결합 시각화...")
    
    # 선택된 timesteps (6개만)
    selected_steps = [0, 100, 300, 500, 800, 999]
    selected_indices = [timesteps_to_check.index(t) for t in selected_steps]
    
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    
    for row, idx in enumerate(selected_indices):
        t_val = timesteps_to_check[idx]
        img = images[idx]
        Pk = power_spectra[idx]
        
        # 왼쪽: 이미지
        ax_img = axes[row, 0]
        im = ax_img.imshow(img, cmap='viridis', vmin=-3, vmax=3, origin='lower')
        ax_img.set_title(f't = {t_val} | μ={img.mean():.3f}, σ={img.std():.3f}', 
                        fontsize=13, fontweight='bold')
        ax_img.set_xlabel('x [pixels]', fontsize=11)
        ax_img.set_ylabel('y [pixels]', fontsize=11)
        cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        
        # 진행도 표시
        progress = t_val / 999 * 100
        ax_img.text(0.02, 0.98, f'Progress: {progress:.0f}%', 
                   transform=ax_img.transAxes, fontsize=11,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 오른쪽: 파워 스펙트럼
        ax_ps = axes[row, 1]
        ax_ps.loglog(k, Pk, 'b-', linewidth=2.5, alpha=0.8)
        ax_ps.set_xlabel(r'k [Mpc$^{-1}$h]', fontsize=11, fontweight='bold')
        ax_ps.set_ylabel(r'P(k)', fontsize=11, fontweight='bold')
        ax_ps.set_title(f'Power Spectrum at t = {t_val}', fontsize=13, fontweight='bold')
        ax_ps.grid(True, alpha=0.3, which='both')
        ax_ps.set_ylim([1e-8, 1e2])
        
        # 상태 표시
        if t_val == 0:
            state_text = 'Original Dark Matter Map\nCosmological Structure'
            color = 'yellow'
        elif t_val == 999:
            state_text = 'Pure Gaussian Noise\nN(0, 1)'
            color = 'lime'
        else:
            state_text = f'Partially Noised\n{progress:.0f}% → Gaussian'
            color = 'orange'
        
        ax_ps.text(0.05, 0.95, state_text, 
                  transform=ax_ps.transAxes, fontsize=10,
                  verticalalignment='top', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.suptitle('Forward Diffusion: Single Event Analysis\nImage & Power Spectrum Evolution', 
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    save_path = output_dir / 'combined_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 저장: {save_path}")
    plt.close()
    
    # 7. 시각화 4: 파워 스펙트럼 시간 진화 (하나의 그래프)
    print("   4-4. 파워 스펙트럼 시간 진화 (통합)...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 컬러맵
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps_to_check)))
    
    for i, (t_val, Pk) in enumerate(zip(timesteps_to_check, power_spectra)):
        alpha = 0.4 if i not in selected_indices else 1.0
        linewidth = 1.5 if i not in selected_indices else 3.0
        label = f't={t_val}' if i in selected_indices else None
        
        ax.loglog(k, Pk, color=colors[i], linewidth=linewidth, 
                 alpha=alpha, label=label)
    
    ax.set_xlabel(r'k [Mpc$^{-1}$h]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'P(k)', fontsize=14, fontweight='bold')
    ax.set_title('Power Spectrum Evolution During Forward Diffusion', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # 화살표로 시간 방향 표시
    ax.annotate('', xy=(0.7, 0.3), xytext=(0.7, 0.7),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(0.72, 0.5, 'Time →\nAdding Noise', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           color='red', verticalalignment='center')
    
    plt.tight_layout()
    
    save_path = output_dir / 'power_spectrum_time_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 저장: {save_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)
    print(f"출력 폴더: {output_dir}")
    print("\n생성된 파일:")
    print("  1. image_evolution.png - 12 timesteps 이미지 진화")
    print("  2. power_spectrum_evolution.png - 12 timesteps 파워스펙트럼")
    print("  3. combined_evolution.png - 6 timesteps 결합 (이미지+스펙트럼)")
    print("  4. power_spectrum_time_evolution.png - 전체 진화 한눈에")
    print("=" * 80)


if __name__ == "__main__":
    visualize_single_event_diffusion()


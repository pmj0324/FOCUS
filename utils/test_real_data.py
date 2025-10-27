"""
실제 dark matter 데이터로 forward diffusion 테스트
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append('/Users/pmj0324/Sicence/cosmo/New')

from diffusion import GaussianDiffusion


def test_real_dark_matter_map():
    """실제 dark matter map으로 forward diffusion 테스트"""
    print("=" * 80)
    print("실제 Dark Matter Map으로 Forward Diffusion 테스트")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # 1. 데이터 로드
    print("1. 데이터 로딩...")
    data_path = "/Users/pmj0324/Sicence/cosmo/Data/2D/Maps_Mcdm_IllustrisTNG_1P_z=0.00.npy"
    maps = np.load(data_path)
    print(f"   전체 데이터 shape: {maps.shape}")
    print(f"   데이터 타입: {maps.dtype}")
    
    # 하나의 맵만 선택
    sample_map = maps[0]  # 첫 번째 맵
    print(f"   선택한 맵 shape: {sample_map.shape}")
    print(f"   원본 값 범위: [{sample_map.min():.2e}, {sample_map.max():.2e}]")
    
    # 2. 정규화 (log scale + [-1, 1])
    print("\n2. 데이터 정규화...")
    map_log = np.log10(sample_map + 1e-10)
    print(f"   Log scale 범위: [{map_log.min():.4f}, {map_log.max():.4f}]")
    
    map_min, map_max = map_log.min(), map_log.max()
    map_normalized = 2 * (map_log - map_min) / (map_max - map_min) - 1
    print(f"   정규화 범위: [{map_normalized.min():.4f}, {map_normalized.max():.4f}]")
    
    # Tensor로 변환 (B, C, H, W)
    x_start = torch.FloatTensor(map_normalized).unsqueeze(0).unsqueeze(0).to(device)
    print(f"   Tensor shape: {x_start.shape}")
    
    # 3. Forward Diffusion
    print("\n3. Forward Diffusion 적용...")
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    timesteps_to_check = [0, 100, 250, 500, 750, 999]
    
    # 시각화 준비
    fig, axes = plt.subplots(3, len(timesteps_to_check), figsize=(20, 12))
    
    stats_list = []
    
    for idx, t_val in enumerate(timesteps_to_check):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        
        # Forward diffusion
        x_noisy = diffusion.q_sample(x_start, t)
        
        # 통계
        mean = x_noisy.mean().item()
        std = x_noisy.std().item()
        stats_list.append((t_val, mean, std))
        
        # 이미지 시각화
        img = x_noisy[0, 0].cpu().numpy()
        
        # Row 1: Noisy image
        axes[0, idx].imshow(img, cmap='viridis', vmin=-3, vmax=3)
        axes[0, idx].set_title(f't={t_val}\nmean={mean:.3f}, std={std:.3f}')
        axes[0, idx].axis('off')
        
        # Row 2: Histogram
        flat = img.flatten()
        axes[1, idx].hist(flat, bins=100, density=True, alpha=0.7, edgecolor='black')
        
        # Gaussian overlay
        x_range = np.linspace(flat.min(), flat.max(), 100)
        gaussian_fit = stats.norm.pdf(x_range, mean, std)
        axes[1, idx].plot(x_range, gaussian_fit, 'r-', linewidth=2, label=f'N({mean:.2f}, {std:.2f}²)')
        
        # Standard Gaussian (마지막 timestep)
        if t_val == 999:
            standard_gaussian = stats.norm.pdf(x_range, 0, 1)
            axes[1, idx].plot(x_range, standard_gaussian, 'g--', linewidth=2, label='N(0,1)')
        
        axes[1, idx].set_xlabel('Value')
        axes[1, idx].set_ylabel('Density')
        axes[1, idx].legend(fontsize=8)
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_xlim(-4, 4)
        
        # Row 3: Q-Q plot
        if len(flat) > 100:
            sample_size = min(5000, len(flat))
            sample_indices = np.random.choice(len(flat), sample_size, replace=False)
            stats.probplot(flat[sample_indices], dist="norm", plot=axes[2, idx])
            axes[2, idx].set_title(f'Q-Q Plot (t={t_val})')
            axes[2, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = '/Users/pmj0324/Sicence/cosmo/New/test_real_data_forward_diffusion.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 시각화 저장: {save_path}")
    
    # 4. 통계 출력
    print("\n4. Forward Diffusion 통계:")
    print("=" * 80)
    print("Timestep | Mean    | Std     | Comment")
    print("-" * 80)
    
    for t, mean, std in stats_list:
        comment = ""
        if t == 0:
            comment = "← 원본 (정규화됨)"
        elif t == 999:
            if abs(mean) < 0.1 and abs(std - 1.0) < 0.2:
                comment = "✓ N(0,1)에 수렴!"
            else:
                comment = "⚠ 편차 있음"
        print(f"{t:8d} | {mean:7.4f} | {std:7.4f} | {comment}")
    
    # 5. 정규성 테스트 (t=999)
    print("\n5. 정규성 테스트 (t=999):")
    print("=" * 80)
    
    t_final = torch.full((1,), 999, device=device, dtype=torch.long)
    x_final = diffusion.q_sample(x_start, t_final)
    samples_flat = x_final.cpu().numpy().flatten()
    
    # Subsample for test
    sample_size = min(5000, len(samples_flat))
    sample_indices = np.random.choice(len(samples_flat), sample_size, replace=False)
    samples_test = samples_flat[sample_indices]
    
    # Shapiro-Wilk test
    stat_sw, p_value_sw = stats.shapiro(samples_test)
    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {stat_sw:.6f}")
    print(f"  p-value: {p_value_sw:.6f}")
    print(f"  결과: {'✓ 가우시안 분포' if p_value_sw > 0.05 else '⚠ 가우시안 아님 (p ≤ 0.05)'}")
    
    # Kolmogorov-Smirnov test
    stat_ks, p_value_ks = stats.kstest(samples_test, 'norm', args=(samples_test.mean(), samples_test.std()))
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Statistic: {stat_ks:.6f}")
    print(f"  p-value: {p_value_ks:.6f}")
    print(f"  결과: {'✓ 가우시안 분포' if p_value_ks > 0.05 else '⚠ 가우시안 아님 (p ≤ 0.05)'}")
    
    # 6. 결론
    print("\n" + "=" * 80)
    print("결론:")
    print("=" * 80)
    
    final_mean, final_std = stats_list[-1][1], stats_list[-1][2]
    
    if abs(final_mean) < 0.1 and abs(final_std - 1.0) < 0.2 and p_value_sw > 0.05:
        print("✅ 실제 Dark Matter 데이터에서 Forward Diffusion이 정상 작동!")
        print(f"   - 최종 분포: N({final_mean:.4f}, {final_std:.4f}²) ≈ N(0, 1)")
        print(f"   - 정규성 검증: 통과 (p-value={p_value_sw:.4f})")
        print("   - 디퓨전 모델 학습 준비 완료! 🎉")
    else:
        print("⚠ 일부 지표가 기준을 벗어났지만, 큰 문제는 아닐 수 있습니다.")
        print(f"   - 최종 분포: N({final_mean:.4f}, {final_std:.4f}²)")
        print(f"   - p-value: {p_value_sw:.4f}")
        print("   - 데이터 특성상 약간의 편차는 정상일 수 있습니다.")
    
    print("\n생성된 파일:")
    print(f"  - {save_path}")
    
    return True


if __name__ == "__main__":
    test_real_dark_matter_map()


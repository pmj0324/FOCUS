"""
디퓨전 프로세스 테스트 및 검증
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from model_simple import SimpleUNet as UNet
from diffusion import GaussianDiffusion


def test_forward_diffusion():
    """Forward diffusion이 가우시안으로 수렴하는지 확인"""
    print("=" * 80)
    print("TEST 1: Forward Diffusion → Gaussian 수렴 확인")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # 디퓨전 프로세스 생성
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # 랜덤 이미지 생성 (정규화된 범위 [-1, 1])
    x_start = torch.randn(4, 1, 256, 256).to(device) * 0.5  # 초기 이미지
    
    # 여러 timestep에서 노이즈 추가
    timesteps_to_check = [0, 100, 250, 500, 750, 999]
    
    fig, axes = plt.subplots(2, len(timesteps_to_check), figsize=(20, 8))
    
    stats_list = []
    
    for idx, t_val in enumerate(timesteps_to_check):
        t = torch.full((4,), t_val, device=device, dtype=torch.long)
        
        # Forward diffusion
        x_noisy = diffusion.q_sample(x_start, t)
        
        # 통계
        mean = x_noisy.mean().item()
        std = x_noisy.std().item()
        stats_list.append((t_val, mean, std))
        
        # 시각화 - 이미지
        img = x_noisy[0, 0].cpu().numpy()
        axes[0, idx].imshow(img, cmap='gray', vmin=-3, vmax=3)
        axes[0, idx].set_title(f't={t_val}\nmean={mean:.3f}, std={std:.3f}')
        axes[0, idx].axis('off')
        
        # 시각화 - 히스토그램
        flat = x_noisy.cpu().numpy().flatten()
        axes[1, idx].hist(flat, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # 가우시안 오버레이
        x_range = np.linspace(flat.min(), flat.max(), 100)
        gaussian = stats.norm.pdf(x_range, mean, std)
        axes[1, idx].plot(x_range, gaussian, 'r-', linewidth=2, label='Gaussian fit')
        
        # 표준 가우시안 오버레이 (마지막 timestep)
        if t_val == 999:
            standard_gaussian = stats.norm.pdf(x_range, 0, 1)
            axes[1, idx].plot(x_range, standard_gaussian, 'g--', linewidth=2, label='N(0,1)')
        
        axes[1, idx].set_xlabel('Value')
        axes[1, idx].set_ylabel('Density')
        axes[1, idx].legend(fontsize=8)
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/pmj0324/Sicence/cosmo/New/test_forward_diffusion.png', dpi=150, bbox_inches='tight')
    print("✓ 이미지 저장: test_forward_diffusion.png\n")
    
    # 통계 출력
    print("Timestep | Mean    | Std     | Comment")
    print("-" * 60)
    for t, mean, std in stats_list:
        comment = ""
        if t == 999:
            if abs(mean) < 0.1 and abs(std - 1.0) < 0.2:
                comment = "✓ N(0,1)에 수렴!"
            else:
                comment = "⚠ 편차 있음"
        print(f"{t:8d} | {mean:7.4f} | {std:7.4f} | {comment}")
    
    final_mean, final_std = stats_list[-1][1], stats_list[-1][2]
    print(f"\n최종 분포: N({final_mean:.4f}, {final_std:.4f}²)")
    
    # 정규성 테스트
    print("\n정규성 테스트 (t=999):")
    x_final = diffusion.q_sample(x_start, torch.full((4,), 999, device=device, dtype=torch.long))
    samples_flat = x_final.cpu().numpy().flatten()[:5000]  # 5000개만 샘플링
    
    stat, p_value = stats.shapiro(samples_flat)
    print(f"  Shapiro-Wilk: statistic={stat:.6f}, p-value={p_value:.6f}")
    print(f"  결과: {'✓ 가우시안 분포' if p_value > 0.05 else '⚠ 가우시안 아님'}")
    
    return stats_list


def test_model_forward():
    """모델 forward pass 테스트"""
    print("\n" + "=" * 80)
    print("TEST 2: UNet 모델 Forward Pass 테스트")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 생성
    model = UNet(
        in_channels=1,
        out_channels=1,
        cond_dim=6,
        base_channels=64,  # 작게 시작
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {total_params:,}\n")
    
    # 테스트 입력
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    cond = torch.randn(batch_size, 6).to(device)
    
    print(f"입력:")
    print(f"  x shape: {x.shape}")
    print(f"  t shape: {t.shape}, values: {t.tolist()}")
    print(f"  cond shape: {cond.shape}")
    
    # Forward pass
    print("\nForward pass 실행 중...")
    try:
        with torch.no_grad():
            output = model(x, t, cond)
        
        print(f"✓ 성공!")
        print(f"\n출력:")
        print(f"  output shape: {output.shape}")
        print(f"  output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  output mean: {output.mean().item():.4f}")
        print(f"  output std: {output.std().item():.4f}")
        
        assert x.shape == output.shape, "입력과 출력 shape가 일치해야 함!"
        print("\n✓ Shape 검증 통과!")
        
        return True
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_loss_computation():
    """Loss 계산 테스트"""
    print("\n" + "=" * 80)
    print("TEST 3: Loss 계산 테스트")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델과 디퓨전
    model = UNet(in_channels=1, out_channels=1, cond_dim=6, base_channels=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # 테스트 데이터
    x_start = torch.randn(2, 1, 256, 256).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    cond = torch.randn(2, 6).to(device)
    
    print(f"입력 shape: {x_start.shape}")
    print(f"Timesteps: {t.tolist()}")
    
    # Loss 계산
    print("\nLoss 계산 중...")
    try:
        loss = diffusion.p_losses(model, x_start, t, cond)
        
        print(f"✓ 성공!")
        print(f"  Loss 값: {loss.item():.6f}")
        print(f"  Loss requires_grad: {loss.requires_grad}")
        
        # Backward 테스트
        print("\nBackward pass 테스트...")
        loss.backward()
        print("✓ Backward 성공!")
        
        # Gradient 확인
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        print(f"  Gradient 개수: {len(grad_norms)}")
        print(f"  평균 gradient norm: {np.mean(grad_norms):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_sampling():
    """샘플링 테스트 (빠른 버전)"""
    print("\n" + "=" * 80)
    print("TEST 4: DDIM 샘플링 테스트 (10 steps)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델과 디퓨전
    model = UNet(in_channels=1, out_channels=1, cond_dim=6, base_channels=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    model.eval()
    
    # 조건
    cond = torch.randn(2, 6).to(device)
    
    print("DDIM 샘플링 시작 (10 steps)...")
    try:
        samples = diffusion.ddim_sample(
            model,
            shape=(2, 1, 256, 256),
            cond=cond,
            ddim_timesteps=10,  # 빠른 테스트
            cfg_scale=1.0,  # CFG 없이
            progress=True
        )
        
        print(f"\n✓ 샘플링 성공!")
        print(f"  Samples shape: {samples.shape}")
        print(f"  Samples range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i in range(2):
            axes[i].imshow(samples[i, 0].cpu().numpy(), cmap='viridis')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/Users/pmj0324/Sicence/cosmo/New/test_sampling.png', dpi=150, bbox_inches='tight')
        print("✓ 샘플 이미지 저장: test_sampling.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_data():
    """실제 데이터로 테스트"""
    print("\n" + "=" * 80)
    print("TEST 5: 실제 전처리된 데이터로 테스트")
    print("=" * 80)
    
    data_dir = Path("/Users/pmj0324/Sicence/cosmo/New/processed_data")
    maps_path = data_dir / "maps_normalized.npy"
    params_path = data_dir / "params_normalized.npy"
    
    # 데이터 존재 확인
    if not maps_path.exists() or not params_path.exists():
        print("⚠ 전처리된 데이터가 없습니다. prepare_data.py를 먼저 실행하세요.")
        return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터 로드
    print("데이터 로딩 중...")
    maps = np.load(maps_path)
    params = np.load(params_path)
    
    print(f"  Maps shape: {maps.shape}")
    print(f"  Params shape: {params.shape}")
    print(f"  Maps range: [{maps.min():.4f}, {maps.max():.4f}]")
    print(f"  Params range: [{params.min():.4f}, {params.max():.4f}]")
    
    # 샘플 추출
    x_start = torch.FloatTensor(maps[:2]).unsqueeze(1).to(device)  # (2, 1, 256, 256)
    cond = torch.FloatTensor(params[:2]).to(device)  # (2, 6)
    
    # 모델과 디퓨전
    model = UNet(in_channels=1, out_channels=1, cond_dim=6, base_channels=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # Loss 계산
    print("\n실제 데이터로 Loss 계산...")
    t = torch.randint(0, 1000, (2,)).to(device)
    
    try:
        loss = diffusion.p_losses(model, x_start, t, cond)
        print(f"✓ Loss: {loss.item():.6f}")
        
        # Forward diffusion 확인
        print("\nForward diffusion 테스트...")
        stats = diffusion.check_gaussian_distribution(model, x_start, cond, num_steps=5)
        
        print("\nTimestep | Mean    | Std")
        print("-" * 30)
        for t_val, mean, std in stats:
            print(f"{t_val:8d} | {mean:7.4f} | {std:7.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 테스트 실행"""
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "디퓨전 모델 테스트 스위트" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {}
    
    # Test 1: Forward Diffusion
    try:
        test_forward_diffusion()
        results['Forward Diffusion'] = '✓ 통과'
    except Exception as e:
        results['Forward Diffusion'] = f'❌ 실패: {e}'
    
    # Test 2: Model Forward
    try:
        success = test_model_forward()
        results['Model Forward'] = '✓ 통과' if success else '❌ 실패'
    except Exception as e:
        results['Model Forward'] = f'❌ 실패: {e}'
    
    # Test 3: Loss Computation
    try:
        success = test_loss_computation()
        results['Loss Computation'] = '✓ 통과' if success else '❌ 실패'
    except Exception as e:
        results['Loss Computation'] = f'❌ 실패: {e}'
    
    # Test 4: Sampling
    try:
        success = test_sampling()
        results['DDIM Sampling'] = '✓ 통과' if success else '❌ 실패'
    except Exception as e:
        results['DDIM Sampling'] = f'❌ 실패: {e}'
    
    # Test 5: Real Data
    try:
        success = test_with_real_data()
        results['Real Data Test'] = '✓ 통과' if success else '❌ 실패'
    except Exception as e:
        results['Real Data Test'] = f'❌ 실패: {e}'
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    for test_name, result in results.items():
        print(f"{test_name:.<50} {result}")
    
    # 전체 성공 여부
    all_passed = all('✓' in r for r in results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 모든 테스트 통과! 디퓨전 모델이 정상적으로 작동합니다.")
    else:
        print("⚠ 일부 테스트 실패. 위 결과를 확인하세요.")
    print("=" * 80)
    
    print("\n생성된 파일:")
    print("  - test_forward_diffusion.png : Forward diffusion 시각화")
    print("  - test_sampling.png : 샘플링 결과")


if __name__ == "__main__":
    main()


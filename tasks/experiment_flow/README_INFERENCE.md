# Parameter Inference with Flow Matching

이 폴더는 Flow Matching 모델을 사용한 **파라미터 추론(Parameter Inference)** 코드를 포함합니다.

## 개요

`parameter_inference.py` 스크립트는 학습된 Flow Matching 모델을 사용하여, 관측된 데이터(2D dark matter maps)로부터 우주론적 파라미터를 추정합니다.

### 주요 기능

1. **Likelihood 계산**: Flow Matching 모델의 확률 흐름(probability flow)을 이용한 likelihood 추정
2. **Maximum Likelihood Estimation (MLE)**: Gradient 기반 최적화를 통한 파라미터 추정
3. **Grid Search**: 파라미터 공간에서 격자 탐색
4. **MCMC Sampling**: Metropolis-Hastings 알고리즘을 통한 베이지안 추론

## 사용 방법

### 1. 기본 실행

```bash
cd /home/work/Cosmology/FOCUS/tasks/experiment_flow
python parameter_inference.py
```

이 명령어는 다음을 수행합니다:
- 학습된 모델(`checkpoints/checkpoint_best.pt`)을 로드
- 테스트 데이터에서 랜덤하게 하나의 샘플을 선택
- MLE를 통해 파라미터 추정
- 결과를 `inference_results/` 폴더에 저장

### 2. 코드 구조

#### `FlowMatchingLikelihood` 클래스
Flow Matching 모델의 likelihood를 계산하는 클래스입니다.

**주요 메서드:**
- `compute_nll_approximate()`: 근사 negative log-likelihood 계산
- `compute_reconstruction_error()`: 재구성 오류 기반 likelihood 계산 (gradient 계산 가능)

#### `ParameterInference` 클래스
파라미터 추론을 위한 메인 클래스입니다.

**주요 메서드:**
- `infer_mle()`: Maximum Likelihood Estimation
- `infer_grid_search()`: 격자 탐색
- `infer_mcmc()`: MCMC 샘플링
- `visualize_results()`: 결과 시각화

### 3. 커스터마이징

#### 자신의 데이터로 추론하기

```python
from parameter_inference import ParameterInference
import torch
import numpy as np

# Initialize
inferencer = ParameterInference(
    checkpoint_path='checkpoints/checkpoint_best.pt',
    config_path='config.yaml',
    device='cuda'
)

# Load your observed data
x_obs = torch.tensor(your_data, device='cuda', dtype=torch.float32)
# Shape: [1, 1, 256, 256] (batch, channels, height, width)

# Run MLE
params_mle, losses = inferencer.infer_mle(
    x_obs,
    num_iterations=200,
    lr=0.01,
    method='reconstruction'
)

print(f"Estimated parameters: {params_mle}")
```

#### MLE 파라미터 조정

```python
params_mle, losses = inferencer.infer_mle(
    x_obs,
    num_iterations=500,      # 더 많은 반복
    lr=0.005,                # 더 작은 learning rate
    method='reconstruction'  # 또는 'nll'
)
```

#### Grid Search 실행

```python
params_grid, likelihoods = inferencer.infer_grid_search(
    x_obs,
    grid_points=20  # 각 파라미터당 20개 포인트
)
```

#### MCMC Sampling 실행

```python
samples, acc_rate = inferencer.infer_mcmc(
    x_obs,
    num_samples=2000,    # MCMC 샘플 수
    burn_in=200,         # Burn-in 기간
    proposal_std=0.05    # Proposal 분포의 표준편차
)

# Posterior 분석
print(f"Posterior mean: {samples.mean(axis=0)}")
print(f"Posterior std: {samples.std(axis=0)}")
```

### 4. 결과 해석

#### 출력 파라미터

추정된 파라미터는 **denormalized** 값입니다:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | Ωm | Matter density parameter |
| 1 | Ωb | Baryon density parameter |
| 2 | h | Hubble parameter |
| 3 | ns | Scalar spectral index |
| 4 | σ8 | Amplitude of matter fluctuations |
| 5 | w | Dark energy equation of state |

#### 출력 파일

- `inference_mle_results.png`: MLE 추정 결과 시각화
  - 관측 데이터
  - 추정된 파라미터로 생성한 샘플 4개
  - 추정된 파라미터 vs 실제 파라미터 (있는 경우)

- `mle_loss_history.png`: MLE 최적화 과정의 loss 변화

## 구현 세부사항

### Likelihood 계산 방법

Flow Matching에서 likelihood는 이론적으로 다음과 같이 계산됩니다:

```
log p(x_0) = log p(x_1) - ∫_0^1 ∇·v(x_t, t) dt
```

하지만 이는 계산이 매우 비싸므로, 본 구현에서는 **근사 방법**을 사용합니다:

1. **재구성 오류 (Reconstruction Error)**: 
   - 여러 시간 t에서 vector field 예측 오류를 측정
   - Gradient 계산 가능 (MLE에 사용)

2. **완전한 샘플링 기반**: 
   - 조건부 분포에서 샘플을 생성하고 관측 데이터와 비교
   - 더 정확하지만 gradient 계산 불가 (MCMC에 사용)

### 최적화 알고리즘

**MLE (Maximum Likelihood Estimation)**:
- Adam optimizer 사용
- Learning rate: 0.01 (기본값)
- Parameter bounds: [0, 1] (normalized space)
- 200 iterations (기본값)

**MCMC (Metropolis-Hastings)**:
- Gaussian proposal distribution
- Proposal std: 0.05 (기본값)
- Acceptance rate: 보통 20-40%가 적절

## 성능 및 제한사항

### 계산 비용

- **MLE (200 iterations)**: ~35초 (GPU)
- **Grid Search (20x20)**: ~5분 (GPU)
- **MCMC (1000 samples)**: ~10분 (GPU)

### 제한사항

1. **근사 Likelihood**: 정확한 likelihood가 아닌 근사치 사용
2. **Local Minima**: MLE는 local minima에 빠질 수 있음 (여러 초기값 시도 권장)
3. **고차원 Grid Search**: 6차원 전체 탐색은 계산적으로 불가능 (2D subspace만 탐색)
4. **MCMC Convergence**: Burn-in과 mixing이 충분한지 확인 필요

## 예제 결과

```
Test sample index: 2485
True parameters (denormalized): [0.434, 0.661, 0.537, 2.856, 0.878, 0.617]
Estimated parameters (denormalized): [0.410, 0.608, 1.186, 3.457, 1.333, 2.003]
Final loss: 0.075826
```

일부 파라미터(특히 Ωm, Ωb)는 잘 추정되지만, 다른 파라미터(h, ns, σ8, w)는 더 어렵습니다. 이는:
1. 모델이 일부 파라미터에 더 민감하게 학습됨
2. 데이터가 일부 파라미터 조합에 대해 degeneracy를 가질 수 있음

## 개선 방향

1. **더 정확한 Likelihood 계산**: 
   - Exact divergence 계산 (Hutchinson estimator)
   - Neural ODE를 사용한 정확한 probability flow

2. **Amortized Inference**:
   - 별도의 inference network 학습
   - 한 번의 forward pass로 posterior 추정

3. **Variational Inference**:
   - Variational posterior approximation
   - 더 빠르고 확장 가능한 추론

4. **Ensemble Methods**:
   - 여러 초기값에서 MLE 실행
   - 결과 평균 또는 가중 평균

## 참고자료

- Flow Matching: [Lipman et al., 2023](https://arxiv.org/abs/2210.02747)
- Simulation-Based Inference: [Cranmer et al., 2020](https://www.pnas.org/doi/10.1073/pnas.1912789117)
- Neural Posterior Estimation: [Papamakarios et al., 2019](https://arxiv.org/abs/1905.07488)

## 문의

이슈나 질문이 있으시면 GitHub Issues를 통해 문의해주세요.





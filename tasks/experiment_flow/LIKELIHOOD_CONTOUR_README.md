# Likelihood Contour Visualization

실제 관측 이미지가 주어졌을 때, 파라미터 공간에서 **likelihood contour**를 그려 어떤 파라미터 조합이 데이터를 가장 잘 설명하는지 시각화합니다.

## 🎯 기능

1. **2D Likelihood Contour**: 두 파라미터에 대한 likelihood surface
2. **Corner Plot**: 여러 파라미터 쌍에 대한 contour 동시 표시
3. **True Value 표시**: 실제 파라미터 값 (알고 있는 경우)
4. **MLE Result 표시**: Maximum likelihood 위치
5. **Grid Maximum**: Grid search로 찾은 최대 likelihood 지점

## 🚀 사용 방법

### 1. 기본 2D Contour

```bash
# 기본: Ωm vs Ωb
python likelihood_contour.py --test_idx 1234

# 특정 파라미터 쌍
python likelihood_contour.py --test_idx 1234 --param1 0 --param2 2
# param1=0: Ωm, param2=2: h

# 더 높은 해상도
python likelihood_contour.py --test_idx 1234 --grid_points 50
```

### 2. Corner Plot (여러 파라미터 쌍)

```bash
# 6개 파라미터 쌍을 한번에
python likelihood_contour.py --test_idx 1234 --corner --grid_points 15
```

### 3. MLE와 함께 표시

```bash
# MLE를 먼저 실행하고 결과를 contour에 표시
python likelihood_contour.py --test_idx 1234 --run_mle --grid_points 25
```

## 📊 파라미터 인덱스

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | Ωm | Matter density parameter |
| 1 | Ωb | Baryon density parameter |
| 2 | h | Hubble parameter |
| 3 | ns | Scalar spectral index |
| 4 | σ8 | Amplitude of fluctuations |
| 5 | w | Dark energy equation of state |

## 💡 사용 예시

### 예시 1: Ωm vs Ωb contour

```bash
python likelihood_contour.py --test_idx 1234 --param1 0 --param2 1 --grid_points 30
```

**결과:**
- `likelihood_contour_0_1_idx1234.png`
- Ωm vs Ωb 파라미터 공간의 likelihood landscape
- 빨간 별: 실제 파라미터 값
- 노란 삼각형: Grid에서 찾은 maximum

### 예시 2: Corner plot

```bash
python likelihood_contour.py --test_idx 1234 --corner --grid_points 20
```

**결과:**
- `likelihood_corner_idx1234.png`
- 6개 파라미터 쌍의 likelihood contour
- 각 subplot: 다른 파라미터 조합
- 전체적인 likelihood 구조 파악 가능

### 예시 3: MLE 결과와 비교

```bash
python likelihood_contour.py --test_idx 1234 --param1 0 --param2 1 --run_mle --grid_points 30
```

**결과:**
- 빨간 별: 실제 파라미터
- 녹색 원: MLE로 찾은 파라미터
- 노란 삼각형: Grid search maximum
- 세 위치의 차이를 시각적으로 비교

## 🎨 시각화 요소

### Contour Plot
- **색상**: Likelihood 값 (진한 색 = 높은 likelihood)
- **등고선**: 같은 likelihood 값을 가지는 점들
- **빨간 별 (★)**: 실제 파라미터 값 (ground truth)
- **녹색 원 (●)**: MLE 추정값 (--run_mle 사용 시)
- **노란 삼각형 (▲)**: Grid maximum (가장 높은 likelihood)

### Corner Plot
- 여러 파라미터 쌍에 대한 likelihood contour
- 각 subplot은 독립적인 2D slice
- 다른 파라미터는 0.5 (중간값)로 고정

## ⚙️ 설정 옵션

### Grid 해상도

```bash
--grid_points N    # 각 차원당 N개 포인트 (기본: 30)
```

**추천:**
- 빠른 테스트: `--grid_points 15` (~3분)
- 일반 사용: `--grid_points 25` (~10분)
- 고해상도: `--grid_points 40` (~30분)
- 출판용: `--grid_points 50` (~50분)

### 계산 시간

| Grid Size | Single Contour | Corner Plot (6 pairs) |
|-----------|----------------|----------------------|
| 15×15 | ~3분 | ~18분 |
| 25×25 | ~10분 | ~60분 |
| 30×30 | ~15분 | ~90분 |
| 50×50 | ~45분 | ~270분 |

## 📈 해석 방법

### 1. Likelihood Peak
- **높은 peak**: 이 파라미터 조합이 데이터를 잘 설명
- **넓은 peak**: 파라미터 불확실성 높음
- **좁은 peak**: 파라미터가 잘 제약됨

### 2. Degeneracy
- **대각선 구조**: 두 파라미터가 상관관계
- **수직/수평**: 파라미터들이 독립적
- **복잡한 구조**: 비선형 관계

### 3. True vs Estimated
- **별과 삼각형이 가까움**: 추정 성공
- **별과 삼각형이 멀음**: 추정 실패 또는 degeneracy
- **Peak이 별 근처**: 모델이 데이터 잘 설명

## 🔬 Python API

```python
from likelihood_contour import (
    compute_likelihood_grid_2d,
    plot_likelihood_contour,
    plot_corner_likelihood
)
from parameter_inference import ParameterInference

# Initialize
inferencer = ParameterInference(
    checkpoint_path='checkpoints/checkpoint_best.pt',
    config_path='config.yaml',
    device='cuda'
)

# Your observed data
x_obs = torch.tensor(your_data, device='cuda')  # [1, 1, 256, 256]
true_params = torch.tensor([[0.3, 0.8, 1.0, 2.8, 1.2, 1.5]], device='cuda')

# Compute likelihood grid
grid_0, grid_1, likelihoods = compute_likelihood_grid_2d(
    inferencer,
    x_obs,
    param_idx1=0,  # Ωm
    param_idx2=1,  # Ωb
    grid_points=30
)

# Plot contour
fig, ax = plot_likelihood_contour(
    grid_0, grid_1, likelihoods,
    param_idx1=0,
    param_idx2=1,
    inferencer=inferencer,
    true_params=true_params,
    save_path='my_contour.png'
)

# Or corner plot
fig = plot_corner_likelihood(
    inferencer,
    x_obs,
    grid_points=20,
    param_pairs=[(0,1), (0,2), (1,2), (2,3)],
    true_params=true_params,
    save_path='my_corner.png'
)
```

## 💡 팁과 트릭

### 1. 빠른 탐색
먼저 낮은 해상도로 전체 구조 파악:
```bash
python likelihood_contour.py --test_idx 1234 --corner --grid_points 10
```

### 2. 관심 영역 확대
특정 파라미터 쌍만 고해상도로:
```bash
python likelihood_contour.py --test_idx 1234 --param1 0 --param2 1 --grid_points 50
```

### 3. 여러 샘플 비교
```bash
for idx in 100 200 300 400 500; do
    python likelihood_contour.py --test_idx $idx --grid_points 20
done
```

### 4. MLE 검증
MLE가 제대로 작동하는지 확인:
```bash
python likelihood_contour.py --test_idx 1234 --run_mle --grid_points 30
```
녹색 원(MLE)이 노란 삼각형(grid max) 근처에 있어야 함

## 🔍 예상 결과

### 성공적인 경우
- Likelihood peak이 뚜렷함
- True value (빨간 별)가 peak 근처
- MLE result (녹색 원)가 peak 근처
- Grid maximum (노란 삼각형)이 peak에 위치

### 어려운 경우
- Flat likelihood (여러 파라미터 조합이 비슷)
- Multiple peaks (여러 국소 최대값)
- Degeneracy (대각선 ridge 구조)

## 🎓 고급 사용

### 커스텀 Fixed Parameters

```python
# 특정 파라미터를 고정하고 싶을 때
fixed_params = torch.tensor([[0.3, 0.8, 1.0, 2.8, 1.2, 1.5]], device='cuda')

grid_0, grid_1, likelihoods = compute_likelihood_grid_2d(
    inferencer,
    x_obs,
    param_idx1=0,
    param_idx2=1,
    grid_points=30,
    fixed_params=fixed_params  # 다른 파라미터를 이 값으로 고정
)
```

### 더 많은 파라미터 쌍

```python
# 모든 가능한 쌍 (15개)
param_pairs = []
for i in range(6):
    for j in range(i+1, 6):
        param_pairs.append((i, j))

fig = plot_corner_likelihood(
    inferencer, x_obs,
    grid_points=15,
    param_pairs=param_pairs,
    save_path='full_corner.png'
)
```

## 📁 출력 파일

### Single Contour
- `likelihood_contour_{param1}_{param2}_idx{N}.png`
- 예: `likelihood_contour_0_1_idx1234.png`

### Corner Plot
- `likelihood_corner_idx{N}.png`
- 예: `likelihood_corner_idx1234.png`

## 🐛 문제 해결

### 계산이 너무 느림
```bash
# Grid 크기 줄이기
--grid_points 15

# CPU 사용 (느리지만 메모리 적게 사용)
--device cpu
```

### Contour가 너무 평평함
- Likelihood 차이가 작음 → 모델이 파라미터 변화에 둔감
- Grid 범위를 조정하거나 더 민감한 파라미터 쌍 선택

### Memory Error
```bash
# 배치 크기 줄이기 또는 CPU 사용
--device cpu
```

## 📚 관련 자료

- 원본 논문: Flow Matching (Lipman et al., 2023)
- Parameter Inference: `README_INFERENCE.md`
- Technical Details: `PARAMETER_INFERENCE_SUMMARY.md`

---

**Happy Visualizing!** 🎨📊





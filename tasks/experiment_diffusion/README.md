# Experiment: Diffusion Model

표준 Diffusion 기반 우주론적 필드 생성 실험

## 개요

이 실험은 **DDPM/DDIM** 방법을 사용하여 우주론적 Dark Matter density field를 생성합니다.

### 주요 특징

- **Method**: Diffusion (확률적 노이즈 제거)
- **Model**: SimpleUNet (128 base channels)
- **Training**: 노이즈 예측
- **Sampling**: DDIM (50 steps)
- **Parameters**: 146M (Experiment 02와 동일한 모델 크기)

## Diffusion이란?

**노이즈 추가→제거** 과정을 학습하는 생성 모델:
- 노이즈 ε(x_t, t) 예측
- 확률적 경로로 데이터 생성
- 검증된 방법 (DDPM, DDIM)

## 실행 방법

### 방법 1: 스크립트 사용
```bash
cd /home/work/Cosmology/FOCUS/tasks/experiment_diffusion
./run_model.sh
```

### 방법 2: 직접 실행
```bash
cd /home/work/Cosmology/FOCUS
python train.py \
  --config tasks/experiment_diffusion/config.yaml \
  --exp_dir tasks/experiment_diffusion
```

## 설정 파일

`config.yaml`의 핵심 설정:

```yaml
method: "diffusion"  # Diffusion 사용 (또는 생략 가능, 기본값)

model:
  from: "SimpleUNet"  # 표준 Diffusion U-Net
  base_channels: 128
  time_dim: 256

diffusion:
  timesteps: 1000  # 학습 timesteps
  beta_start: 1.0e-4
  beta_end: 0.02
  schedule: "linear"

sampling:
  method: "ddim"  # 빠른 샘플링
  ddim_timesteps: 50  # 샘플링 steps
```

## 출력 결과

### 디렉토리 구조
```
experiment_diffusion/
├── config.yaml
├── run_model.sh
├── checkpoints/
│   ├── checkpoint_best.pt
│   ├── checkpoint_last.pt
│   └── training_history.png
├── figs/
│   ├── samples_epoch_XXXX.png
│   └── power_spectrum_epoch_XXXX.png
└── logs/
```

### 학습 정보 출력

```
================================================================================
Epoch 1/200 [WARMUP]
================================================================================
  Train Loss: 0.1234
  Val Loss:   0.1456
  LR:         2.00e-05
  Time:       120.5s (2.0m)
  Plateau:    0/2 bad epochs
  Early Stop: 0/10 patience
  ✓ New best model!
```

## 평가 지표

- **Train/Val Loss**: Diffusion MSE loss
- **Power Spectrum**: 우주론적 파워 스펙트럼 비교
- **샘플 품질**: 시각적 비교

## Flow Matching과 비교

| 항목 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 샘플링 속도 | 느림 (50-1000 steps) | ⚡⚡ 빠름 (50 steps) |
| 학습 목적 | 노이즈 ε | 벡터 필드 v |
| 시간 | 이산 T=1000 | 연속 t∈[0,1] |
| 경로 | 확률적 | 직선 |
| 검증도 | ⭐⭐⭐ 높음 | ⭐⭐ 중간 |

## 팁

1. **Warmup 기간**: 처음 5 epoch은 learning rate warmup
2. **Plateau Scheduler**: 2 epoch 동안 개선 없으면 LR 감소
3. **Early Stopping**: 10 epoch 동안 개선 없으면 학습 중단
4. **DDIM**: DDPM보다 빠르고 deterministic

## 문제 해결

### GPU 메모리 부족
```yaml
training:
  batch_size: 32  # 64에서 32로 감소
```

### 학습이 너무 느림
- `num_workers` 조정
- Mixed precision training 사용 고려

### 샘플 품질 개선
```yaml
sampling:
  method: "ddim"
  ddim_timesteps: 100  # 더 많은 steps
  eta: 0.0  # Deterministic (기본값)
```

### 노이즈 스케줄 조정
```yaml
diffusion:
  schedule: "cosine"  # Linear 대신 cosine
```

## Diffusion 스케줄

### Linear (기본)
- 균등한 노이즈 증가
- 빠른 학습

### Cosine
- 더 부드러운 노이즈 곡선
- 더 나은 샘플 품질 (가능성)

```yaml
diffusion:
  schedule: "cosine"
```

## 참고 자료

- DDPM 논문: Ho et al. (2020)
- DDIM 논문: Song et al. (2021)
- FOCUS 문서: `/home/work/Cosmology/FOCUS/METHOD_SELECTION.md`
- 기본 설정: `/home/work/Cosmology/FOCUS/configs/example_diffusion.yaml`






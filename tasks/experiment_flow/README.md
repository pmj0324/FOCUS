# Experiment: Flow Matching

Flow Matching 기반 우주론적 필드 생성 실험

## 개요

이 실험은 **Flow Matching** 방법을 사용하여 우주론적 Dark Matter density field를 생성합니다.

### 주요 특징

- **Method**: Flow Matching (연속 시간 ODE)
- **Model**: FlowUNet (128 base channels)
- **Training**: Vector field 예측
- **Sampling**: Euler 방법 (50 steps)
- **Parameters**: 146M (Experiment 02와 동일한 모델 크기)

## Flow Matching이란?

Diffusion 대신 **연속적인 흐름(flow)**를 사용하는 생성 모델:
- 벡터 필드 v(x_t, t) 학습
- 직선 경로로 노이즈→데이터 변환
- 빠른 샘플링 (10-50 steps)

## 실행 방법

### 방법 1: 스크립트 사용
```bash
cd /home/work/Cosmology/FOCUS/tasks/experiment_flow
./run_model.sh
```

### 방법 2: 직접 실행
```bash
cd /home/work/Cosmology/FOCUS
python train.py \
  --config tasks/experiment_flow/config.yaml \
  --exp_dir tasks/experiment_flow
```

## 설정 파일

`config.yaml`의 핵심 설정:

```yaml
method: "flow"  # Flow Matching 사용

model:
  from: "FlowUNet"  # Flow 전용 모델
  base_channels: 128
  time_dim: 256

flow_matching:
  sigma_min: 0.0
  sigma_max: 1.0

sampling:
  method: "euler"  # ODE 적분 방법
  num_steps: 50
```

## 출력 결과

### 디렉토리 구조
```
experiment_flow/
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

- **Train/Val Loss**: Flow Matching loss
- **Power Spectrum**: 우주론적 파워 스펙트럼 비교
- **샘플 품질**: 시각적 비교

## Diffusion과 비교

| 항목 | Flow Matching | Diffusion |
|------|--------------|-----------|
| 샘플링 속도 | ⚡⚡ 빠름 (50 steps) | 느림 (50-1000 steps) |
| 학습 목적 | 벡터 필드 v | 노이즈 ε |
| 시간 | 연속 t∈[0,1] | 이산 T=1000 |
| 경로 | 직선 | 확률적 |

## 팁

1. **Warmup 기간**: 처음 5 epoch은 learning rate warmup
2. **Plateau Scheduler**: 2 epoch 동안 개선 없으면 LR 감소
3. **Early Stopping**: 10 epoch 동안 개선 없으면 학습 중단
4. **CFG Scale**: 2.0이 좋은 품질 제공

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
  method: "heun"  # 2차 정확도 (더 느리지만 더 좋음)
  num_steps: 100  # 더 많은 steps
```

## 참고 자료

- Flow Matching 논문: Lipman et al. (2023)
- FOCUS 문서: `/home/work/Cosmology/FOCUS/METHOD_SELECTION.md`
- Flow Matching 상세: `/home/work/Cosmology/FOCUS/flowmatching/README.md`


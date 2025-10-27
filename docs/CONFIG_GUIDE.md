# FOCUS: Configuration Guide

## 📝 YAML 설정 파일 구조

### 1. 데이터 설정 (Data)

```yaml
data:
  data_dir: "processed"  # 'processed' 단축어 사용 가능!
  # 또는 전체 경로:
  # maps_path: "./processed_data/maps_normalized.npy"
  # params_path: "./processed_data/params_normalized.npy"
  
  train_split: 0.9       # Train/Val 비율
  num_workers: 4         # 데이터 로딩 워커
  shuffle: true          # train/val random split (true/false)
```

### 2. 모델 설정 (Model) - 모듈화된 구조

```yaml
model:
  from: "models.unet.SimpleUNet"  # Import 경로
  # 모델별 인자
  in_channels: 1
  out_channels: 1
  cond_dim: 6
  base_channels: 64
  channel_mults: [1, 2, 4, 8]
  time_dim: 256
```

**새 모델 추가 예시:**
```yaml
# DiT 모델을 만들었을 때
model:
  from: "models.dit.DiT"
  hidden_size: 384
  depth: 12
  num_heads: 6
```

### 3. 디퓨전 설정 (Diffusion)

```yaml
diffusion:
  timesteps: 1000
  beta_start: 1.0e-4
  beta_end: 0.02
  schedule: "linear"  # linear, cosine, quadratic
```

### 4. 학습 설정 (Training)

#### 기본 옵션
```yaml
training:
  batch_size: 2
  num_epochs: 200
  lr: 1.0e-4
  weight_decay: 1.0e-4
  cfg_prob: 0.1
  sample_every: 10
  gradient_clip: 1.0
```

#### Optimizer 선택
```yaml
training:
  optimizer: "adamw"  # adamw, adam, sgd
```

#### Scheduler 선택

**Plateau (기본값)**
```yaml
training:
  scheduler:
    name: "plateau"
    factor: 0.5      # Learning rate를 50% 감소
    patience: 3      # 3 epoch 기다림
    min_lr: 1.0e-7   # 최소 learning rate
```

**Cosine Annealing**
```yaml
training:
  scheduler:
    name: "cosine"
    T_max: 200       # Cosine period
    eta_min: 1.0e-6  # 최소 learning rate
```

**Step LR**
```yaml
training:
  scheduler:
    name: "step"
    step_size: 50    # 50 epoch마다
    gamma: 0.1       # 10% 감소
```

**Scheduler 사용 안함**
```yaml
training:
  scheduler:
    name: "none"  # 또는 scheduler: null
```

### 5. 샘플링 설정 (Sampling)

```yaml
sampling:
  method: "ddim"  # ddim or ddpm
  ddim_timesteps: 50
  cfg_scale: 2.0
  eta: 0.0
```

### 6. 하드웨어 설정

```yaml
device: "cuda"  # cuda or cpu
```

## 🎯 완전한 예시

### 기본 설정
```yaml
data:
  data_dir: "processed"
  train_split: 0.9
  num_workers: 4
  shuffle: true

model:
  from: "models.unet.SimpleUNet"
  in_channels: 1
  out_channels: 1
  cond_dim: 6
  base_channels: 64
  channel_mults: [1, 2, 4, 8]
  time_dim: 256

diffusion:
  timesteps: 1000
  beta_start: 1.0e-4
  beta_end: 0.02
  schedule: "linear"

training:
  batch_size: 2
  num_epochs: 200
  lr: 1.0e-4
  weight_decay: 1.0e-4
  cfg_prob: 0.1
  sample_every: 10
  gradient_clip: 1.0
  optimizer: "adamw"
  
  scheduler:
    name: "plateau"
    factor: 0.5
    patience: 3
    min_lr: 1.0e-7

sampling:
  method: "ddim"
  ddim_timesteps: 50
  cfg_scale: 2.0
  eta: 0.0

device: "cuda"
```

### DiT 모델 예시 (미래)
```yaml
data:
  data_dir: "processed"
  shuffle: false  # Sequential split

model:
  from: "models.dit.DiT"
  hidden_size: 384
  depth: 12
  num_heads: 6

training:
  batch_size: 4
  num_epochs: 500
  optimizer: "adamw"
  
  scheduler:
    name: "cosine"
    T_max: 500
    eta_min: 1.0e-6
```

## 🚀 사용 방법

### 1. 새 실험 생성
```bash
mkdir -p tasks/my_exp/{checkpoints,logs,figs}
cp configs/default.yaml tasks/my_exp/config.yaml
```

### 2. 설정 수정
```bash
nano tasks/my_exp/config.yaml
# 또는 에디터로 수정
```

### 3. 학습 실행
```bash
python train.py --config tasks/my_exp/config.yaml --exp_dir tasks/my_exp
```

## 💡 Tips

1. **`processed` 단축어**: `data_dir: "processed"` 로 간단히 설정 가능
2. **모델 교체**: `from: "models.새모델.클래스명"` 으로 쉽게 교체
3. **Scheduler 선택**: 성능에 따라 적절히 선택
   - Plateau: 안정적, 수렴 확인 가능
   - Cosine: 긴 학습에 좋음
   - Step: 단순한 감소
4. **Shuffle 옵션**: `shuffle: false` 로 sequential split 가능


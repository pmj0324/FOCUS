# ✅ Data Augmentation 시스템 완성!

## 🎯 구현된 기능

### 1. **2D 이미지 회전** (RandomRotation90)
- 90°, 180°, 270° 회전 (픽셀 정보 손실 없음)
- 우주론적으로 타당: 우주는 등방성(isotropic)

### 2. **이미지 뒤집기** (RandomFlip)
- 수평 뒤집기 (Horizontal flip)
- 수직 뒤집기 (Vertical flip)
- 우주론적으로 타당: Parity symmetry

### 3. **통합 Augmentation** (CosmologyAugmentation)
- 회전 + 뒤집기 조합
- 확률 기반 적용
- Training/Validation 구분

## 📁 생성된 파일

```
FOCUS/
├── dataloaders/
│   ├── augmentation.py              # 🆕 Augmentation 모듈
│   └── cosmology_dataset.py         # 업데이트된 Dataset
├── visualize_augmentation.py        # 🆕 시각화 스크립트
└── augmentation_vis/                # 🆕 시각화 결과
    ├── augmentation_comparison_idx100.png
    ├── rotation_examples_idx100.png
    └── flip_examples_idx100.png
```

## 🚀 사용 방법

### 1. 기본 사용

```python
from dataloaders.cosmology_dataset import create_dataloaders

# Augmentation 활성화
train_loader, val_loader = create_dataloaders(
    maps_path='processed_data/maps_normalized.npy',
    params_path='processed_data/params_normalized.npy',
    batch_size=16,
    use_augmentation=True,  # 🆕 Augmentation 활성화!
    augmentation_config={
        'rotation_p': 0.75,  # 75% 확률로 회전
        'flip_p': 0.5        # 50% 확률로 뒤집기
    }
)
```

### 2. 커스텀 Augmentation

```python
from dataloaders.augmentation import CosmologyAugmentation

# 커스텀 설정
augmentation = CosmologyAugmentation(
    rotation_p=0.8,        # 더 높은 회전 확률
    horizontal_flip_p=0.3, # 낮은 수평 뒤집기 확률
    vertical_flip_p=0.7,   # 높은 수직 뒤집기 확률
    use_rotation=True,
    use_flip=True
)

# Dataset에 적용
dataset = CosmologyDataset(
    maps_path='data.npy',
    params_path='params.npy',
    augmentation=augmentation
)
```

### 3. 시각화

```bash
# 기본 시각화
python visualize_augmentation.py

# 커스텀 설정
python visualize_augmentation.py \
    --sample_idx 100 \
    --num_augs 8 \
    --rotation_p 0.8 \
    --flip_p 0.6
```

## 🎨 시각화 결과

생성된 이미지들:
- **`augmentation_comparison_idx100.png`**: 원본 vs 7개 augmented 버전
- **`rotation_examples_idx100.png`**: 0°, 90°, 180°, 270° 회전
- **`flip_examples_idx100.png`**: 원본, 수평, 수직, 둘 다 뒤집기

## 💡 물리적 타당성

### 왜 이 Augmentation이 적절한가?

1. **회전 (Rotation)**:
   - 우주는 등방성(isotropic) → 방향에 무관
   - 90° 배수 회전 → 픽셀 정보 손실 없음
   - 실제 관측에서도 방향이 무작위

2. **뒤집기 (Flip)**:
   - Parity symmetry → 좌우/상하 대칭
   - 우주 구조는 좌표계에 무관
   - 실제 관측에서도 방향성 없음

3. **조합**:
   - 회전 + 뒤집기 = 더 다양한 변형
   - 데이터 다양성 증가
   - 과적합 방지

## ⚙️ 설정 옵션

### CosmologyAugmentation 파라미터

```python
augmentation = CosmologyAugmentation(
    rotation_p=0.75,        # 회전 확률 (0.0-1.0)
    horizontal_flip_p=0.5, # 수평 뒤집기 확률
    vertical_flip_p=0.5,   # 수직 뒤집기 확률
    use_rotation=True,     # 회전 사용 여부
    use_flip=True          # 뒤집기 사용 여부
)
```

### 추천 설정

| 용도 | rotation_p | flip_p | 설명 |
|------|------------|--------|------|
| 기본 | 0.75 | 0.5 | 균형잡힌 augmentation |
| 강한 augmentation | 0.9 | 0.7 | 더 많은 변형 |
| 약한 augmentation | 0.5 | 0.3 | 보수적 접근 |
| 회전만 | 0.8 | 0.0 | 회전만 사용 |
| 뒤집기만 | 0.0 | 0.6 | 뒤집기만 사용 |

## 🔧 Training에 적용

### Flow Matching Training

```python
# config.yaml에 추가
training:
  batch_size: 16
  num_epochs: 200
  lr: 1.0e-4
  use_augmentation: true  # 🆕
  augmentation:
    rotation_p: 0.75
    flip_p: 0.5

# 또는 코드에서 직접
train_loader, val_loader = create_dataloaders(
    maps_path=config['data']['maps_path'],
    params_path=config['data']['params_path'],
    batch_size=config['training']['batch_size'],
    use_augmentation=True,
    augmentation_config=config['training'].get('augmentation', {})
)
```

## 📊 성능 향상 예상

### Data Augmentation 효과

1. **데이터 다양성 증가**:
   - 원본 데이터의 8배 변형 가능
   - 과적합 방지
   - 일반화 성능 향상

2. **Training 안정성**:
   - 더 robust한 feature 학습
   - 방향성 bias 제거
   - 모델 견고성 향상

3. **Parameter Inference 개선**:
   - 다양한 관측 각도에 robust
   - 실제 관측과 유사한 조건
   - 추정 정확도 향상

## 🎓 사용 예시

### 예시 1: 기본 Training

```python
# 기존 코드
train_loader, val_loader = create_dataloaders(
    maps_path='data.npy',
    params_path='params.npy',
    batch_size=16
)

# Augmentation 추가
train_loader, val_loader = create_dataloaders(
    maps_path='data.npy',
    params_path='params.npy',
    batch_size=16,
    use_augmentation=True,  # 이것만 추가!
    augmentation_config={'rotation_p': 0.75, 'flip_p': 0.5}
)
```

### 예시 2: 실험 비교

```python
# No augmentation
train_loader_no_aug, val_loader_no_aug = create_dataloaders(
    maps_path='data.npy',
    params_path='params.npy',
    use_augmentation=False
)

# With augmentation
train_loader_aug, val_loader_aug = create_dataloaders(
    maps_path='data.npy',
    params_path='params.npy',
    use_augmentation=True,
    augmentation_config={'rotation_p': 0.75, 'flip_p': 0.5}
)

# 두 모델 비교 가능!
```

### 예시 3: 커스텀 Augmentation

```python
from dataloaders.augmentation import RandomRotation90, RandomFlip, Compose

# 회전만
rotation_only = RandomRotation90(p=0.8)

# 뒤집기만
flip_only = RandomFlip(horizontal_p=0.5, vertical_p=0.5)

# 커스텀 조합
custom_aug = Compose([rotation_only, flip_only])

# Dataset에 적용
dataset = CosmologyDataset(
    maps_path='data.npy',
    params_path='params.npy',
    augmentation=custom_aug
)
```

## 🔍 디버깅 및 검증

### Augmentation이 제대로 작동하는지 확인

```python
# 1. 시각화로 확인
python visualize_augmentation.py --sample_idx 0 --num_augs 8

# 2. 코드로 확인
from dataloaders.augmentation import CosmologyAugmentation
import torch

aug = CosmologyAugmentation(rotation_p=1.0, horizontal_flip_p=1.0)
img = torch.randn(1, 256, 256)

print("Original shape:", img.shape)
augmented = aug(img)
print("Augmented shape:", augmented.shape)
print("Shape preserved:", img.shape == augmented.shape)
```

### Training 중 확인

```python
# Training loop에서 확인
for epoch in range(num_epochs):
    for batch_idx, (images, params) in enumerate(train_loader):
        # images는 이미 augmented됨
        print(f"Batch {batch_idx}: {images.shape}")
        
        # 첫 번째 배치만 시각화
        if batch_idx == 0:
            visualize_batch(images[:4])  # 처음 4개만
```

## 📈 모니터링

### Augmentation 효과 측정

1. **Training Loss**: Augmentation 사용 시 더 안정적
2. **Validation Loss**: 일반화 성능 향상
3. **Parameter Inference**: 더 robust한 추정
4. **Visualization**: 다양한 변형 확인

## 🎉 완성!

### 체크리스트

- [x] RandomRotation90 구현
- [x] RandomFlip 구현  
- [x] CosmologyAugmentation 통합
- [x] Dataset 통합
- [x] 시각화 스크립트
- [x] 사용 가이드
- [x] 테스트 완료

### 바로 사용하기

```bash
cd /home/work/Cosmology/FOCUS

# 1. 시각화 확인
python visualize_augmentation.py

# 2. Training에 적용
# config.yaml에서 use_augmentation: true 설정

# 3. 또는 코드에서 직접
train_loader, val_loader = create_dataloaders(
    maps_path='processed_data/maps_normalized.npy',
    params_path='processed_data/params_normalized.npy',
    use_augmentation=True
)
```

## 🚀 다음 단계

원하시면 추가 구현 가능:
1. **더 많은 Augmentation**: Noise, Elastic deformation
2. **Adaptive Augmentation**: 학습 중 확률 조정
3. **Augmentation Policy**: AutoML로 최적 확률 찾기
4. **Mixup/CutMix**: 이미지 블렌딩 기법

---

**Data Augmentation 시스템이 완성되었습니다!** 🎊

**위치**: `/home/work/Cosmology/FOCUS/dataloaders/augmentation.py`  
**시각화**: `/home/work/Cosmology/FOCUS/visualize_augmentation.py`  
**결과**: `/home/work/Cosmology/FOCUS/augmentation_vis/`

**이제 2D cosmological maps를 회전하고 뒤집어서 더 robust한 모델을 훈련할 수 있습니다!** 🌌✨
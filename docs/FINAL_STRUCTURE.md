# FOCUS: 최종 프로젝트 구조

## ✨ Flow Matching & Diffusion Model for Cosmological Universe Simulation

```
focus/
│
├── 📚 문서
│   ├── README.md              # 메인 문서
│   ├── QUICKSTART.md          # 빠른 시작 가이드
│   ├── PROJECT_STRUCTURE.md   # 구조 설명
│   └── GITHUB_SETUP.md        # GitHub 설정
│
├── 🎯 메인 스크립트
│   ├── train.py               # 학습 진입점
│   ├── inference.py           # 추론 진입점
│   ├── setup.py               # 패키지 설정
│   └── requirements.txt       # 의존성
│
├── 🎨 models/                 # 모델 아키텍처
│   ├── __init__.py
│   ├── embeddings.py          # Time & condition embeddings
│   └── unet.py                # SimpleUNet
│
├── 📦 dataloaders/            # 데이터 처리
│   ├── __init__.py
│   ├── cosmology_dataset.py   # PyTorch Dataset
│   └── prepare_data.py        # 데이터 전처리
│
├── 🌊 diffusion/              # 디퓨전 알고리즘
│   ├── __init__.py
│   ├── schedules.py            # Noise schedules
│   └── ddpm.py                # DDPM/DDIM 구현
│
├── 🏃 training/                # 학습 인프라
│   ├── __init__.py
│   ├── trainer.py             # DiffusionTrainer
│   └── callbacks.py           # EarlyStopping, Checkpointing
│
├── 🔧 utils/                   # 유틸리티 & 테스트
│   ├── __init__.py
│   ├── normalization.py        # 정규화 유틸
│   ├── visualization.py       # 시각화 유틸
│   ├── power_spectrum.py      # 파워스펙트럼 분석
│   ├── README.md               # utils 설명
│   ├── read_data.py            # 데이터 읽기
│   ├── test_diffusion.py       # 디퓨전 테스트
│   ├── test_real_data.py       # 실제 데이터 테스트
│   ├── visualize_data.py       # 데이터 시각화
│   ├── visualize_forward_process.py  # Forward process 시각화
│   └── run_all.sh              # 전체 실행 스크립트
│
├── 🔮 parameter_inference/    # 파라미터 추론
│   ├── __init__.py
│   ├── inference.py            # 모델 로딩 & 샘플링
│   └── sampling.py             # 샘플링 유틸
│
├── ⚙️ configs/                  # 설정 파일
│   └── default.yaml            # 기본 설정
│
├── 🧪 tasks/                    # 실험 관리
│   ├── experiment_01/          # 예제 실험
│   │   ├── config.yaml
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── figs/
│   ├── train_experiment.py     # 학습 스크립트
│   └── inference_experiment.py # 추론 스크립트
│
├── 🌀 flowmatching/             # Flow matching (미래)
│   └── __init__.py
│
└── 📊 manifold_analysis/        # 매니폴드 분석 (미래)
    └── __init__.py
```

## ✅ 정리 완료 내용

### 1. **scripts 폴더 삭제**
- ❌ `scripts/` 폴더 완전 삭제
- ✅ 모든 파일을 `utils/`로 통합

### 2. **utils 폴더 통합**
- ✅ Core utilities: `normalization.py`, `visualization.py`, `power_spectrum.py`
- ✅ Testing scripts: `test_diffusion.py`, `test_real_data.py`
- ✅ Visualization: `visualize_data.py`, `visualize_forward_process.py`
- ✅ Data utilities: `read_data.py`
- ✅ Setup script: `run_all.sh`

### 3. **깔끔한 구조**
- ✅ 루트 디렉토리에 불필요한 파일 없음
- ✅ 모든 모듈이 논리적으로 정리됨
- ✅ 확장하기 쉬운 구조

## 🚀 사용 방법

### 학습
```bash
python train.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01
```

### 추론
```bash
python inference.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01
```

### 테스트 & 시각화
```bash
# 데이터 시각화
python utils/visualize_data.py

# 디퓨전 테스트
python utils/test_diffusion.py

# 실제 데이터 테스트
python utils/test_real_data.py
```

## 📝 변경 사항

| 이전 | 현재 | 상태 |
|-----|------|-----|
| `scripts/` | 삭제됨 | ✅ 통합됨 |
| `scripts/legacy/` | 삭제됨 | ✅ 더 이상 불필요 |
| 모든 테스트 스크립트 | `utils/` | ✅ 이동 완료 |
| `flow` | `flowmatching/` | ✅ 이름 수정 |

## 🎯 다음 단계

이제 프로젝트가 완전히 정리되었습니다:
- ✅ 모듈화 완료
- ✅ 확장 가능한 구조
- ✅ 실험 관리 체계화
- ✅ 테스트/시각화 통합

Flow matching, Transformer, 새로운 모델 등을 쉽게 추가할 수 있습니다!


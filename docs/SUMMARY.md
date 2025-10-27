# ✅ FOCUS: 정리 완료!

## 🎉 Flow Matching & Diffusion Model for Cosmological Universe Simulation

### ✅ 완료된 작업들

#### 1. **문서 정리**
- ✅ 모든 MD 파일을 `docs/` 폴더로 이동
- ✅ `README.md`, `QUICKSTART.md`, `PROJECT_STRUCTURE.md` 등 모두 docs에

#### 2. **YAML 설정 개선**
```yaml
# 모델 모듈화
model:
  from: "models.unet.SimpleUNet"  # 나중에 DiT 등 추가 가능!
  in_channels: 1
  ...

# 데이터 단축어
data:
  data_dir: "processed"  # 간단하게!
  
# Optimizer 선택
training:
  optimizer: "adamw"  # adamw, adam, sgd
  
# Scheduler 선택
  scheduler:
    name: "plateau"  # plateau, cosine, step
```

#### 3. **추가된 기능들**
- ✅ `from` 으로 모델 동적 import
- ✅ `processed` 단축어 지원
- ✅ Optimizer 선택 (adamw, adam, sgd)
- ✅ Scheduler 선택 (plateau, cosine, step)
- ✅ Shuffle 옵션 (true/false)

#### 4. **코드 모듈화**
- ✅ 모든 파일 정리
- ✅ utils에 테스트/시각화 스크립트
- ✅ 깔끔한 프로젝트 구조

### 📁 최종 구조

```
focus/
├── 📚 docs/                  # ✅ 모든 문서
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── PROJECT_STRUCTURE.md
│   └── CONFIG_GUIDE.md
│
├── ⚙️ configs/               # ✅ YAML 설정
│   └── default.yaml
│
├── 🧪 tasks/                 # ✅ 실험 관리
│   └── experiment_01/
│
├── 🎯 train.py              # ✅ 메인 스크립트
├── inference.py
└── ... (모듈들)
```

### 🚀 사용 방법

```bash
# 학습
python train.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01

# 새로운 실험
mkdir -p tasks/my_exp/{checkpoints,logs,figs}
cp configs/default.yaml tasks/my_exp/config.yaml
# config.yaml 수정
python train.py --config tasks/my_exp/config.yaml --exp_dir tasks/my_exp
```

### 📝 YAML 설정 예시

```yaml
data:
  data_dir: "processed"  # ✅ 단축어!
  shuffle: true          # ✅ random/sequential

model:
  from: "models.unet.SimpleUNet"  # ✅ 동적 import!
  ...

training:
  optimizer: "adamw"     # ✅ 선택 가능!
  scheduler:
    name: "plateau"      # ✅ 선택 가능!
```

**모든 작업이 완료되었습니다!** 🎊


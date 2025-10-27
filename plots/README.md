# Plots Directory

이 폴더에는 모든 시각화 결과가 저장됩니다.

## 폴더 구조

각 폴더는 특정 분석이나 실험을 나타냅니다:

- `00_data_overview/` - 원본 데이터 개요 및 기본 통계
- `01_forward_diffusion_process/` - Forward diffusion 과정 분석
- `02_*/` - 추가 실험 (필요시 생성)

## 각 폴더 설명

### 00_data_overview/
- `original_data_visualization.png` - 6개 샘플의 이미지 + 파워스펙트럼
- `average_power_spectrum.png` - 100개 샘플 평균 파워스펙트럼
- `test_forward_diffusion.png` - Forward diffusion 가우시안 수렴 테스트
- `test_real_data_forward_diffusion.png` - 실제 데이터 테스트
- `test_sampling.png` - DDIM 샘플링 테스트

### 01_forward_diffusion_process/
- Forward diffusion 각 단계에서의 이미지 및 파워스펙트럼 변화

## 명명 규칙

새로운 실험을 추가할 때:
- 폴더명: `NN_experiment_name/` (NN은 순서 번호)
- 파일명: 실험 내용을 명확히 설명

예시:
```
plots/
├── 00_data_overview/
├── 01_forward_diffusion_process/
├── 02_training_results/
└── 03_generation_comparison/
```


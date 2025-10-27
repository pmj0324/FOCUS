#!/bin/bash

# 조건부 디퓨전 모델 전체 파이프라인 실행 스크립트

echo "=================================="
echo "Conditional Diffusion Model Pipeline"
echo "=================================="

# 프로젝트 디렉토리로 이동
cd /Users/pmj0324/Sicence/cosmo/New

echo ""
echo "[1/3] 데이터 전처리 중..."
echo "=================================="
python prepare_data.py
if [ $? -ne 0 ]; then
    echo "❌ 데이터 전처리 실패"
    exit 1
fi

echo ""
echo "[2/3] 모델 학습 중..."
echo "=================================="
python train.py
if [ $? -ne 0 ]; then
    echo "❌ 학습 실패"
    exit 1
fi

echo ""
echo "[3/3] 추론 및 샘플링..."
echo "=================================="
python inference.py
if [ $? -ne 0 ]; then
    echo "❌ 추론 실패"
    exit 1
fi

echo ""
echo "=================================="
echo "✓ 전체 파이프라인 완료!"
echo "=================================="
echo "결과 확인:"
echo "  - 학습 결과: outputs/"
echo "  - 추론 결과: inference_results/"


#!/bin/bash

# PDBbind 데이터 전처리 전체 프로세스 실행 스크립트

echo "=========================================="
echo "PDBbind 데이터 전처리 프로세스 시작"
echo "=========================================="

# 1단계: PDBbind 데이터 전처리
echo ""
echo "Step 1: PDBbind 데이터 전처리 중..."
python preprocess_pdbbind.py

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed!"
    exit 1
fi

echo ""
echo "Step 1 완료!"

# 2단계: Distance Matrix 생성
echo ""
echo "Step 2: Distance Matrix 생성 중..."
python create_target_distance_matrix.py

if [ $? -ne 0 ]; then
    echo "Error: Step 2 failed!"
    exit 1
fi

echo ""
echo "Step 2 완료!"

# 3단계: Drug Graph 생성
echo ""
echo "Step 3: Drug Graph 생성 중..."
python create_drug_graph.py

if [ $? -ne 0 ]; then
    echo "Error: Step 3 failed!"
    exit 1
fi

echo ""
echo "Step 3 완료!"

echo ""
echo "=========================================="
echo "모든 전처리 완료!"
echo "이제 training.py를 실행할 수 있습니다."
echo "=========================================="


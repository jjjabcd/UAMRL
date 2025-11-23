#!/bin/bash

echo "==============================="
echo "  CLEANING GENERATED FILES..."
echo "==============================="

# 1) 전처리용 CSV 삭제
rm -f ../data/train_data.csv
rm -f ../data/val_data.csv
rm -f ../data/test_data.csv

echo "[✓] CSV files removed."

# 2) 생성된 SMILES 삭제
rm -f train_set/drug_smiles/*.smi
echo "[✓] SMILES files removed."

# 3) 생성된 FASTA 삭제
rm -f train_set/target_fasta/*.fasta
echo "[✓] FASTA files removed."

# 4) 생성된 Distance Matrix 삭제
rm -f train_set/distance_matrix/*.npz
echo "[✓] Distance matrix files removed."

# 5) PyG processed dataset 삭제
rm -f train_set/processed/*.pt
echo "[✓] Processed .pt files removed."

# 6) (Optional) 그래프 캐시 폴더 삭제
rm -rf train_set/drug_graph/
rm -rf graph_cache/
echo "[✓] Graph cache removed."

# 7) Python cache 삭제
find . -name "__pycache__" -type d -exec rm -rf {} +
echo "[✓] Python cache removed."

echo "==============================="
echo "       CLEANUP COMPLETE!       "
echo "==============================="

echo ""
echo "원본 입력 데이터(v2016 SDF/PDB)는 삭제하지 않았습니다."
echo "이제 preprocess_pdbbind.py 를 다시 실행하면 됩니다."

#!/bin/bash

##############################################
# 사용자 입력
##############################################

CHECKPOINT="/home/rlawlsgurjh/hdd/work/DTA/UAMRL/code/results/best_model/model_0.pt"

SDF_PATH="/home/rlawlsgurjh/hdd/work/DTA/UAMRL/code/train_set/drug_sdf/1a0t.sdf"
PDB_PATH="/home/rlawlsgurjh/hdd/work/DTA/UAMRL/code/train_set/target_pdb/1a0t.pdb"


##############################################
# 실행
##############################################

echo "[INFO] Running prediction..."
echo " - Checkpoint : $CHECKPOINT"
echo " - SDF file   : $SDF_PATH"
echo " - PDB file   : $PDB_PATH"

python predict.py \
    --sdf "$SDF_PATH" \
    --pdb "$PDB_PATH" \
    --ckpt "$CHECKPOINT"

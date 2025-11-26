#!/bin/bash

CHECKPOINT="./results/best_model/model_0.pt"

PDB_ID="1at5"
SDF_PATH="./train_set/drug_sdf/${PDB_ID}.sdf"
PDB_PATH="./train_set/target_pdb/${PDB_ID}.pdb"


echo "[INFO] Running prediction..."
echo " - Checkpoint : $CHECKPOINT"
echo " - SDF file   : $SDF_PATH"
echo " - PDB file   : $PDB_PATH"

python predict.py \
    --sdf "$SDF_PATH" \
    --pdb "$PDB_PATH" \
    --ckpt "$CHECKPOINT"

#!/bin/bash

CHECKPOINT="./results/best_model/model_0.pt"

LIGAND_PDB_ID="1at5"
TARGET_PDB_ID="1a4g"
SDF_PATH="./data/v2016/${LIGAND_PDB_ID}/${LIGAND_PDB_ID}_ligand.sdf"
PDB_PATH="./data/v2016/${TARGET_PDB_ID}/${TARGET_PDB_ID}_protein.pdb"

echo "[INFO] Running prediction..."
echo " - Checkpoint : $CHECKPOINT"
echo " - LIGAND SDF file   : $SDF_PATH"
echo " - TARGET PDB file   : $PDB_PATH"

python predict.py \
    --sdf "$SDF_PATH" \
    --pdb "$PDB_PATH" \
    --ckpt "$CHECKPOINT"

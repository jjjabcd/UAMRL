#!/bin/bash

# PDBbind Data Preprocessing Full Process Execution Script

echo "=========================================="
echo "Starting PDBbind Data Preprocessing Process"
echo "=========================================="

# Step 1: PDBbind Data Preprocessing
echo ""
echo "Step 1: Preprocessing PDBbind data..."
python -m uamrl.dataset.preprocess_pdbbind

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed!"
    exit 1
fi

echo ""
echo "Step 1 Completed!"

# Step 2: Generating Distance Matrix
echo ""
echo "Step 2: Generating Distance Matrix..."
python -m uamrl.dataset.create_target_distance_matrix

if [ $? -ne 0 ]; then
    echo "Error: Step 2 failed!"
    exit 1
fi

echo ""
echo "Step 2 Completed!"

# Step 3: Generating Drug Graph
echo ""
echo "Step 3: Generating Drug Graph..."
python -m uamrl.dataset.create_drug_graph

if [ $? -ne 0 ]; then
    echo "Error: Step 3 failed!"
    exit 1
fi

echo ""
echo "Step 3 Completed!"

echo ""
echo "=========================================="
echo "All Preprocessing Completed!"
echo "You can now run training.py."
echo "=========================================="
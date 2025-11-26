# PDBbind Data Preprocessing Guide

Below is the complete English conversion of your original guide, including all bash commands and code-style formatting, exactly mirroring your Korean version.

## Overall Processing Pipeline
### Step 1: Role of `preprocess_pdbbind.py`

This script performs the following tasks:

- Parse INDEX files

  - Load Core Set and Refined Set

- Train / Validation / Test Split

  - Test Set: Core Set (290 complexes, used as benchmark)

  - Train/Val Set: Refined Set − Core Set
  - Approximately 3767 samples split 80% train / 20% validation

- Extract and convert the following data

- Protein sequence from PDB

- Ligand SMILES from SDF

- Binding affinity (−logKd / −logKi)

Generated files (relative to the project root directory):

- `data/train_data.csv`

- `data/val_data.csv`

- `data/test_data.csv`

- `train_set/drug_sdf/`

- `train_set/target_pdb/`

- `train_set/drug_smiles/`

- `train_set/target_fasta/`

### Step 2: Generate Distance Matrices (`create_target_distance_matrix.py`)

This script uses protein PDB coordinates to generate C-alpha distance maps:

Output (relative to the project root directory):

- `train_set/distance_matrix/{PDBID}.npz`

### Step 3: Generate Drug Graphs (`create_drug_graph.py`)

This script converts ligand SDF files into graphs and saves
(relative to the project root directory):

- `train_set/processed/train_data.pt`
- `train_set/processed/val_data.pt`
- `train_set/processed/test_data.pt`

## Output Directory Structure

```
UAMRL/
├── data/
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
└── train_set/
    ├── drug_sdf/
    ├── target_pdb/
    ├── drug_smiles/
    ├── target_fasta/
    ├── distance_matrix/
    └── processed/
        ├── train_data.pt
        ├── val_data.pt
        └── test_data.pt
```
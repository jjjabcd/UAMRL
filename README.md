# UAMRL: Multi-Granularity Uncertainty-Aware Multimodal Representation Learning for Drug-Target Affinity Prediction

Original Repository: https://github.com/Astraea2xu/UAMRL

# 1. Environment Setup
It's recommended to set up the environment using the provided env.yml file.

```bash
# Create conda environment
conda create -n uamrl python=3.10 -y
conda activate uamrl
pip install -r requirements.txt
```

# 2. Data Download and Preparation
The model uses the PDBbind dataset.

Download Link: https://www.pdbbind-plus.org.cn/download

Required File: Download the PDBbind v2016 dataset archive (e.g., `pdbbind_v2016.tar.gz`).

Data Placement and Extraction:

Place the downloaded archive file into the data/ directory of this project.

```bash
# Assuming the file is downloaded to the project root:

mv pdbbind_v2016.tar.gz data/

cd data

# Decompress the PDBbind data archive
tar -xzvf pdbbind_v2016.tar.gz

cd ..
```


# 3. Example Usage (Running the Pipeline)
Once the environment is active and the data is extracted to data/, you can run the full pipeline step-by-step.

## 1. Preprocessing Data
This step preprocesses the PDBbind data, generates the target distance matrices, and creates the drug graph representations required for the model.

```Bash
bash run_preprocessing.sh
```

## 2. Model Training
After the preprocessing is complete, train the UAMRL model.

```Bash
bash run_training.sh
```

## 3. Prediction
Run predictions using the trained model.

```Bash
bash run_predict.sh
```


## Directory Structure

```
UAMRL/
├── data/
├── results/
├── train_set/
├── uamrl/
│    ├── dataset/
│    │   ├── __init__.py
│    │   ├── create_data.py
│    │   ├── create_drug_graph.py
│    │   ├── create_target_distance_matrix.py
│    │   ├── preprocess_pdbbind.py
│    │   └── README.md
│    │
│    ├── models/
│    │   ├── __init__.py
│    │   └── UAMRL.py
│    │
│    ├── util/
│    │   ├── __init__.py
│    │   ├── graphUtil.py
│    │   └── util.py
│    │
│    ├── __init__.py
│    └── config.py
│
├── env.yml
├── requirements.txt
├── predict.py
├── README.md
├── .gitignore
│
├── run_predict.sh
├── run_preprocessing.sh
├── run_training.sh
│
├── training.py
└── pdbbind_v2016.tar.gz

```
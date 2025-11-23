import argparse
import os
import numpy as np
import torch
from rdkit import Chem
from Bio.PDB import PDBParser
import math
from models.UAMRL import UAMRL
from util.util import smiles_dict
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1. Extract SMILES from SDF
# -----------------------------
def extract_smiles_from_sdf(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
    for mol in supplier:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            return smiles
        except:
            continue
    raise ValueError("❌ SDF에서 SMILES를 추출할 수 없음.")


# -----------------------------
# 2. Convert SMILES → one-hot
# -----------------------------
def smiles_to_onehot(smiles, max_len=150):
    integer_encoded = []
    for ch in smiles:
        if ch not in smiles_dict:
            raise ValueError(f"❌ 지원하지 않는 SMILES 문자 발견: {ch}")
        integer_encoded.append(smiles_dict[ch])

    # padding
    if len(integer_encoded) > max_len:
        integer_encoded = integer_encoded[:max_len]

    onehot = []
    for idx in integer_encoded:
        vec = [0] * (len(smiles_dict) + 1)
        vec[idx] = 1
        onehot.append(vec)

    while len(onehot) < max_len:
        pad = [0] * (len(smiles_dict) + 1)
        pad[0] = 1
        onehot.append(pad)

    return torch.tensor(onehot, dtype=torch.float32)


# -----------------------------
# 3. Extract protein sequence from PDB
# -----------------------------
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def pdb_to_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    chains = {}
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                resname = residue.get_resname()
                if resname in AA_MAP:
                    seq.append(AA_MAP[resname])
            if len(seq) > 0:
                chains[chain.id] = "".join(seq)
        break

    if len(chains) == 0:
        raise ValueError("❌ PDB에서 단백질 서열을 찾지 못함.")

    # ⭐ 가장 긴 chain 자동 선택
    best_chain = max(chains, key=lambda x: len(chains[x]))
    return chains[best_chain]


# -----------------------------
# 4. Convert sequence → one-hot
# -----------------------------
AMINO_ACIDS = ['PAD','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
aa_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def seq_to_onehot(seq, max_len=1000):
    encoded = []
    for aa in seq:
        if aa not in aa_index:
            continue
        idx = aa_index[aa]
        vec = [0] * len(AMINO_ACIDS)
        vec[idx] = 1
        encoded.append(vec)

    # crop
    if len(encoded) > max_len:
        encoded = encoded[:max_len]

    # padding
    while len(encoded) < max_len:
        pad = [0] * len(AMINO_ACIDS)
        pad[0] = 1
        encoded.append(pad)

    return torch.tensor(encoded, dtype=torch.float32)


# -----------------------------
# 5. PDB → Distance matrix
# -----------------------------
def pdb_to_distance_matrix(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                if "CA" in residue:
                    coord = residue["CA"].get_vector()
                    coords.append([coord[0], coord[1], coord[2]])
        break

    if len(coords) == 0:
        raise ValueError("❌ CA 원자를 찾지 못함 → distance matrix 생성 불가")

    coords = np.array(coords)
    n = len(coords)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            else:
                dist[i][j] = np.linalg.norm(coords[i] - coords[j])

    # resize for CNN input (224×224)
    import cv2
    resized = cv2.resize(dist, (224, 224), interpolation=cv2.INTER_AREA)
    return torch.tensor(resized).unsqueeze(0).float()  # (1,224,224)


# -----------------------------
# 6. Prediction function
# -----------------------------
def predict(sdf_path, pdb_path, ckpt_path):
    print("\n=== 1) SDF → SMILES 추출 ===")
    smiles = extract_smiles_from_sdf(sdf_path)
    print("SMILES:", smiles)

    print("\n=== 2) SMILES → one-hot ===")
    c_seq = smiles_to_onehot(smiles).unsqueeze(0).to(device)

    print("\n=== 3) PDB → sequence 추출 ===")
    seq = pdb_to_sequence(pdb_path)
    print("SEQ length:", len(seq))

    print("\n=== 4) SEQ → one-hot ===")
    p_seq = seq_to_onehot(seq).unsqueeze(0).to(device)

    print("\n=== 5) PDB → Distance matrix ===")
    p_img = pdb_to_distance_matrix(pdb_path).unsqueeze(0).to(device)

    print("\n=== 6) Load model ===")
    model = UAMRL(65, 21, [32,64,128], 256).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("\n=== 7) Predict ===")
    with torch.no_grad():
        output, cmd_loss = model(c_seq, p_seq, p_img, None)
        mu, v, alpha, beta = output[:4]
        mu = mu.item()

    print("\n==============================")
    print(" Predicted Affinity:", mu)
    print("==============================")
    return mu


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sdf", required=True, help="Drug SDF file")
    parser.add_argument("--pdb", required=True, help="Target PDB file")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint (.pt)")

    args = parser.parse_args()

    predict(args.sdf, args.pdb, args.ckpt)

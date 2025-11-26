import os
import argparse
import torch
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from uamrl.util.util import protein2onehot, smiles_dict
from uamrl.models.UAMRL import UAMRL

import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# Drug SDF → SMILES
############################################
def extract_smiles_from_sdf(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
    for mol in supplier:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    return None


############################################
# SMILES → One-hot
############################################
def smiles_to_onehot(smiles, max_len=150):
    out = []
    for ch in smiles:
        if ch not in smiles_dict:
            continue
        idx = smiles_dict[ch]
        vec = [0] * (len(smiles_dict) + 1)
        vec[idx] = 1
        out.append(vec)

    # pad
    if len(out) < max_len:
        pad = [0] * (len(smiles_dict) + 1)
        out.extend([pad] * (max_len - len(out)))
    else:
        out = out[:max_len]

    return torch.tensor(out).float().unsqueeze(0).to(device)


############################################
# PDB → FASTA sequence
############################################
AA_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
}


def pdb_to_fasta(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)

    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                name = residue.get_resname()
                if name in AA_3to1:
                    seq.append(AA_3to1[name])
            break
        break

    return "".join(seq)


############################################
# FASTA → one-hot
############################################
amino_acids = ['PAD','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
               'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

def fasta_to_onehot(seq, max_len=1000):
    out = []
    for ch in seq:
        if ch not in aa_to_idx:
            continue
        idx = aa_to_idx[ch]
        vec = [0] * len(amino_acids)
        vec[idx] = 1
        out.append(vec)

    # PAD
    if len(out) < max_len:
        pad = [0] * len(amino_acids)
        out.extend([pad] * (max_len - len(out)))
    else:
        out = out[:max_len]

    return torch.tensor(out).float().unsqueeze(0).to(device)


############################################
# Distance Matrix
############################################
def pdb_to_distance_matrix(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    coords.append(atom.coord)
            break
        break

    coords = np.array(coords)
    if len(coords) == 0:
        raise ValueError("No CA atoms found")

    dist = np.sqrt(
        np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    )

    # (L, L) → (1, 1, L, L)
    dist = torch.tensor(dist).float().unsqueeze(0).unsqueeze(0).to(device)
    return dist


############################################
# SDF → PyG graph
############################################
atom_list = ['C','H','O','N','F','S','P','I','Cl','As','Se','Br','B','Pt','V',
             'Fe','Hg','Rh','Mg','Be','Si','Ru','Sb','Cu','Re','Ir','Os']


def sdf_to_graph(sdf_path):
    with open(sdf_path, "r") as f:
        f.readline(); f.readline(); f.readline()

        node_list = []
        edge_list = []
        feature_list = []

        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == "$$$$":
                break

            parts = list(filter(None, line.split(" ")))

            if len(parts) > 4 and parts[3] in atom_list:
                node_list.append(parts[3])

            if len(parts) == 6 and parts[0].isdigit():
                a = int(parts[0]) - 1
                b = int(parts[1]) - 1
                edge_list.append([a, b])
                edge_list.append([b, a])

    # feature
    for atom in node_list:
        v = [0] * len(atom_list)
        v[atom_list.index(atom)] = 1
        feature_list.append(v)

    x = torch.tensor(feature_list).float().to(device)

    if len(edge_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
    else:
        edge_index = torch.tensor(edge_list).long().t().contiguous().to(device)

    g = Data(x=x, edge_index=edge_index)
    g.batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

    return g


############################################
# PREDICT FUNCTION
############################################
def predict(sdf_path, pdb_path, ckpt_path):

    print("=== Load checkpoint ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UAMRL(65, 21, [32, 64, 128], 256)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # -------------------------
    # 1) SDF → SMILES → One-hot
    # -------------------------
    print("=== Drug SDF → SMILES ===")
    smiles = extract_smiles_from_sdf(sdf_path)
    c_seq = smiles_to_onehot(smiles).to(device)   # shape (1,150,65)

    # -------------------------
    # 2) SDF → Graph
    # -------------------------
    print("=== SDF → Graph ===")
    graph = sdf_to_graph(sdf_path)

    pdb_id = os.path.basename(pdb_path).replace(".pdb", "")
    graph.id = [pdb_id]
    graph = graph.to(device)

    # -------------------------
    # 3) PDB → FASTA → One-hot
    # -------------------------
    print("=== PDB → FASTA ===")
    fasta_seq = pdb_to_fasta(pdb_path)

    fasta_save_path = f"train_set/target_fasta/{pdb_id}.fasta"
    os.makedirs("train_set/target_fasta", exist_ok=True)
    with open(fasta_save_path, "w") as f:
        f.write(f">{pdb_id}\n{fasta_seq}\n")

    p_seq = fasta_to_onehot(fasta_seq).to(device)   # (1,1000,21)

    # -------------------------
    # 4) PDB → Distance Matrix
    # -------------------------
    print("=== PDB → Distance matrix ===")
    p_img = pdb_to_distance_matrix(pdb_path)

    if isinstance(p_img, np.ndarray):
        p_img = torch.tensor(p_img, dtype=torch.float32)

    if p_img.dim() == 2:
        p_img = p_img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    p_img = p_img.to(device)

    # -------------------------
    # 5) Predict
    # -------------------------
    print("=== Predict ===")
    with torch.no_grad():
        output, _ = model(c_seq, p_seq, p_img, graph)

        # -----------------------------
        #  output = 32 values (8 modes × 4 params)
        #  last 4 values are gs_private
        # -----------------------------
        mu, v, alpha, beta = output[-4:]

        # scalars
        mu  = mu.squeeze()
        v   = v.squeeze()
        alpha = alpha.squeeze()
        beta  = beta.squeeze()

    print("\n===== Prediction Result (NIG Parameters) =====")
    print(f"Predicted Mean (delta) : {mu.item():.4f}")
    print(f"Gamma (gamma) : {v.item():.4f}")
    print(f"Alpha (alpha) : {alpha.item():.4f}")
    print(f"Beta (beta) : {beta.item():.4f}")

    return {
        "mu": mu.item(),
        "v": v.item(),
        "alpha": alpha.item(),
        "beta": beta.item()
    }


############################################
# CLI
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=str, required=True)
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    args = parser.parse_args()
    predict(args.sdf, args.pdb, args.ckpt)

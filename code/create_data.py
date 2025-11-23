# create_data.py
import os
import math
import numpy as np
from rdkit import Chem
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm

###############################################
# 1) Drug: SDF → SMILES
###############################################
def sdf_to_smiles(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True)
    for mol in suppl:
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    return None


###############################################
# 2) Drug: SDF → Graph (UAMRL 방식 그대로)
###############################################
from util.graphUtil import getCompoundGraph

def sdf_to_graph(sdf_path, temp_id="temp_drug"):
    return getCompoundGraph(temp_id, sdf_path=sdf_path)


###############################################
# 3) Protein: PDB → FASTA
###############################################
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def pdb_to_fasta(pdb_path, chain_id=None):
    """
    PDB → FASTA sequence 변환
    chain_id 지정 안 하면 첫 번째 단백질 체인을 사용
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        # chain 선택
        for model in structure:
            for chain in model:
                if chain_id is None or chain.id == chain_id:
                    seq = []
                    for residue in chain:
                        if residue.id[0] != " ":  # HETATM 등 제외
                            continue
                        resname = residue.get_resname()
                        if resname in AA_MAP:
                            seq.append(AA_MAP[resname])
                    if seq:
                        return "".join(seq)
            break

        return None

    except Exception as e:
        print(f"[ERROR] pdb_to_fasta({pdb_path}) failed:", e)
        return None


###############################################
# 4) Protein: PDB → Distance Matrix
###############################################
def pdb_to_distance_matrix(pdb_path, save_npz_path=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coordinate_list = []

    # === C-alpha 좌표 추출 (원본 코드 그대로) ===
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == 'CA':
                        coord = atom.get_vector()
                        coordinate_list.append([coord[0], coord[1], coord[2]])
        break  # 첫 번째 model만 사용

    if len(coordinate_list) == 0:
        raise ValueError(f"No CA atoms found in protein: {pdb_path}")

    # === 거리 행렬 계산 (원본 코드 그대로) ===
    CA_Metric = []
    for i in range(len(coordinate_list)):
        row = []
        for j in range(len(coordinate_list)):
            if i == j:
                row.append(0.0)
            else:
                dx = coordinate_list[i][0] - coordinate_list[j][0]
                dy = coordinate_list[i][1] - coordinate_list[j][1]
                dz = coordinate_list[i][2] - coordinate_list[j][2]
                row.append(math.sqrt(dx*dx + dy*dy + dz*dz))
        CA_Metric.append(row)

    CA_np = np.array(CA_Metric, dtype=np.float32)

    # 선택적으로 저장
    if save_npz_path:
        np.savez(save_npz_path, map=CA_np)

    return CA_np

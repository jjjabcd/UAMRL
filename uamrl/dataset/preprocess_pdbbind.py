import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
import shutil
import re

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings('ignore')


# PDBBIND_ROOT = '/HDD1/rlawlsgurjh/work/DTA/UAMRL/data/v2016'

PDBBIND_ROOT = os.path.join('data', 'v2016')
INDEX_DIR = os.path.join(PDBBIND_ROOT, 'index')
OUTPUT_DIR = os.path.join('data')
TRAIN_SET_DIR = 'train_set'

def parse_index_file(index_path, has_affinity_column=True):

    pdb_ids = []
    affinities = []
    
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
    file_content = None
    
    for encoding in encodings:
        try:
            with open(index_path, 'r', encoding=encoding, errors='ignore') as f:
                file_content = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    
    if file_content is None:
        print(f"Error: Could not read {index_path} with any encoding")
        return pdb_ids, affinities
    
    for line in file_content:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 4:
            continue
        
        pdb_id = parts[0].lower()
        
        if has_affinity_column:
            try:
                log_affinity = float(parts[3])
                affinity = -log_affinity
            except ValueError:
                continue
        else:
            binding_data = parts[3]
            affinity = None
            
            try:
                if 'Kd=' in binding_data or 'Ki=' in binding_data:
                    value_str = binding_data.split('=')[1]

                    if 'pM' in value_str:
                        value = float(value_str.replace('pM', '')) * 1e-12
                    elif 'nM' in value_str:
                        value = float(value_str.replace('nM', '')) * 1e-9
                    elif 'uM' in value_str or 'µM' in value_str:
                        value = float(value_str.replace('uM', '').replace('µM', '')) * 1e-6
                    elif 'mM' in value_str:
                        value = float(value_str.replace('mM', '')) * 1e-3
                    elif 'M' in value_str:
                        value = float(value_str.replace('M', ''))
                    else:
                        value = float(value_str)

                    if value > 0:
                        affinity = -np.log10(value)
            except (ValueError, IndexError):
                continue
            
            if affinity is None:
                continue
        
        pdb_ids.append(pdb_id)
        affinities.append(affinity)
    
    return pdb_ids, affinities

def extract_protein_sequence_from_pdb(pdb_path):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':
                        resname = residue.get_resname()
                        aa_map = {
                            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                        }
                        if resname in aa_map:
                            sequence.append(aa_map[resname])
                if sequence:
                    break
            if sequence:
                break
        
        return ''.join(sequence) if sequence else None
    except Exception as e:
        print(f"Error extracting sequence from {pdb_path}: {e}")
        return None

def extract_smiles_from_sdf(sdf_path):
    try:
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
        for mol in supplier:
            if mol is None:
                continue

            # <-- sanitize must be required
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None

            return Chem.MolToSmiles(mol, canonical=True)

        return None
    except:
        return None



def process_pdbbind_data():
    print("=" * 60)
    print("Step 1: Parsing INDEX files...")
    print("=" * 60)

    core_index_path = os.path.join(INDEX_DIR, 'INDEX_core_data.2016')
    core_pdb_ids, core_affinities = parse_index_file(core_index_path)
    print(f"Core set: {len(core_pdb_ids)} complexes")

    general_index_path = os.path.join(INDEX_DIR, 'INDEX_general_PL.2016')
    if os.path.exists(general_index_path):
        general_pdb_ids, general_affinities = parse_index_file(general_index_path, has_affinity_column=False)
        print(f"General set: {len(general_pdb_ids)} complexes")
    else:
        print("Warning: INDEX_general_PL.2016 not found, using refined set only")
        general_pdb_ids, general_affinities = [], []
    
    refined_index_path = os.path.join(INDEX_DIR, 'INDEX_refined_data.2016')
    refined_pdb_ids, refined_affinities = parse_index_file(refined_index_path)
    print(f"Refined set: {len(refined_pdb_ids)} complexes")
    
    print("\nMerging general and refined sets...")
    all_train_val_pdb_ids = {}
    all_train_val_affinities = {}
    
    for pdb_id, affinity in zip(general_pdb_ids, general_affinities):
        all_train_val_pdb_ids[pdb_id] = affinity
        all_train_val_affinities[pdb_id] = affinity
    
    for pdb_id, affinity in zip(refined_pdb_ids, refined_affinities):
        all_train_val_pdb_ids[pdb_id] = affinity
        all_train_val_affinities[pdb_id] = affinity
    
    print(f"Total (general + refined): {len(all_train_val_pdb_ids)} complexes")
    
    core_set = set(core_pdb_ids)
    train_val_pdb_ids = []
    train_val_affinities = []
    
    for pdb_id, affinity in all_train_val_affinities.items():
        if pdb_id not in core_set:
            train_val_pdb_ids.append(pdb_id)
            train_val_affinities.append(affinity)
    
    print(f"Train/Val set (after removing core set): {len(train_val_pdb_ids)} complexes")
    
    np.random.seed(42)
    indices = np.arange(len(train_val_pdb_ids))
    np.random.shuffle(indices)

    val_size = int(len(train_val_pdb_ids) * 0.1)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_pdb_ids = [train_val_pdb_ids[i] for i in train_indices]
    train_affinities = [train_val_affinities[i] for i in train_indices]
    val_pdb_ids = [train_val_pdb_ids[i] for i in val_indices]
    val_affinities = [train_val_affinities[i] for i in val_indices]
    
    print(f"\nFinal split:")
    print(f"  Train set: {len(train_pdb_ids)} complexes")
    print(f"  Val set: {len(val_pdb_ids)} complexes (~10%)")
    print(f"  Test set (core): {len(core_pdb_ids)} complexes")
    
    print("\n" + "=" * 60)
    print("Step 2: Processing files and extracting data...")
    print("=" * 60)

    def collect_data(pdb_ids, affinities, split_name):
        data_list = []
        missing_files = []
        
        for pdb_id, affinity in tqdm(zip(pdb_ids, affinities), total=len(pdb_ids), desc=f"Processing {split_name}"):
            pdb_folder = os.path.join(PDBBIND_ROOT, pdb_id)
            protein_pdb = os.path.join(pdb_folder, f"{pdb_id}_protein.pdb")
            ligand_sdf = os.path.join(pdb_folder, f"{pdb_id}_ligand.sdf")

            if not os.path.exists(protein_pdb):
                missing_files.append(f"{pdb_id}: protein PDB not found")
                continue
            if not os.path.exists(ligand_sdf):
                missing_files.append(f"{pdb_id}: ligand SDF not found")
                continue

            sequence = extract_protein_sequence_from_pdb(protein_pdb)
            if sequence is None or len(sequence) == 0:
                missing_files.append(f"{pdb_id}: failed to extract sequence")
                continue
            
            smiles = extract_smiles_from_sdf(ligand_sdf)
            if smiles is None:
                missing_files.append(f"{pdb_id}: failed to extract SMILES")
                continue
            
            if ':' in smiles:
                missing_files.append(f"{pdb_id}: SMILES contains ':'")
                continue
            
            data_list.append({
                'PDBID': pdb_id,
                'affinity': affinity,
                'target_sequence': sequence,
                'compound_iso_smiles': smiles
            })
        
        if missing_files:
            print(f"\nWarning: {len(missing_files)} files missing or failed:")
            for msg in missing_files[:10]:
                print(f"  {msg}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
        
        return pd.DataFrame(data_list)
    
    train_df = collect_data(train_pdb_ids, train_affinities, "Train")
    val_df = collect_data(val_pdb_ids, val_affinities, "Val")
    test_df = collect_data(core_pdb_ids, core_affinities, "Test")
    
    print(f"\nFinal counts:")
    print(f"Train: {len(train_df)} complexes")
    print(f"Val: {len(val_df)} complexes")
    print(f"Test: {len(test_df)} complexes")
    
    print("\n" + "=" * 60)
    print("Step 3: Creating CSV files...")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_df[['PDBID', 'affinity']].to_csv(
        os.path.join(OUTPUT_DIR, 'train_data.csv'), index=False
    )
    val_df[['PDBID', 'affinity']].to_csv(
        os.path.join(OUTPUT_DIR, 'val_data.csv'), index=False
    )
    test_df[['PDBID', 'affinity']].to_csv(
        os.path.join(OUTPUT_DIR, 'test_data.csv'), index=False
    )
    
    print("CSV files saved!")
    
    print("\n" + "=" * 60)
    print("Step 4: Copying and organizing files...")
    print("=" * 60)
    
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'drug_sdf'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'target_pdb'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'drug_smiles'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'target_fasta'), exist_ok=True)
    
    all_df = pd.concat([train_df, val_df, test_df])

    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Copying files"):
        pdb_id = row['PDBID']
        pdb_folder = os.path.join(PDBBIND_ROOT, pdb_id)

        src_sdf = os.path.join(pdb_folder, f"{pdb_id}_ligand.sdf")
        dst_sdf = os.path.join(TRAIN_SET_DIR, 'drug_sdf', f"{pdb_id}.sdf")
        if os.path.exists(src_sdf):
            shutil.copy2(src_sdf, dst_sdf)

        src_pdb = os.path.join(pdb_folder, f"{pdb_id}_protein.pdb")
        dst_pdb = os.path.join(TRAIN_SET_DIR, 'target_pdb', f"{pdb_id}.pdb")
        if os.path.exists(src_pdb):
            shutil.copy2(src_pdb, dst_pdb)

        smiles_path = os.path.join(TRAIN_SET_DIR, 'drug_smiles', f"{pdb_id}.smi")
        with open(smiles_path, 'w') as f:
            f.write(row['compound_iso_smiles'])

        fasta_path = os.path.join(TRAIN_SET_DIR, 'target_fasta', f"{pdb_id}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">{pdb_id}\n")
            f.write(f"{row['target_sequence']}\n")
    
    print("\n" + "=" * 60)
    print("Data preprocessing completed!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Train: {len(train_df)} complexes")
    print(f"  Val: {len(val_df)} complexes")
    print(f"  Test: {len(test_df)} complexes")
    print(f"\nFiles created:")
    print(f"  - {OUTPUT_DIR}/train_data.csv")
    print(f"  - {OUTPUT_DIR}/val_data.csv")
    print(f"  - {OUTPUT_DIR}/test_data.csv")
    print(f"  - {TRAIN_SET_DIR}/drug_sdf/ (SDF files)")
    print(f"  - {TRAIN_SET_DIR}/target_pdb/ (PDB files)")
    print(f"  - {TRAIN_SET_DIR}/drug_smiles/ (SMILES files)")
    print(f"  - {TRAIN_SET_DIR}/target_fasta/ (FASTA files)")

if __name__ == '__main__':
    process_pdbbind_data()


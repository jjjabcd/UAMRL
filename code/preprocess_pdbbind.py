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

# 경로 설정
# PDBbind 데이터 경로 설정
# 방법 1: 절대 경로 사용 (권장)
PDBBIND_ROOT = '/HDD1/rlawlsgurjh/work/DTA/UAMRL/data/v2016'

# 방법 2: 상대 경로 사용 (절대 경로가 작동하지 않을 경우)
# PDBBIND_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'v2016')
INDEX_DIR = os.path.join(PDBBIND_ROOT, 'index')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TRAIN_SET_DIR = 'train_set'

def parse_index_file(index_path, has_affinity_column=True):
    """
    INDEX 파일 파싱
    - INDEX_core_data.2016, INDEX_refined_data.2016: 
      형식: PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
    - INDEX_general_PL.2016:
      형식: PDB code, resolution, release year, binding data, reference, ligand name
      (binding data에서 Kd/Ki 값을 파싱해야 함)
    """
    pdb_ids = []
    affinities = []
    
    # 인코딩 시도 (UTF-8, latin-1, ISO-8859-1 순서로)
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
        # 주석이나 빈 줄 건너뛰기
        if not line or line.startswith('#'):
            continue
        
        # 공백으로 분리
        parts = line.split()
        if len(parts) < 4:
            continue
        
        pdb_id = parts[0].lower()
        
        if has_affinity_column:
            # Core/Refined set: 4번째 컬럼이 -logKd/Ki
            try:
                log_affinity = float(parts[3])
                affinity = -log_affinity  # pKd 또는 pKi
            except ValueError:
                continue
        else:
            # General set: binding data에서 파싱
            # 예: "Kd=49uM", "Ki=0.43uM", "Kd=5nM" 등
            binding_data = parts[3]
            affinity = None
            
            try:
                # Kd= 또는 Ki= 패턴 찾기
                if 'Kd=' in binding_data or 'Ki=' in binding_data:
                    # 값 추출
                    value_str = binding_data.split('=')[1]
                    
                    # 단위 변환
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
                        # 숫자만 있는 경우 (단위 없음)
                        value = float(value_str)
                    
                    # -log10(Kd/Ki)로 변환
                    if value > 0:
                        affinity = -np.log10(value)
            except (ValueError, IndexError):
                # 파싱 실패 시 건너뛰기
                continue
            
            if affinity is None:
                continue
        
        pdb_ids.append(pdb_id)
        affinities.append(affinity)
    
    return pdb_ids, affinities

def extract_protein_sequence_from_pdb(pdb_path):
    """PDB 파일에서 단백질 서열 추출"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # 일반 residue만
                        resname = residue.get_resname()
                        # 3-letter code를 1-letter code로 변환
                        aa_map = {
                            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                        }
                        if resname in aa_map:
                            sequence.append(aa_map[resname])
                if sequence:  # 첫 번째 비어있지 않은 chain만 사용
                    break
            if sequence:  # 첫 번째 비어있지 않은 model만 사용
                break
        
        return ''.join(sequence) if sequence else None
    except Exception as e:
        print(f"Error extracting sequence from {pdb_path}: {e}")
        return None

def extract_smiles_from_sdf(sdf_path):
    """SDF 파일에서 SMILES 추출 (RDKit 사용)"""
    try:
        # SDF 파일에서 분자 읽기
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
        
        for mol in supplier:
            if mol is not None:
                try:
                    # 분자 sanitize 시도
                    Chem.SanitizeMol(mol)
                    # SMILES로 변환
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if smiles and len(smiles) > 0:
                        return smiles
                except:
                    # sanitize 실패 시 그래도 시도
                    try:
                        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, sanitize=False)
                        if smiles and len(smiles) > 0:
                            return smiles
                    except:
                        continue
        
        # 실패 시 다른 방법 시도: 파일 직접 읽기 후 RDKit으로 변환
        try:
            with open(sdf_path, 'r') as f:
                content = f.read()
            # 첫 번째 분자만 읽기
            mol = Chem.MolFromMolBlock(content.split('$$$$')[0], sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if smiles and len(smiles) > 0:
                        return smiles
                except:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, sanitize=False)
                    if smiles and len(smiles) > 0:
                        return smiles
        except:
            pass
        
        return None
    except Exception as e:
        # 에러는 출력하지 않음 (너무 많은 출력 방지)
        return None

def process_pdbbind_data():
    """
    PDBbind 데이터 전처리 (논문 방법에 따라)
    1. Core set을 test set으로 사용 (290개)
    2. General set과 Refined set을 합침
    3. Core set과의 overlap 제거
    4. Train/Val split (약 10%를 validation set으로)
    """
    
    print("=" * 60)
    print("Step 1: Parsing INDEX files...")
    print("=" * 60)
    
    # Core set 파싱 (test set)
    core_index_path = os.path.join(INDEX_DIR, 'INDEX_core_data.2016')
    core_pdb_ids, core_affinities = parse_index_file(core_index_path)
    print(f"Core set: {len(core_pdb_ids)} complexes")
    
    # General set 파싱 (affinity 컬럼이 없음)
    general_index_path = os.path.join(INDEX_DIR, 'INDEX_general_PL.2016')
    if os.path.exists(general_index_path):
        general_pdb_ids, general_affinities = parse_index_file(general_index_path, has_affinity_column=False)
        print(f"General set: {len(general_pdb_ids)} complexes")
    else:
        print("Warning: INDEX_general_PL.2016 not found, using refined set only")
        general_pdb_ids, general_affinities = [], []
    
    # Refined set 파싱
    refined_index_path = os.path.join(INDEX_DIR, 'INDEX_refined_data.2016')
    refined_pdb_ids, refined_affinities = parse_index_file(refined_index_path)
    print(f"Refined set: {len(refined_pdb_ids)} complexes")
    
    # General set과 Refined set을 합침 (중복 제거)
    print("\nMerging general and refined sets...")
    all_train_val_pdb_ids = {}
    all_train_val_affinities = {}
    
    # General set 추가
    for pdb_id, affinity in zip(general_pdb_ids, general_affinities):
        all_train_val_pdb_ids[pdb_id] = affinity
        all_train_val_affinities[pdb_id] = affinity
    
    # Refined set 추가 (중복시 refined set 값 사용)
    for pdb_id, affinity in zip(refined_pdb_ids, refined_affinities):
        all_train_val_pdb_ids[pdb_id] = affinity
        all_train_val_affinities[pdb_id] = affinity
    
    print(f"Total (general + refined): {len(all_train_val_pdb_ids)} complexes")
    
    # Core set과의 overlap 제거
    core_set = set(core_pdb_ids)
    train_val_pdb_ids = []
    train_val_affinities = []
    
    for pdb_id, affinity in all_train_val_affinities.items():
        if pdb_id not in core_set:
            train_val_pdb_ids.append(pdb_id)
            train_val_affinities.append(affinity)
    
    print(f"Train/Val set (after removing core set): {len(train_val_pdb_ids)} complexes")
    
    # Train/Val split (약 10%를 validation set으로)
    np.random.seed(42)
    indices = np.arange(len(train_val_pdb_ids))
    np.random.shuffle(indices)
    
    # 약 10%를 validation set으로
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
    
    # 데이터 수집
    def collect_data(pdb_ids, affinities, split_name):
        data_list = []
        missing_files = []
        
        for pdb_id, affinity in tqdm(zip(pdb_ids, affinities), total=len(pdb_ids), desc=f"Processing {split_name}"):
            pdb_folder = os.path.join(PDBBIND_ROOT, pdb_id)
            protein_pdb = os.path.join(pdb_folder, f"{pdb_id}_protein.pdb")
            ligand_sdf = os.path.join(pdb_folder, f"{pdb_id}_ligand.sdf")
            
            # 파일 존재 확인
            if not os.path.exists(protein_pdb):
                missing_files.append(f"{pdb_id}: protein PDB not found")
                continue
            if not os.path.exists(ligand_sdf):
                missing_files.append(f"{pdb_id}: ligand SDF not found")
                continue
            
            # 단백질 서열 추출
            sequence = extract_protein_sequence_from_pdb(protein_pdb)
            if sequence is None or len(sequence) == 0:
                missing_files.append(f"{pdb_id}: failed to extract sequence")
                continue
            
            # SMILES 추출
            smiles = extract_smiles_from_sdf(ligand_sdf)
            if smiles is None:
                missing_files.append(f"{pdb_id}: failed to extract SMILES")
                continue
            
            if ':' in smiles:
                missing_files.append(f"{pdb_id}: SMILES contains ':'")
            
            data_list.append({
                'PDBID': pdb_id,
                'affinity': affinity,
                'target_sequence': sequence,
                'compound_iso_smiles': smiles
            })
        
        if missing_files:
            print(f"\nWarning: {len(missing_files)} files missing or failed:")
            for msg in missing_files[:10]:  # 처음 10개만 출력
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
    
    # CSV 파일 저장 (PDBID, affinity 형식)
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
    
    # 디렉토리 생성
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'drug_sdf'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'target_pdb'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'drug_smiles'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_SET_DIR, 'target_fasta'), exist_ok=True)
    
    # 모든 데이터 합치기
    all_df = pd.concat([train_df, val_df, test_df])
    
    # 파일 복사 및 생성
    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Copying files"):
        pdb_id = row['PDBID']
        pdb_folder = os.path.join(PDBBIND_ROOT, pdb_id)
        
        # SDF 파일 복사
        src_sdf = os.path.join(pdb_folder, f"{pdb_id}_ligand.sdf")
        dst_sdf = os.path.join(TRAIN_SET_DIR, 'drug_sdf', f"{pdb_id}.sdf")
        if os.path.exists(src_sdf):
            shutil.copy2(src_sdf, dst_sdf)
        
        # PDB 파일 복사
        src_pdb = os.path.join(pdb_folder, f"{pdb_id}_protein.pdb")
        dst_pdb = os.path.join(TRAIN_SET_DIR, 'target_pdb', f"{pdb_id}.pdb")
        if os.path.exists(src_pdb):
            shutil.copy2(src_pdb, dst_pdb)
        
        # SMILES 파일 생성
        smiles_path = os.path.join(TRAIN_SET_DIR, 'drug_smiles', f"{pdb_id}.smi")
        with open(smiles_path, 'w') as f:
            f.write(row['compound_iso_smiles'])
        
        # FASTA 파일 생성
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


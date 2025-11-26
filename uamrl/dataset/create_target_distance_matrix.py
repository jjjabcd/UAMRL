import math
import os
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pdb_dir = 'train_set/target_pdb'
distance_dir = 'train_set/distance_matrix'

def generate_dis_metirc(pdb_dir, distance_dir):
    os.makedirs(distance_dir, exist_ok=True)
    
    parser = PDBParser(QUIET=True)
    
    success_count = 0
    fail_count = 0
    
    if not os.path.exists(pdb_dir):
        print(f"Error: {pdb_dir} directory not found!")
        return
    
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    
    for pdb_file in tqdm(pdb_files, desc="Generating distance matrices"):
        try:
            CA_Metric = []
            coordinate_list = []
            
            pdb_path = os.path.join(pdb_dir, pdb_file)
            pdb_id = pdb_file.replace('.pdb', '')
            dis_path = os.path.join(distance_dir, pdb_id + '.npz')

            if os.path.exists(dis_path):
                success_count += 1
                continue
            
            structure = parser.get_structure(pdb_id, pdb_path)
            
            # C-alpha
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                coord = atom.get_vector()
                                coordinate_list.append([coord[0], coord[1], coord[2]])
                break
            
            if len(coordinate_list) == 0:
                print(f"Warning: No CA atoms found in {pdb_file}")
                fail_count += 1
                continue
            
            # Distance matrix
            for i in range(len(coordinate_list)):
                ca_raw_list = []
                for j in range(len(coordinate_list)):
                    if i == j:
                        ca_raw_list.append(0)
                    else:
                        dist = math.sqrt(
                            (coordinate_list[i][0] - coordinate_list[j][0]) ** 2 + 
                            (coordinate_list[i][1] - coordinate_list[j][1]) ** 2 + 
                            (coordinate_list[i][2] - coordinate_list[j][2]) ** 2
                        )
                        ca_raw_list.append(dist)
                CA_Metric.append(ca_raw_list)
            
            CA_np = np.array(CA_Metric)
            np.savez(dis_path, map=CA_np)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            fail_count += 1
            continue
    
    print(f"\nDistance matrix generation completed!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
        
if __name__ == '__main__':
    generate_dis_metirc(pdb_dir, distance_dir)

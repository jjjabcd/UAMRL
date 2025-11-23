# PDBbind 데이터 전처리 가이드

## 전체 프로세스

### 1단계: PDBbind 데이터 전처리
```bash
cd code
python preprocess_pdbbind.py
```

이 스크립트는 다음을 수행합니다:
- **INDEX 파일 파싱**: Core set과 Refined set 읽기
- **Train/Val/Test Split**:
  - **Test set**: Core set (290개, 표준 벤치마크)
  - **Train/Val set**: Refined set - Core set (약 3767개를 8:2로 분할)
- **데이터 추출 및 변환**:
  - PDB 파일에서 단백질 서열 추출
  - SDF 파일에서 SMILES 추출
  - Affinity 값 추출 (-logKd/Ki)
- **파일 생성**:
  - `data/train_data.csv`, `val_data.csv`, `test_data.csv` (PDBID, affinity)
  - `train_set/drug_sdf/` (SDF 파일들)
  - `train_set/target_pdb/` (PDB 파일들)
  - `train_set/drug_smiles/` (SMILES 파일들)
  - `train_set/target_fasta/` (FASTA 파일들)

### 2단계: Distance Matrix 생성
```bash
python create_target_distance_matrix.py
```

PDB 파일로부터 실제 구조 기반 distance matrix 생성:
- `train_set/distance_matrix/{PDBID}.npz` 파일들 생성

### 3단계: Drug Graph 생성
```bash
python create_drug_graph.py
```

SDF 파일로부터 그래프 구조 생성:
- `train_set/processed/train_data.pt`
- `train_set/processed/val_data.pt`
- `train_set/processed/test_data.pt`

### 4단계: 학습 실행
```bash
python training.py
```

## 데이터 Split 설명

### PDBbind v2016 데이터셋 구조

1. **General Set**: 모든 binding 데이터를 포함한 전체 데이터셋
2. **Refined Set**: 고품질 데이터만 선별 (약 4057개)
   - Resolution ≤ 2.5 Å
   - Binding data quality 기준 통과
3. **Core Set**: 표준 벤치마크 테스트셋 (290개)
   - Refined set에서 단백질 동일성 90% 기준으로 클러스터링
   - 각 클러스터에서 대표 구조 선택

### Split 방법

- **Test Set**: Core set (290개) - 표준 벤치마크
- **Train Set**: Refined set - Core set 중 80% (약 3014개)
- **Val Set**: Refined set - Core set 중 20% (약 753개)

이 방식은 논문에서 일반적으로 사용하는 표준 방법입니다.

## 주의사항

1. **경로 설정**: `preprocess_pdbbind.py`의 경로가 올바른지 확인
   - 기본: `../data/v2016`
   - 실제 경로: `/HDD1/rlawlsgurjh/work/DTA/UAMRL/data/v2016`

2. **필요한 라이브러리**:
   ```bash
   pip install pandas numpy biopython rdkit tqdm
   ```

3. **실행 시간**: 전체 데이터셋 처리에 시간이 걸릴 수 있습니다 (수 시간)

4. **에러 처리**: 일부 파일이 없거나 손상된 경우 스크립트가 자동으로 건너뜁니다

## 출력 파일 구조

```
DTA/UAMRL/
├── data/
│   ├── train_data.csv      (PDBID, affinity)
│   ├── val_data.csv        (PDBID, affinity)
│   └── test_data.csv       (PDBID, affinity)
└── train_set/
    ├── drug_sdf/           ({PDBID}.sdf)
    ├── target_pdb/         ({PDBID}.pdb)
    ├── drug_smiles/        ({PDBID}.smi)
    ├── target_fasta/       ({PDBID}.fasta)
    ├── distance_matrix/    ({PDBID}.npz)
    └── processed/          (train_data.pt, val_data.pt, test_data.pt)
```

## 문제 해결

### 파일을 찾을 수 없는 경우
- PDBbind 데이터 경로 확인
- 파일명 형식 확인: `{pdbid}_protein.pdb`, `{pdbid}_ligand.sdf`

### 메모리 부족
- 배치 처리로 수정 가능
- 일부 데이터만 처리하여 테스트

### SMILES 추출 실패
- RDKit 설치 확인
- SDF 파일 형식 확인


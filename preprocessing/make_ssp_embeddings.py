import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from multiprocessing import Pool
from sklearn.decomposition import PCA

# ========== Config ==========
input_smiles_csv = "smiles.csv"     # Input SMILES file
output_dir = "./ssp_embeddings"                 # Output directory for .pt files
pca_dim = 200                              # PCA dimension
num_processes = 8                          # Parallel workers
# ============================

# 1. Load SMILES
df = pd.read_csv(input_smiles_csv)
drug_ids = df["DrugBank ID"].tolist()
smiles_list = df["SMILES"].tolist()
print(f"[INFO] Loaded {len(drug_ids)} drugs.")

# 2. Generate ECFP4 fingerprints
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) if mol else None

print("[INFO] Generating ECFP4 fingerprints...")
with Pool(processes=num_processes) as pool:
    fingerprints = pool.map(get_fingerprint, smiles_list)

# 3. Compute Tanimoto similarity matrix (upper triangle only)
def compute_similarity(i):
    similarities = np.zeros(len(drug_ids))
    for j in range(i, len(drug_ids)):
        if fingerprints[i] is not None and fingerprints[j] is not None:
            similarities[j] = TanimotoSimilarity(fingerprints[i], fingerprints[j])
    return similarities

print("[INFO] Computing Tanimoto similarity matrix...")
with Pool(processes=num_processes) as pool:
    ssp_matrix = pool.map(compute_similarity, range(len(drug_ids)))

ssp_matrix = np.array(ssp_matrix)
ssp_matrix += ssp_matrix.T - np.diag(np.diag(ssp_matrix))  # Symmetrize
print("[INFO] Similarity matrix shape:", ssp_matrix.shape)

# 4. Apply PCA
print(f"[INFO] Applying PCA to reduce from {ssp_matrix.shape[1]} to {pca_dim} dimensions...")
pca = PCA(n_components=pca_dim)
ssp_pca = pca.fit_transform(ssp_matrix)
print("[INFO] PCA complete. Explained variance ratio:", np.sum(pca.explained_variance_ratio_))

# 5. Save individual .pt files
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"[INFO] Saving embeddings to '{output_dir}'...")
for drug_id, vector in zip(drug_ids, ssp_pca):
    tensor = torch.tensor(vector, dtype=torch.float32)
    output_path = os.path.join(output_dir, f"{drug_id}.pt")
    torch.save(tensor, output_path)

print("✅ All done.")


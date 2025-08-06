import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from multiprocessing import Pool
from sklearn.decomposition import PCA

#config
input_smiles_csv = "smiles.csv"
output_dir = "./ssp_embeddings"
pca_dim = 200
num_processes = 8

# load SMILES
df = pd.read_csv(input_smiles_csv)
drug_ids = df["DrugBank ID"].tolist()
smiles_list = df["SMILES"].tolist()
print(f"Loaded {len(drug_ids)} drugs.")

# ECFP4
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) if mol else None # can be altered to ECFP6

with Pool(processes=num_processes) as pool:
    fingerprints = pool.map(get_fingerprint, smiles_list)

# Tanimoto similarity matrix (upper triangle only)
def compute_similarity(i):
    similarities = np.zeros(len(drug_ids))
    for j in range(i, len(drug_ids)):
        if fingerprints[i] is not None and fingerprints[j] is not None:
            similarities[j] = TanimotoSimilarity(fingerprints[i], fingerprints[j])
    return similarities

with Pool(processes=num_processes) as pool:
    ssp_matrix = pool.map(compute_similarity, range(len(drug_ids)))

ssp_matrix = np.array(ssp_matrix)
ssp_matrix += ssp_matrix.T - np.diag(np.diag(ssp_matrix))
print("Similarity matrix shape:", ssp_matrix.shape)

# PCA
print(f"Applying PCA to reduce from {ssp_matrix.shape[1]} to {pca_dim} dimensions...")
pca = PCA(n_components=pca_dim)
ssp_pca = pca.fit_transform(ssp_matrix)
print("PCA complete. Explained variance ratio:", np.sum(pca.explained_variance_ratio_))

# save .pt files
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving embeddings to '{output_dir}'...")
for drug_id, vector in zip(drug_ids, ssp_pca):
    tensor = torch.tensor(vector, dtype=torch.float32)
    output_path = os.path.join(output_dir, f"{drug_id}.pt")
    torch.save(tensor, output_path)

print("All done.")


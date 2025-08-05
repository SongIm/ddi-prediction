import os
import torch
import pandas as pd

psp_pca_txt_path = "PSP_pca.txt"  # input psp file (after pca to 300 dim)
output_dir = "psp_embeddings"

os.makedirs(output_dir, exist_ok=True)

# load PSP_pca.txt
psp_pca_df = pd.read_csv(psp_pca_txt_path, sep="\t")

# string to number
psp_pca_df.iloc[:, 1:] = psp_pca_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
psp_pca_df = psp_pca_df.dropna()  # process NaN 

# .pt for each drugs
for index, row in psp_pca_df.iterrows():
    drug_id = row.iloc[0]  # first row: DrugBank ID
    embedding = torch.tensor(row.iloc[1:].values.astype(float), dtype=torch.float32)  # rest of the rows: PSP vector
    
    # save .pt
    output_path = os.path.join(output_dir, f"{drug_id}.pt")
    torch.save(embedding, output_path)

print(f"All drug's PSP vector saved at `{output_dir}/`")
print(f"total number of drugs: {len(psp_pca_df)}")


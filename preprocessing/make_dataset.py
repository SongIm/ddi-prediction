import os
import sys
import torch
import pandas as pd

# check command
if len(sys.argv) != 2:
    print(" correct command: python unified_make_dataset_4shapes_remap.py <embedding directory path>")
    sys.exit(1)

embedding_dir = sys.argv[1]
dir_name = os.path.basename(embedding_dir.rstrip("/"))
output_file = f"{dir_name}_dataset.pt"

INTERACTION_CSV = "final_interaction.csv"

df = pd.read_csv(INTERACTION_CSV)

X, y = [], []
drug_a_list, drug_b_list = [], []

# load embeddings
def load_embedding(drug_id):
    path = os.path.join(embedding_dir, f"{drug_id}.pt")
    return torch.load(path) if os.path.exists(path) else None

# process drug pairs
for _, row in df.iterrows():
    drug_a, drug_b, interaction = row["Drug_A"], row["Drug_B"], row["Interaction"]
    emb_a = load_embedding(drug_a)
    emb_b = load_embedding(drug_b)

    if emb_a is None or emb_b is None:
        continue

    pair_embedding = torch.cat((emb_a, emb_b), dim=0)
    X.append(pair_embedding)
    y.append(interaction)
    drug_a_list.append(drug_a)
    drug_b_list.append(drug_b)

# check valid drug pairs
if len(X) == 0:
    print("No valid drug pairs")
    sys.exit(1)

# convert to tensor
X = torch.stack(X)
y = torch.tensor(y)

# remap index
y_remapped = y.clone()
y_remapped[y >= 76] -= 1

# print summary
print(f"\n completed making dataset!")
print(f" - original number of index: {len(torch.unique(y))}")
print(f" - after remapping: {len(torch.unique(y_remapped))}")
print(f" - list of index: {torch.unique(y_remapped).tolist()}")

# save file
torch.save((X, y_remapped, drug_a_list, drug_b_list), output_file)
print(f"\n saved as: {output_file}")


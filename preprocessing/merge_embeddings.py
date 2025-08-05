import os
import sys
import torch

# check parameters
if len(sys.argv) != 3:
    print("correct command: python merge_embeddings.py <combination> <output_dir>")
    sys.exit(1)

combo_name = sys.argv[1]
output_dir = sys.argv[2]

# source directories
base_dirs = {
    "psp": "./psp_embeddings",
    "bio": "./biobert_embeddings",
    "ssp": "./ssp_embeddings",
}

def get_embedding_path(source_key, drug_id):
    return os.path.join(base_dirs[source_key], f"{drug_id}.pt")

os.makedirs(output_dir, exist_ok=True)

# extract key
components = combo_name.split("+")
for key in components:
    if key not in base_dirs:
        print(f"Unknown embedding key: {key}")
        sys.exit(1)

# get common drug list
drug_sets = []
for key in components:
    files = os.listdir(base_dirs[key])
    ids = {f.replace(".pt", "") for f in files if f.endswith(".pt")}
    drug_sets.append(ids)

# merge embeddings using common drug list
common_drug_ids = set.intersection(*drug_sets)
total = len(common_drug_ids)

count = 0
for drug_id in common_drug_ids:
    tensors = []
    valid = True

    for key in components:
        emb_path = get_embedding_path(key, drug_id)
        if not os.path.exists(emb_path):
            print(f"⚠️ missing: {drug_id} - {key} can't find embedding")
            valid = False
            break
        tensors.append(torch.load(emb_path))

    if valid:
        merged = torch.cat(tensors, dim=0)
        torch.save(merged, os.path.join(output_dir, f"{drug_id}.pt"))
        count += 1

print(f"\n '{combo_name}' combination summary:")
print(f" - total common drugs   : {total}")
print(f" - merged drugs         : {count}")
print(f" - saved at             : {output_dir}")


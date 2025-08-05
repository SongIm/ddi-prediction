import torch
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

# check command
if len(sys.argv) != 2:
    print("correct command: python unified_split.py <input_dataset.pt>")
    sys.exit(1)

input_path = sys.argv[1]
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = os.path.dirname(input_path)

data = torch.load(input_path)

# check data structure
if isinstance(data, tuple) and len(data) == 2:
    X, y = data
    has_ids = False
elif isinstance(data, tuple) and len(data) == 4:
    X, y, drug_a_list, drug_b_list = data
    has_ids = True
    drug_a_array = np.array(drug_a_list)
    drug_b_array = np.array(drug_b_list)
else:
    print("data structure not valid. (X, y) or (X, y, drug_a_list, drug_b_list) only")
    sys.exit(1)

# train:test=8:2
train_idx, test_idx = train_test_split(
    range(len(y)), test_size=0.2, random_state=42, stratify=y
)

# split
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

if has_ids:
    drug_a_train = drug_a_array[train_idx].tolist()
    drug_a_test = drug_a_array[test_idx].tolist()
    drug_b_train = drug_b_array[train_idx].tolist()
    drug_b_test = drug_b_array[test_idx].tolist()

# save paths
train_file = os.path.join(output_dir, f"{base_name}_train.pt")
test_file  = os.path.join(output_dir, f"{base_name}_test.pt")

# save files
if has_ids:
    torch.save((X_train, y_train, drug_a_train, drug_b_train), train_file)
    torch.save((X_test, y_test, drug_a_test, drug_b_test), test_file)
else:
    torch.save((X_train, y_train), train_file)
    torch.save((X_test, y_test), test_file)

# print summary
print(f"completed spliting dataset!")
print(f" - input            : {input_path}")
print(f" - Train set saved  : {train_file} ({X_train.shape[0]} samples)")
print(f" - Test set saved   : {test_file}  ({X_test.shape[0]} samples)")


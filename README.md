This repository contains PyTorch training code for predicting Drug–Drug Interactions (DDIs) using combinations of different drug embeddings, including BioBERT, SSP (Structure-based), and PSP (Protein-based) representations.

---

## Input Data Structure

All training and test .pt files must be placed in the same directory (e.g., ./data/), and follow the naming convention:

- `{embedding_name}_train.pt`

- `{embedding_name}_test.pt`

For example:

- `psp_bio_ssp_dataset_train.pt`

- `psp_bio_ssp_dataset_test.pt`

- `ssp_bio_dataset_train.pt`

- `ssp_bio_dataset_test.pt`

Each dataset must be a .pt file created with the following structure:

### (X, y, drug_a_list, drug_b_list)

- `X: Tensor of shape [N, D] — feature matrix (e.g., concatenated embeddings)`

- `y: Tensor of shape [N] — labels from 0 to 78 (79 classes)`

- `drug_a_list, drug_b_list: list of DrugBank IDs used for evaluation or interpretation`

- `drug_a_list and drug_b_list are not used during training, but may be required later for evaluation or post-analysis.`

---

## Training process

### Step 1

```bash
python training/opt.py --embedding_names psp_bio_ssp_dataset ssp_bio_dataset
```

Uses Optuna to search for best hidden_dim, dropout, and learning_rate

Saves result as best_params_<name>.json

### Step 2

```bash
python training/train.py --embedding_names psp_bio_ssp_dataset ssp_bio_dataset
```

Loads best hyperparameters

Trains model with early stopping (≥ 200 epochs)

Saves result as best_model_<name>.pth

### Step 3

```bash
python training/fine_tune.py --embedding_names psp_bio_ssp_dataset ssp_bio_dataset
```

Loads previous best model and parameters

Applies ReduceLROnPlateau for LR decay

Saves result as fine_tuned_best_model_<name>.pth

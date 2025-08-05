This repository contains PyTorch training code for predicting Drug–Drug Interactions (DDIs) using combinations of different drug embeddings, including BioBERT, SSP (Structural Similarity Profile), and PSP (Protein Similarity Profile) representations.

<img width="1168" height="655" alt="Figure1_1" src="https://github.com/user-attachments/assets/e5f799a5-7391-433e-a4a8-7203d8d742c7" />

## Preprocessing

The preprocessing/ directory provides scripts to generate and process drug embeddings into final training-ready datasets.

### Step 1: Generate Individual Embeddings

Create individual embedding files for each drug using the following scripts:

- `make_biobert_embeddings.py – generates BioBERT embeddings from drug names/descriptions`

- `make_ssp_embeddings.py – computes structure-based (SSP) embeddings from SMILES`

- `make_psp_embeddings.py – computes protein-based (PSP) embeddings from interaction profiles`

Each script will generate .pt files (e.g., DB00001.pt) in its respective embedding folder.

### Step 2: Merge Embeddings

To create a unified embedding per drug by concatenating multiple sources (e.g., PSP + BioBERT), use:

```bash
python merge_embeddings.py psp+biobert+ssp merged_output/
```

This merges the .pt files from each source (must have matching drug IDs).

Merged .pt files are saved in merged_output/.

### Step 3: Make Dataset from Drug Pairs

Use interaction annotations (e.g., final_interaction.csv) to build the dataset:

```bash
python make_dataset.py merged_output/
```

### Step 4: Train-Test Split

Split the dataset into 80% train / 20% test:

```bash
python split.py merged_output/{embedding_name}.pt
```

This generates:

- `{embedding_name}_train.pt`

- `{embedding_name}_test.pt`

## Final Input Format for Training

After preprocessing, training and test .pt files must follow this structure:

- `{embedding_name}_train.pt`

- `{embedding_name}_test.pt`

For example:

- `psp_biobert_ssp_dataset_train.pt`

- `psp_biobert_ssp_dataset_test.pt`

- `ssp_biobert_dataset_train.pt`

- `ssp_biobert_dataset_test.pt`

Each dataset must be a .pt file created with the following structure:

### (X, y, drug_a_list, drug_b_list)

- `X: Tensor of shape [N, D] — feature matrix (e.g., concatenated embeddings)`

- `y: Tensor of shape [N] — labels from 0 to 78 (79 classes)`

- `drug_a_list, drug_b_list: list of DrugBank IDs used for evaluation or interpretation`

### **Note:**
drug_a_list and drug_b_list are not used during training, but may be required later for evaluation or post-analysis.

---

## Training

### Step 1: Hyperparameter Optimization

```bash
python training/opt.py --embedding_names psp_biobert_ssp_dataset ssp_biobert_dataset
```

Uses Optuna to search for best hidden_dim, dropout, and learning_rate

### Step 2: Model Training



```bash
python training/train.py --embedding_names psp_biobert_ssp_dataset ssp_biobert_dataset
```

Loads best hyperparameters

Trains model with early stopping (≥ 200 epochs)

### Step 3: Fine-tuning (Additional Training)

```bash
python training/fine_tune.py --embedding_names psp_biobert_ssp_dataset ssp_biobert_dataset
```

Loads previous best model and parameters

Applies learning rate decay (ReduceLROnPlateau)

---

## Output Files

- ```opt.py:```
  
 best_params_{embedding_name}.json – best hyperparameters found by Optuna

- ```train.py:```

best_model_{embedding_name}.pth – model saved with best validation accuracy

{embedding_name}_result.csv - summary including input dim, epoch count, best validation accuracy, and test accuracy

- ```fine_tune.py:```

fine_tuned_best_model_{embedding_name}.pth – fine-tuned model with LR decay

fine_tuning_results_{embedding_name}.csv - summary after fine-tuning, including validation and test accuracies

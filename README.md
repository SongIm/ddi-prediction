This repository contains PyTorch training code for predicting Drug–Drug Interactions (DDIs) using combinations of different drug embeddings, including BioBERT, SSP (Structure-based), and PSP (Protein-based) representations.

Each dataset must be a .pt file created with the following structure:

#### (X, y, drug_a_list, drug_b_list)

X: tensor of shape [N, D] (concatenated drug embeddings)

y: tensor of shape [N] with class labels (0~78)

drug_a_list, drug_b_list: Drugbank ID

Files should be named like: concat_embeddings_train.pt, concat_embeddings_test.pt

### Training process

#### 1. python training/opt.py

Search for optimal hidden size, dropout, and learning rate using Optuna and save best parameters as best_params_*.json.

#### 2. python training/train.py

Uses the best hyperparameters from Optuna and trains the model with early stopping. Final models are saved as best_model_*.pth.

#### 3. python training/fine_tune.py

Loads the best model and continues training with learning rate decay (ReduceLROnPlateau).

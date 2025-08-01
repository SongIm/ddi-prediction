import torch
import torch.nn as nn
import optuna
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# embedding list
embedding_names = [
    "psp_bio_ssp_dataset",
    "ssp_bio_dataset"
]

for emb_name in embedding_names:
    print(f"\n Running Optuna for {emb_name}...")

    # load data
    data = torch.load(f"{emb_name}_train.pt")
    X_train_full = data[0].float()
    y_train_full = data[1].long()

    # train/val split
    X_train_np = X_train_full.cpu().numpy()
    y_train_np = y_train_full.cpu().numpy()

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
    )

    X_train_torch = torch.tensor(X_train_split).to(device)
    y_train_torch = torch.tensor(y_train_split).to(device)
    X_val_torch = torch.tensor(X_val_split).to(device)
    y_val_torch = torch.tensor(y_val_split).to(device)

    input_dim = X_train_torch.shape[1]

    # model definition
    class DDI_Predictor(nn.Module):
        def __init__(self, input_dim, hidden_dim=1024, dropout=0.3, output_dim=79):
            super(DDI_Predictor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            return self.network(x)

    # DataLoader
    def get_dataloaders(batch_size=64):
        train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    # Optuna objective
    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 512, 2048, step=256)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

        train_loader, val_loader = get_dataloaders()
        model = DDI_Predictor(input_dim, hidden_dim, dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(10):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # val accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total
        print(f"Trial val_acc: {val_acc:.4f}")
        return val_acc

    # run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Hyperparameters:", study.best_params)

    # save best params each embeddings
    with open(f"best_params_{emb_name}.json", "w") as f:
        json.dump(study.best_params, f)
    print(f"Best hyperparameters saved to best_params_{emb_name}.json")


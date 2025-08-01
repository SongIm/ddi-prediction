import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DDI model using pre-trained weights and learning rate decay.")
    parser.add_argument(
        "--embedding_names",
        nargs="+",
        required=True,
        help="List of embedding names (prefix only, without _train.pt)."
    )
    return parser.parse_args()

args = parse_args()
embedding_names = args.embedding_names

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

results = []

for emb_name in embedding_names:
    print(f"\n Fine-tuning for {emb_name} with LR decay (stratified split)...")

    # best hyperparameter load
    with open(f"best_params_{emb_name}.json") as f:
        params = json.load(f)
    hidden_dim = params['hidden_dim']
    dropout = params['dropout']
    lr = params['lr']
    print(f"✅ Loaded best params: {params}")

    # dataset load
    trainset = torch.load(f"{emb_name}_train.pt")
    testset = torch.load(f"{emb_name}_test.pt")
    X_train_full, y_train_full = trainset[0], trainset[1]
    X_test, y_test = testset[0], testset[1]

    if y_train_full.min() == 1:
        y_train_full -= 1
        y_test -= 1

    input_dim = X_train_full.shape[1]
    output_dim = int(y_train_full.max().item() + 1)

    # stratified train/val split
    X_train_np = X_train_full.cpu().numpy()
    y_train_np = y_train_full.cpu().numpy()
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_np, y_train_np,
        test_size=0.2,
        random_state=42,
        stratify=y_train_np
    )

    X_train = torch.tensor(X_train_split).to(device)
    y_train = torch.tensor(y_train_split).to(device)
    X_val = torch.tensor(X_val_split).to(device)
    y_val = torch.tensor(y_val_split).to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512)

    # model definition
    model = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim)
    ).to(device)

    # Load previous best model weights
    best_model_path = f"best_model_{emb_name}.pth"
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded previous best model: {best_model_path}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    # fine-tuning
    num_epochs = 100
    best_val_acc = 0
    patience = 20
    wait = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        y_pred = []
        y_val_true = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                out = model(Xb)
                y_pred.append(out.argmax(dim=1).cpu())
                y_val_true.append(yb.cpu())
        y_pred = torch.cat(y_pred)
        y_val_true = torch.cat(y_val_true)
        acc = accuracy_score(y_val_true, y_pred)

        scheduler.step(acc)

        print(f"Epoch {epoch}, Val Acc: {acc:.4f}, Best Val Acc: {best_val_acc:.4f}, Wait: {wait}/{patience}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if acc >= best_val_acc:
            best_val_acc = acc
            wait = 0
            torch.save(model.state_dict(), f"fine_tuned_best_model_{emb_name}.pth")
            print(f"New fine-tuned best model saved. Best Val Acc updated to {best_val_acc:.4f}")
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping triggered.")
            break

    print(f"Fine-tuning complete for {emb_name}. Best Val Acc: {best_val_acc:.4f}")

    # test
    model.load_state_dict(torch.load(f"fine_tuned_best_model_{emb_name}.pth"))
    model.eval()
    y_pred_test = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            out = model(Xb)
            y_pred_test.append(out.argmax(dim=1).cpu())
    y_pred_test = torch.cat(y_pred_test)
    test_acc = accuracy_score(y_test.cpu(), y_pred_test)
    print(f"Final Test Accuracy for {emb_name} after fine-tuning: {test_acc:.4f}")

    results.append([emb_name, input_dim, epoch, best_val_acc, test_acc])

# save result
df_results = pd.DataFrame(results, columns=['Feature Combination', 'Input dim', 'Epochs', 'Best Val Acc', 'Test Acc'])
print(df_results)
df_results.to_csv('fine_tuning_results_all.csv', index=False)
print("All results saved to fine_tuning_results_all.csv")
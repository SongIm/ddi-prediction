import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# embedding list
embedding_names = [
    "psp_bio_ssp_dataset",
    "ssp_bio_dataset"
]

results = []

# loop over each embedding combination
for emb_name in embedding_names:
    print(f"\n Training for {emb_name} ...")

    # load best hyperparameters
    with open(f"best_params_{emb_name}.json") as f:
        best_params = json.load(f)
    hidden_dim = best_params["hidden_dim"]
    dropout = best_params["dropout"]
    lr = best_params["lr"]

    # load train/test dataset
    trainset = torch.load(f"{emb_name}_train.pt")
    testset = torch.load(f"{emb_name}_test.pt")
    X_train_full, y_train_full = trainset[0], trainset[1]
    X_test, y_test = testset[0], testset[1]

    # check label range
    if y_train_full.min() == 1:
        y_train_full -= 1
        y_test -= 1

    input_dim = X_train_full.shape[1]
    output_dim = 79

    # train/validation stratified split
    X_train_np = X_train_full.numpy()
    y_train_np = y_train_full.numpy()
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # early stopping setup
    best_val_acc = 0
    patience = 20
    wait = 0

    for epoch in range(1, 301):  # train 200 epoch before starting early stopping
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # validation accuracy
        model.eval()
        y_pred = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                out = model(Xb)
                y_pred.append(out.argmax(dim=1).cpu())
        y_pred = torch.cat(y_pred)
        acc = accuracy_score(y_val.cpu(), y_pred)

        print(f"Epoch {epoch}, Val Acc: {acc:.4f}, Best: {best_val_acc:.4f}, Wait: {wait}/{patience}")

        # early stopping
        if acc >= best_val_acc:
            best_val_acc = acc
            wait = 0
            torch.save(model.state_dict(), f"best_model_{emb_name}.pth")
        else:
            if epoch >= 200:
                wait += 1

        if wait >= patience:
            print("Early stopping triggered.")
            break

    # test accuracy
    model.eval()
    y_pred_test = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            out = model(Xb)
            y_pred_test.append(out.argmax(dim=1).cpu())
    y_pred_test = torch.cat(y_pred_test)
    test_acc = accuracy_score(y_test.cpu(), y_pred_test)
    print(f"Final Test Accuracy for {emb_name}: {test_acc:.4f}")

    results.append([emb_name, input_dim, epoch, best_val_acc, test_acc])

# save result as csv
df_results = pd.DataFrame(results, columns=['Feature Combination', 'Input dim', 'Epochs', 'Best Val Accuracy', 'Test Accuracy'])
print(df_results)
df_results.to_csv('ssp_bio_and_psp_ssp_bio_result.csv', index=False)
print("All results saved")
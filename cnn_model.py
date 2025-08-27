# ==================
# cnn_model.py
# ==================

"""Simple 1D CNN baseline for sequence classification.
Uses two convolutional blocks followed by global average pooling and a
final linear layer. Training routine mirrors IGANN's interface so the
same data pipeline can be reused.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score
import pandas as pd


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SimpleCNN(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):  # x: (B, L)
        x = x.unsqueeze(1)  # (B,1,L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lr: float = 1e-3,
    max_epochs: int = 200,
    batch_size: int = 128,
    seed: int = 42,
    patience: int = 20,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    model = SimpleCNN(input_len=X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_auc = -1.0
    best_state = None
    wait = 0

    print("[CNN] Start training...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_scores, val_labels = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                proba = torch.sigmoid(model(xb)).cpu().numpy()
                val_scores.append(proba)
                val_labels.append(yb.numpy().astype(int))
            y_val_score = np.concatenate(val_scores)
            y_val_true = np.concatenate(val_labels)
            val_auc = roc_auc_score(y_val_true, y_val_score)

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:02d} - loss: {avg_loss:.4f} - val_auc: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Determine threshold on validation set
    model.eval()
    with torch.no_grad():
        val_scores = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            val_scores.append(torch.sigmoid(model(xb)).cpu().numpy())
        y_val_score = np.concatenate(val_scores)
    y_val_true = y_val
    p_arr, r_arr, thr_arr = precision_recall_curve(y_val_true, y_val_score)
    f1_arr = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-12)
    idx = f1_arr.argmax()
    best_thr = thr_arr[idx] if idx < len(thr_arr) else 0.5

    with torch.no_grad():
        test_scores, test_labels = [], []
        for xb, yb in test_loader:
            xb = xb.to(device)
            proba = torch.sigmoid(model(xb)).cpu().numpy()
            test_scores.append(proba)
            test_labels.append(yb.numpy().astype(int))
        y_score = np.concatenate(test_scores)
        y_true = np.concatenate(test_labels)
        y_pred = (y_score >= best_thr).astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    eps = 1e-12
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    prec = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    fpr = FP / (FP + TN + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    auc = roc_auc_score(y_true, y_score)

    temp = pd.DataFrame({'label_0': y_true, 'label_1': 1 - y_true,
                         'preds_0': y_score, 'preds_1': 1 - y_score})
    map40 = mean_average_precision([
        list(temp.sort_values(by='preds_0', ascending=False).label_0[:40]),
        list(temp.sort_values(by='preds_1', ascending=False).label_1[:40])
    ])

    best_metrics = {
        'threshold': float(best_thr),
        'acc': float(acc),
        'prec': float(prec),
        'rec': float(rec),
        'fpr': float(fpr),
        'map40': float(map40),
        'auc': float(auc),
        'f1': float(f1),
    }

    print(f"[CNN] Training complete. Best Val AUC: {best_auc:.4f}")
    return model, best_metrics

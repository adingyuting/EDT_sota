# ==================
# igann_model.py
# ==================

"""PyTorch IGANN implementation: per-feature small subnetworks + linear term.
Loss: BCEWithLogits + λ_task * L1_on_linear + λ_BG * L2_on_subnets.
Only the L1/L2 regularizations from the paper are used; no extra tricks."""

# --- file: igann_model.py ---
from __future__ import annotations
import math
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


class AdditiveSubNet(nn.Module):
    """Small MLP applied to a single feature with dropout."""

    def __init__(self, hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1):  # x1: (B,1)
        h = F.relu(self.fc1(x1))
        h = self.drop(F.relu(self.fc2(h)))
        out = self.fc3(h)  # (B,1)
        return out


class IGANN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.linear_a = nn.Parameter(torch.zeros(n_features))
        self.bias_b = nn.Parameter(torch.zeros(1))
        self.subnets = nn.ModuleList([
            AdditiveSubNet(hidden=hidden, dropout=dropout) for _ in range(n_features)
        ])

    def forward(self, x):  # x: (B, F)
        # linear term <a, x>
        lin = (x * self.linear_a).sum(dim=1, keepdim=True)  # (B,1)
        # additive subnets
        add_terms = []
        for j in range(self.n_features):
            add_terms.append(self.subnets[j](x[:, j:j+1]))
        add_sum = torch.stack(add_terms, dim=0).sum(dim=0)  # (B,1)
        logits = lin + self.bias_b + add_sum  # (B,1)
        return logits.squeeze(1)


def _l1_linear(params: torch.Tensor) -> torch.Tensor:
    return params.abs().sum()


def _l2_subnets(model: IGANN) -> torch.Tensor:
    reg = 0.0
    for sn in model.subnets:
        for p in sn.parameters():
            reg = reg + (p.pow(2).sum())
    return reg


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def train_igann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lr: float = 1e-3,
    max_epochs: int = 200,
    batch_size: int = 128,
    hidden: int = 16,
    dropout: float = 0.0,
    lambda_task: float = 1e-4,  # L1 on linear
    lambda_bg: float = 1e-5,    # L2 on subnets
    seed: int = 42,
    patience: int = 20,
    weight_decay: float = 0.0,
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    model = IGANN(n_features=X_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val).float(),   torch.from_numpy(y_val).float())
    test_ds  = TensorDataset(torch.from_numpy(X_test).float(),  torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_auc = -1.0
    best_state = None
    wait = 0

    print("[IGANN] Start training...")
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_bce = F.binary_cross_entropy_with_logits(logits, yb)
            reg_l1 = lambda_task * _l1_linear(model.linear_a)
            reg_l2 = lambda_bg * _l2_subnets(model)
            loss = loss_bce + reg_l1 + reg_l2
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
        print(f"Epoch {epoch+1:02d} - loss: {avg_loss:.4f} - val_auc: {val_auc:.4f}")

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

    # Evaluate on test set once
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

    print(f"[IGANN] Training complete. Best Val AUC: {best_auc:.4f}")
    return model, best_metrics



def predict_proba(model: IGANN, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        proba = torch.sigmoid(logits).cpu().numpy()
    return proba

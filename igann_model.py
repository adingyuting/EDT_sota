# ==================
# igann_model.py
# ==================

"""PyTorch IGANN implementation: per-feature small subnetworks + linear term.
Loss: BCEWithLogits + λ_task * L1_on_linear + λ_BG * L2_on_subnets.
"""

# --- file: igann_model.py ---
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


class AdditiveSubNet(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x1):  # x1: (B,1)
        h = F.relu(self.fc1(x1))
        out = self.fc2(h)  # (B,1)
        return out


class IGANN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 16):
        super().__init__()
        self.n_features = n_features
        self.linear_a = nn.Parameter(torch.zeros(n_features))
        self.bias_b = nn.Parameter(torch.zeros(1))
        self.subnets = nn.ModuleList([AdditiveSubNet(hidden=hidden) for _ in range(n_features)])

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


def train_igann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    max_epochs: int = 200,
    batch_size: int = 128,
    hidden: int = 16,
    lambda_task: float = 1e-4,  # L1 on linear
    lambda_bg: float = 1e-5,    # L2 on subnets
    seed: int = 42,
    patience: int = 20,
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sanitize inputs to avoid NaNs during loss computation
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    model = IGANN(n_features=X_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ========= 关键：按训练集类比计算 pos_weight =========
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    # 避免除零 & 过大；经验上 cap 到 [1, 50]
    pw = max(1.0, min(50.0, (n_neg + 1e-6) / (n_pos + 1e-6)))
    pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val).float(),   torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_f1 = -1.0
    best_state = None
    wait = 0

    print("[IGANN] Start training...")
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # ====== 关键：带 pos_weight 的 BCE ======
            loss_bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            reg_l1 = lambda_task * _l1_linear(model.linear_a)
            reg_l2 = lambda_bg * _l2_subnets(model)
            loss = loss_bce + reg_l1 + reg_l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # 验证：以 F1 作为早停指标（默认阈值先用 0.5）
        model.eval()
        with torch.no_grad():
            preds, gts = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                proba = torch.sigmoid(model(xb)).cpu().numpy()
                preds.append((proba >= 0.5).astype(int))
                gts.append(yb.numpy().astype(int))
            y_pred = np.concatenate(preds)
            y_true = np.concatenate(gts)
            f1 = f1_score(y_true, y_pred, zero_division=0)

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{max_epochs} - loss: {avg_loss:.4f} - val_f1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[IGANN] Training complete. Best val F1: {best_f1:.4f}")
    return model



def predict_proba(model: IGANN, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        proba = torch.sigmoid(logits).cpu().numpy()
    return proba

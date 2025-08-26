# Paper: N. Nandhini et al., Energy & Buildings (2025)
# Notes:
# - This monolithic file lists the contents of multiple project files. Create the directory
#   structure below and copy each section into its respective .py file.
# - Data expected: data/feature.csv (N_users x T_timepoints), data/label.csv (N_users,)
# - Pipeline: RN-SMOTE → RLKF → MSGVT(6 stats) → IGANN (+SBOA) → Metrics + SHAP
# - Repro switches via main.py flags; sensible defaults provided.

# ============================
# requirements.txt (reference)
# ============================
# numpy>=1.24
# pandas>=2.0
# scikit-learn>=1.4
# imbalanced-learn>=0.12
# torch>=2.1
# scipy>=1.10
# pywavelets>=1.5
# shap>=0.44
# matplotlib>=3.8

# =================
# data_loader.py
# =================

"""Data loading utilities for SGCC-formatted CSVs.
Assumes:
- feature.csv: (N_users, T_timepoints)
- label.csv: (N_users,)
"""

# --- file: data_loader.py ---
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_sgcc_csv(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    # 使用绝对路径
    data_dir = os.path.abspath(data_dir)

    X = pd.read_csv(r'D:\学术工作\pythonProject\Igann_Etd\data\features.csv').values
    y_df = pd.read_csv(r'D:\学术工作\pythonProject\Igann_Etd\data\label.csv')
    y = y_df.values.squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)
    assert X.shape[0] == y.shape[0], f"X users {X.shape[0]} != y users {y.shape[0]}"
    return X.astype(float), y.astype(int)


def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_tr, X_te, y_tr, y_te


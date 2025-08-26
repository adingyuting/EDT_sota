# ================
# rn_smote.py
# ================

"""RN-SMOTE: Reduced Noise SMOTE for class balancing.
Implementation approach:
1) Identify noisy minority samples via kNN-based density (average neighbor distance).
2) Drop top-q% noisiest minority samples.
3) Apply SMOTE to reach target minority/majority ratio.
This mirrors the paper's intent to reduce noise before/after oversampling.
"""

# --- file: rn_smote.py ---
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE


def _minority_mask(y: np.ndarray, minority_label: int = 1) -> np.ndarray:
    return (y == minority_label)


def _drop_noisy_minority(X: np.ndarray, y: np.ndarray, minority_label: int = 1, k: int = 5, drop_frac: float = 0.1):
    msk = _minority_mask(y, minority_label)
    X_min = X[msk]
    if X_min.shape[0] <= k + 5:
        return X, y  # not enough to prune
    nbrs = NearestNeighbors(n_neighbors=min(k+1, X_min.shape[0]-1), metric='euclidean')
    nbrs.fit(X_min)
    dists, _ = nbrs.kneighbors(X_min)
    # skip self distance in column 0
    avg_dist = dists[:, 1:].mean(axis=1)
    n_drop = max(0, int(drop_frac * X_min.shape[0]))
    if n_drop == 0:
        return X, y
    drop_idx_local = np.argsort(avg_dist)[-n_drop:]  # largest distance ⇒ noisier
    keep_local = np.ones(X_min.shape[0], dtype=bool)
    keep_local[drop_idx_local] = False  # mark drops as False
    # Map back to global indices
    global_idx = np.where(msk)[0]
    keep_global = np.ones(X.shape[0], dtype=bool)
    keep_global[global_idx[~keep_local]] = False
    return X[keep_global], y[keep_global]


def apply_rn_smote(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: int = 1,
    k: int = 5,
    drop_frac: float = 0.1,
    target_ratio: float = 1.0,
    random_state: int = 42,
):
    """Apply Reduced-Noise SMOTE.
    target_ratio = desired minority/majority count ratio after oversampling (≤1 for undersample majority? here we oversample minority to match ratio).
    """
    Xp, yp = _drop_noisy_minority(X, y, minority_label, k=k, drop_frac=drop_frac)
    # Compute sampling strategy for SMOTE
    n_min = (yp == minority_label).sum()
    n_maj = (yp != minority_label).sum()
    desired_min = int(target_ratio * n_maj)
    sampling_strategy = None
    if desired_min > n_min:
        sampling_strategy = desired_min / n_maj  # ratio form accepted by SMOTE (minority:majority)
    else:
        sampling_strategy = 'auto'
    sm = SMOTE(k_neighbors=max(1, k), sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(Xp, yp)
    return X_res, y_res

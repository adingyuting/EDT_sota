
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict

def load_sgcc_csv(feature_csv: str, label_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load SGCC-style data.
    feature_csv: shape (N_users, T) numeric matrix (NaN allowed for missing)
    label_csv: shape (N_users,) with 0 (honest) or 1 (dishonest)
    Returns:
        X: (N, T) float32
        y: (N,) int64
    """
    X = pd.read_csv(feature_csv).values.astype(np.float32)
    y_raw = pd.read_csv(label_csv, header=None).values.squeeze()
    y = y_raw.astype(np.int64)
    return X, y

def train_test_split_stratified(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split keeping class ratios."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return Xtr, Xte, ytr, yte

def standardize_per_series(X: np.ndarray) -> np.ndarray:
    """Z-normalize each series independently along time axis."""
    X = X.astype(np.float32)
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

def pad_or_truncate(X: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with nan or truncate to a fixed length along time axis."""
    N, T = X.shape
    if T == target_len:
        return X
    if T > target_len:
        return X[:, :target_len]
    # pad
    pad = np.full((N, target_len - T), np.nan, dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)

def make_tensor_data_from_modes(modes: np.ndarray, stack: bool = True) -> np.ndarray:
    """Convert VMD modes into model input tensor.
    modes: (N, K, T) where K is number of modes (e.g., 6). We will:
      - drop two highest-frequency modes (indices 0 and 1 assume sorted high->low or user-provided order)
      - keep remaining 4 modes => shape (N, 4, T)
      - if stack=True: return (N, T, 4); else: return (N, T, 1) summing the 4 modes.
    """
    assert modes.ndim == 3
    N, K, T = modes.shape
    if K < 4:
        raise ValueError("Expect at least 4 modes from VMD.")
    # Assume input modes are ordered from high-frequency to low-frequency.
    keep = modes[:, 2:, :]  # keep last K-2 modes
    if keep.shape[1] > 4:
        # If more than 4 kept, choose the 4 lowest-frequency modes
        keep = keep[:, -4:, :]
    if stack:
        x = np.transpose(keep, (0, 2, 1))  # (N, T, 4)
    else:
        x = np.sum(keep, axis=1, keepdims=True)  # (N, 1, T)
        x = np.transpose(x, (0, 2, 1))  # (N, T, 1)
    return x.astype(np.float32)

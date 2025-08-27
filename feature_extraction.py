# ========================
# feature_extraction.py
# ========================

"""MSGVT-inspired statistical feature extraction.

The feature set strictly follows the six statistics used in the paper:
Entropy, Mean, Variance, Kurtosis, lag-1 Autocorrelation and Contrast.
Optionally, the raw signal is first projected onto a wavelet basis
(``db4`` up to level 4) before computing the statistics, but no
additional hand-crafted features are appended.
"""

# --- file: feature_extraction.py ---
from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis
import pywt


def _entropy_from_signal(x: np.ndarray, bins: int = 64, eps: float = 1e-12) -> float:
    hist, _ = np.histogram(x[~np.isnan(x)], bins=bins, density=True)
    p = hist / (hist.sum() + eps)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _lag1_autocorr(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size < 3:
        return 0.0
    mu = x.mean()
    num = np.sum((x[:-1] - mu) * (x[1:] - mu))
    den = np.sum((x - mu) ** 2) + 1e-12
    return float(num / den)


def msgvt_stats_for_series(x: np.ndarray, use_wavelet: bool = True) -> np.ndarray:
    """Return the six MSGVT statistics for a single series."""

    if use_wavelet:
        coeffs = pywt.wavedec(x, 'db4', level=min(4, int(np.log2(max(2, x.size)))))
        sig = np.concatenate([c.ravel() for c in coeffs])
    else:
        sig = x

    ent = _entropy_from_signal(sig)
    mean = float(np.nanmean(sig))
    var = float(np.nanvar(sig, ddof=1)) if sig.size > 1 else 0.0
    kur = float(kurtosis(sig, fisher=False, nan_policy='omit')) if sig.size > 3 else 3.0
    if np.isnan(kur):
        kur = 3.0
    corr = _lag1_autocorr(sig)
    contrast = float(np.nanstd(sig, ddof=1)) if sig.size > 1 else 0.0

    feats = np.array([ent, mean, var, kur, corr, contrast], dtype=float)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_msgvt_features(X: np.ndarray, use_wavelet: bool = True) -> np.ndarray:
    feats = np.vstack([msgvt_stats_for_series(row, use_wavelet=use_wavelet) for row in X])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

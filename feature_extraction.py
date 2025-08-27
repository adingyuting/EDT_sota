# ========================
# feature_extraction.py
# ========================

"""MSGVT-inspired statistical feature extraction.

Along with the six classical MSGVT statistics (Entropy, Mean, Variance,
Kurtosis, lag-1 Autocorrelation and Contrast) this module can enrich the
feature set with two groups of descriptors that help IGANN capture more
temporal structure without resorting to CNNs:

* **Wavelet band energies** – energy of each wavelet coefficient band
  when decomposed with ``db4`` up to level 4.
* **Rolling-window statistics** – average rolling mean and variance over
  fixed windows (lengths 10 and 20).

These additional features are inexpensive to compute yet provide
information about local oscillations and scale-specific energy
distribution, which often boosts AUC by a few points in practice.
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
    """Return MSGVT stats plus optional wavelet/rolling descriptors."""

    if use_wavelet:
        coeffs = pywt.wavedec(x, 'db4', level=min(4, int(np.log2(max(2, x.size)))))
        sig = np.concatenate([c.ravel() for c in coeffs])
    else:
        coeffs = []
        sig = x

    ent = _entropy_from_signal(sig)
    mean = float(np.nanmean(sig))
    var = float(np.nanvar(sig, ddof=1)) if sig.size > 1 else 0.0
    kur = float(kurtosis(sig, fisher=False, nan_policy='omit')) if sig.size > 3 else 3.0
    if np.isnan(kur):
        kur = 3.0
    corr = _lag1_autocorr(sig)
    contrast = float(np.nanstd(sig, ddof=1)) if sig.size > 1 else 0.0
    base_feats = [ent, mean, var, kur, corr, contrast]

    # Wavelet band energies (mean squared coefficient per level)
    wavelet_feats = []
    for c in coeffs:
        if c.size == 0:
            wavelet_feats.append(0.0)
        else:
            wavelet_feats.append(float(np.mean(c ** 2)))

    # Rolling statistics on the original signal
    def _rolling_stats(arr: np.ndarray, win: int):
        if arr.size < win:
            return 0.0, 0.0
        shape = (arr.size - win + 1, win)
        strides = (arr.strides[0], arr.strides[0])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        means = windows.mean(axis=1)
        vars = windows.var(axis=1)
        return float(np.mean(means)), float(np.mean(vars))

    roll_feats = []
    for w in (10, 20):
        m, v = _rolling_stats(sig, w)
        roll_feats.extend([m, v])

    feats = np.array(base_feats + wavelet_feats + roll_feats, dtype=float)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_msgvt_features(X: np.ndarray, use_wavelet: bool = True) -> np.ndarray:
    feats = np.vstack([msgvt_stats_for_series(row, use_wavelet=use_wavelet) for row in X])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

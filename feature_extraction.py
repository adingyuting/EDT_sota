# ========================
# feature_extraction.py
# ========================

"""MSGVT-inspired statistical feature extraction.
For each time series we compute classical statistics plus

- Wavelet band energies to retain coarse-to-fine spectral information
- Rolling-window statistics to capture local variation

The final feature vector contains:
Entropy, Mean, Variance, Kurtosis, lag-1 Autocorrelation, Contrast,
band energies for each wavelet level, rolling mean std, rolling std mean.
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
    """Return statistics for a single series.

    Besides the classical MSGVT features, this function appends:
    - Energy of each wavelet coefficient band
    - Rolling-window stats (std of rolling mean, mean of rolling std)
    """

    if use_wavelet:
        coeffs = pywt.wavedec(x, 'db4', level=min(4, int(np.log2(max(2, x.size)))))
        # concatenate all coeffs to build a surrogate signal for global stats
        ww = np.concatenate([c.ravel() for c in coeffs])
        sig = ww
        # wavelet band energies
        energies = [float(np.mean(c ** 2)) for c in coeffs]
    else:
        sig = x
        energies = []

    ent = _entropy_from_signal(sig)
    mean = float(np.nanmean(sig))
    var = float(np.nanvar(sig, ddof=1)) if sig.size > 1 else 0.0
    kur = float(kurtosis(sig, fisher=False, nan_policy='omit')) if sig.size > 3 else 3.0
    if np.isnan(kur):
        kur = 3.0
    corr = _lag1_autocorr(sig)
    contrast = float(np.nanstd(sig, ddof=1)) if sig.size > 1 else 0.0

    # rolling-window statistics on original signal (window=5)
    w = 5
    if x.size >= w:
        roll_mean = np.convolve(x, np.ones(w) / w, mode='valid')
        roll_std = np.array([np.std(x[i:i+w]) for i in range(x.size - w + 1)])
        roll_mean_std = float(np.std(roll_mean))
        roll_std_mean = float(np.mean(roll_std))
    else:
        roll_mean_std = 0.0
        roll_std_mean = 0.0

    feats = np.array([ent, mean, var, kur, corr, contrast, *energies, roll_mean_std, roll_std_mean], dtype=float)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_msgvt_features(X: np.ndarray, use_wavelet: bool = True) -> np.ndarray:
    feats = np.vstack([msgvt_stats_for_series(row, use_wavelet=use_wavelet) for row in X])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

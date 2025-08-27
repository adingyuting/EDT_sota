
import numpy as np
from typing import Tuple
try:
    from vmdpy import VMD
except Exception as e:
    VMD = None

def vmd_decompose(x: np.ndarray, K: int = 6, alpha: float = 2000, tau: float = 0.0, DC: int = 0, init: int = 1, tol: float = 1e-7) -> np.ndarray:
    """Decompose a 1D signal into K modes using VMD (if available).
    Returns modes of shape (K, T). If vmdpy not installed, falls back to simple FFT band split approximation.
    """
    x = np.asarray(x, dtype=np.float64)
    if VMD is not None:
        u, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)
        return u  # (K, T)
    # Fallback: naive FFT band splitting into K bands
    T = x.shape[-1]
    Xf = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(T, d=1.0)
    maxf = freqs.max() + 1e-8
    bands = np.linspace(0, maxf, K+1)
    modes = []
    for i in range(K):
        f_low, f_high = bands[i], bands[i+1]
        mask = (freqs >= f_low) & (freqs < f_high)
        Xband = np.zeros_like(Xf)
        Xband[mask] = Xf[mask]
        mode = np.fft.irfft(Xband, n=T).real
        modes.append(mode)
    return np.stack(modes, axis=0)

def batch_vmd(X: np.ndarray, K: int = 6) -> np.ndarray:
    """Apply VMD to each series in a batch. X shape (N, T) -> (N, K, T)."""
    N, T = X.shape
    modes = np.zeros((N, K, T), dtype=np.float32)
    for i in range(N):
        m = vmd_decompose(X[i], K=K)
        modes[i] = m.astype(np.float32)
    return modes

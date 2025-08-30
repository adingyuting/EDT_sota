import numpy as np
from vmdpy import VMD


def vmd_decompose(
    x: np.ndarray,
    K: int = 6,
    alpha: float = 2000,
    tau: float = 0.0,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7,
) -> np.ndarray:
    """Decompose a 1D signal into ``K`` modes using VMD.

    Parameters mirror the formulation in the paper: ``alpha`` is the penalty
    coefficient (η), ``tau`` the dual ascent time-step, and ``tol`` the
    convergence tolerance ε from Eq. (6).
    """
    x = np.asarray(x, dtype=np.float64)
    u, _, _ = VMD(x, alpha, tau, K, DC, init, tol)
    return u  # (K, T)


def batch_vmd(
    X: np.ndarray,
    K: int = 6,
    alpha: float = 2000,
    tau: float = 0.0,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7,
) -> np.ndarray:
    """Apply VMD to each series in a batch.

    Parameters
    ----------
    X : array-like, shape (N, T)
        Batch of signals.
    Returns
    -------
    modes : ndarray, shape (N, K, T)
        Decomposed modes for each series.
    """
    N, T = X.shape
    modes = np.zeros((N, K, T), dtype=np.float32)
    for i in range(N):
        m = vmd_decompose(X[i], K=K, alpha=alpha, tau=tau, DC=DC, init=init, tol=tol)
        modes[i] = m.astype(np.float32)
    return modes

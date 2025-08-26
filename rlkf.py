# ============
# rlkf.py
# ============

"""Reverse Lognormal Kalman Filter (RLKF) for per-user time series cleanup.
We implement a 1D Kalman filter in the transformed space z_t = ln(ξ_f - x_t),
with ξ_f estimated per-series as 1.2 * max(observed), ensuring positivity.
- Handles missing values (NaN) via masking.
- Outlier gating: if residual too large, we downweight the observation.
"""

# --- file: rlkf.py ---
from __future__ import annotations
import numpy as np


def _estimate_upper_bound(x: np.ndarray, eps: float = 1e-6) -> float:
    mx = np.nanmax(x) if np.any(np.isnan(x)) else np.max(x)
    return float(max(mx * 1.2 + eps, 1.0))


def _kalman_1d_reverse_lognormal(x: np.ndarray, process_var: float, obs_var: float, gate_std: float = 3.0) -> np.ndarray:
    T = x.shape[0]
    xi_f = _estimate_upper_bound(x)
    # Transform: z = ln(xi_f - x)
    z_obs = np.full(T, np.nan)
    mask_obs = ~np.isnan(x)
    safe = np.clip(x[mask_obs], a_min=None, a_max=xi_f - 1e-6)
    z_obs[mask_obs] = np.log(xi_f - safe)

    # Kalman init
    z_est = np.zeros(T)
    P = np.zeros(T)
    z_est[0] = z_obs[mask_obs][0] if mask_obs[0] else 0.0
    P[0] = 1.0

    for t in range(1, T):
        # Predict
        z_pred = z_est[t-1]
        P_pred = P[t-1] + process_var
        if mask_obs[t]:
            y_t = z_obs[t]
            # Innovation
            innov = y_t - z_pred
            S = P_pred + obs_var
            K = P_pred / S
            # Gating (downweight large residuals)
            if abs(innov) > gate_std * np.sqrt(S):
                # treat as missing
                z_est[t] = z_pred
                P[t] = P_pred
            else:
                z_est[t] = z_pred + K * innov
                P[t] = (1 - K) * P_pred
        else:
            z_est[t] = z_pred
            P[t] = P_pred

    # Inverse transform: x_hat = xi_f - exp(z_est)
    x_hat = xi_f - np.exp(z_est)
    # clamp to [0, xi_f)
    x_hat = np.clip(x_hat, 0.0, xi_f - 1e-6)
    return x_hat


def apply_rlkf(X: np.ndarray, process_var: float = 0.05, obs_var: float = 0.1) -> np.ndarray:
    Xc = np.array([_kalman_1d_reverse_lognormal(row, process_var, obs_var) for row in X])
    return Xc

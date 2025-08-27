# =====================
# sboa_optimizer.py
# =====================

"""Secretary Bird Optimization Algorithm (SBOA) to tune IGANN hyper-params.
We optimize (lambda_task, lambda_bg) by maximizing validation F1.
Note: lightweight implementation to keep runtime reasonable.
"""

# --- file: sboa_optimizer.py ---
from __future__ import annotations
import numpy as np
from typing import Tuple, Callable
from igann_model import train_igann
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def _clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def _levy_flight(dim: int, scale: float = 0.1):
    # Mantegna's algorithm approximation
    beta = 1.5
    sigma_u = (np.math.gamma(1+beta) * np.sin(np.pi*beta/2) / (np.math.gamma((1+beta)/2) * beta * 2**((beta-1)/2))) ** (1/beta)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1/beta))
    return scale * step


def sboa_optimize(
    X: np.ndarray,
    y: np.ndarray,
    eval_split: float = 0.2,
    pop_size: int = 20,
    iterations: int = 30,
    hidden: int = 16,
    lr: float = 1e-3,
    max_epochs: int = 150,
    patience: int = 10,
    seed: int = 42,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((1e-6, 1e-2), (1e-7, 1e-3)),
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=eval_split, stratify=y, random_state=seed)

    # initialize population in bounds
    lo1, hi1 = bounds[0]
    lo2, hi2 = bounds[1]
    pop = np.column_stack([
        rng.uniform(lo1, hi1, size=pop_size),
        rng.uniform(lo2, hi2, size=pop_size)
    ])
    fitness = np.full(pop_size, -np.inf)

    def evaluate(ind):
        lam_task, lam_bg = float(ind[0]), float(ind[1])
        model, _ = train_igann(X_tr, y_tr, X_val, y_val, X_val, y_val,
                               lr=lr, max_epochs=max_epochs, hidden=hidden,
                               lambda_task=lam_task, lambda_bg=lam_bg,
                               patience=patience, seed=seed)
        from igann_model import predict_proba
        proba = predict_proba(model, X_val)
        y_pred = (proba >= 0.5).astype(int)
        return f1_score(y_val, y_pred, zero_division=0)

    # Evaluate initial
    for i in range(pop_size):
        fitness[i] = evaluate(pop[i])

    best_idx = int(np.argmax(fitness))
    best = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    for it in range(iterations):
        for i in range(pop_size):
            # Hunting (Levy flight) on first dimension, Escape on second
            candidate = pop[i].copy()
            candidate += _levy_flight(dim=2, scale=0.05)
            candidate[0] = _clamp(candidate[0], lo1, hi1)
            candidate[1] = _clamp(candidate[1], lo2, hi2)
            fit_c = evaluate(candidate)
            if fit_c > fitness[i]:
                pop[i] = candidate
                fitness[i] = fit_c
                if fit_c > best_fit:
                    best_fit = fit_c
                    best = candidate.copy()
        # optional diversification
        # small Gaussian perturbation around best
        jitter = rng.normal(0, 0.01, size=2)
        best = _clamp(best + jitter * np.array([hi1-lo1, hi2-lo2]), [lo1, lo2], [hi1, hi2])

    return float(best[0]), float(best[1])


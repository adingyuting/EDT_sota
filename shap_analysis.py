# ===================
# shap_analysis.py
# ===================

"""SHAP interpretation for IGANN features (on small sample for speed)."""

# --- file: shap_analysis.py ---
from __future__ import annotations
import numpy as np
import shap
from igann_model import predict_proba


def shap_top_features(model, X: np.ndarray, sample_size: int = 256, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(X.shape[0]), size=min(sample_size, X.shape[0]), replace=False)
    Xs = X[idx]
    # KernelExplainer on probability function
    f = lambda data: predict_proba(model, data)
    explainer = shap.KernelExplainer(f, shap.sample(Xs, min(64, Xs.shape[0])))
    sv = explainer.shap_values(Xs, nsamples=100)
    return sv, idx

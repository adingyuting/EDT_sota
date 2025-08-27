
import numpy as np
from collections import defaultdict
from sklearn.neural_network import MLPRegressor
from typing import Dict, Tuple, List

class SLDIImputer:
    """Shallow Learning-based Data Imputation (SLDI) per missing-type.
    We treat each row (a user's time series) as features.
    For each unique missing pattern (tuple of missing indices), we train a MLPRegressor mapping
    from observed features -> missing features using complete rows as supervision.
    """
    def __init__(self, hidden_multiplier: float = 2.0, max_iter: int = 500, random_state: int = 42):
        self.models: Dict[Tuple[int, ...], MLPRegressor] = {}
        self.obs_idx: Dict[Tuple[int, ...], np.ndarray] = {}
        self.mis_idx: Dict[Tuple[int, ...], np.ndarray] = {}
        self.hidden_multiplier = hidden_multiplier
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        N, F = X.shape
        # complete rows
        complete_mask = ~np.isnan(X).any(axis=1)
        Xco = X[complete_mask]
        if Xco.shape[0] == 0:
            return self

        # enumerate missing patterns among incomplete rows
        patterns = defaultdict(int)
        for i in range(N):
            nan_idx = np.where(np.isnan(X[i]))[0]
            key = tuple(nan_idx.tolist())
            if len(key) > 0:
                patterns[key] += 1

        for key, _ in patterns.items():
            mis = np.array(key, dtype=int)
            obs = np.array([j for j in range(F) if j not in mis], dtype=int)
            if len(mis) == 0 or len(obs) == 0:
                continue
            X_in = Xco[:, obs]
            y_out = Xco[:, mis]
            hidden = max(8, int(self.hidden_multiplier * len(obs)))
            mlp = MLPRegressor(
                hidden_layer_sizes=(hidden,),
                activation="relu",
                solver="adam",
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            mlp.fit(X_in, y_out)
            self.models[key] = mlp
            self.obs_idx[key] = obs
            self.mis_idx[key] = mis

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32).copy()
        N, F = X.shape
        for i in range(N):
            nan_idx = np.where(np.isnan(X[i]))[0]
            key = tuple(nan_idx.tolist())
            if key in self.models:
                obs = self.obs_idx[key]
                mis = self.mis_idx[key]
                x_in = X[i, obs].reshape(1, -1)
                # If still has NaN in observed, skip this fancy imputation
                if np.isnan(x_in).any():
                    continue
                y_hat = self.models[key].predict(x_in).reshape(-1)
                X[i, mis] = y_hat.astype(np.float32)
        # Fallback: any remaining NaN => fill with per-feature mean
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

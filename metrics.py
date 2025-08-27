
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    accuracy_score,
)
from typing import Dict, Iterable


def precision_at_k(r: Iterable[int], k: int) -> float:
    """Compute precision@k."""
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return float(np.mean(r))


def average_precision(r: Iterable[int]) -> float:
    """Compute average precision for a single ranked list."""
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return float(np.mean(out))


def mean_average_precision(rs: Iterable[Iterable[int]]) -> float:
    """Mean average precision over multiple ranked lists."""
    return float(np.mean([average_precision(r) for r in rs]))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, top_k: int = 40) -> Dict[str, float]:
    """Compute binary classification metrics with dynamic threshold.

    Parameters
    ----------
    y_true: np.ndarray
        Ground truth binary labels.
    y_prob: np.ndarray
        Predicted probabilities for the positive class.
    top_k: int, optional
        Number of top predictions per class used for MAP@K calculation.
    """

    # 1) Threshold-independent metric: ROC AUC
    auc = roc_auc_score(y_true, y_prob)

    # 2) Dynamic threshold via F1 maximization
    p_arr, r_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    f1_arr = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-12)
    idx = int(np.argmax(f1_arr))
    best_thr = thr_arr[idx] if idx < len(thr_arr) else 0.5
    y_pred = (y_prob >= best_thr).astype(int)

    # 3) Confusion matrix statistics
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    eps = 1e-12

    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    prec = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    fpr = FP / (FP + TN + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    # 4) MAP@K over both classes
    temp = pd.DataFrame({
        "label_0": 1 - y_true,
        "label_1": y_true,
        "preds_0": 1 - y_prob,
        "preds_1": y_prob,
    })
    mapk = mean_average_precision([
        list(temp.sort_values(by="preds_0", ascending=False).label_0[:top_k]),
        list(temp.sort_values(by="preds_1", ascending=False).label_1[:top_k]),
    ])

    return {
        "Thr": float(best_thr),
        "ACC": float(acc),
        "PRE": float(prec),
        "RE": float(rec),
        "FPR": float(fpr),
        f"map{top_k}": float(mapk),
        "AUC": float(auc),
        "F1": float(f1),
    }

def multiclass_metrics(y_true, y_pred) -> Dict[str, float]:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    return out

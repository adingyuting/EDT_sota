
import numpy as np
from sklearn.metrics import (
    average_precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from typing import Dict

def binary_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    out = {
        "APS": float(average_precision_score(y_true, y_prob)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }
    return out

def multiclass_metrics(y_true, y_pred) -> Dict[str, float]:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    return out

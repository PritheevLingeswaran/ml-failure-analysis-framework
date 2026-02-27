from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def brier_score(y_true: np.ndarray, y_prob_pos: np.ndarray) -> float:
    # Mean squared error between predicted probability and true label
    return float(np.mean((y_prob_pos - y_true) ** 2))

def expected_calibration_error(y_true: np.ndarray, y_prob_pos: np.ndarray, n_bins: int = 10) -> float:
    # Simple ECE: bin by confidence, compare avg confidence vs empirical accuracy
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob_pos >= lo) & (y_prob_pos < hi) if i < n_bins-1 else (y_prob_pos >= lo) & (y_prob_pos <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask] == (y_prob_pos[mask] >= 0.5))
        conf = np.mean(y_prob_pos[mask])
        ece += np.abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)

def compute_binary_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float, n_bins_ece: int) -> Dict[str, Any]:
    y_pred = (y_prob_pos >= threshold).astype(int)
    out: Dict[str, Any] = {}
    out["threshold"] = float(threshold)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    # AUC metrics are threshold-free; still useful but NOT sufficient for decisions.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob_pos))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["brier"] = float(brier_score(y_true, y_prob_pos))
    out["ece"] = float(expected_calibration_error(y_true, y_prob_pos, n_bins=n_bins_ece))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    # Confidence-aware: how many predictions are high confidence, and how accurate they are
    out["avg_confidence"] = float(np.mean(np.maximum(y_prob_pos, 1 - y_prob_pos)))
    return out

def compute_multiclass_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    y_pred = np.argmax(y_proba, axis=1)
    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return out

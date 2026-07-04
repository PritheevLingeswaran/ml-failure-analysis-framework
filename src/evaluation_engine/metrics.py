from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def brier_score(y_true: np.ndarray, y_prob_pos: np.ndarray) -> float:
    # Mean squared error between predicted probability and true label
    return float(np.mean((y_prob_pos - y_true) ** 2))

def expected_calibration_error(y_true: np.ndarray, y_prob_pos: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (Guo et al., 2017).

    Bins predictions by the *confidence of the predicted class* — max(p, 1-p) —
    and, within each bin, measures the gap between that average confidence and the
    empirical accuracy. The weighted sum of gaps is the ECE, in [0, ~0.5].

    A common bug (which this replaces) is to bin by the raw positive-class
    probability and compare its mean against 0.5-threshold accuracy. That inflates
    the error massively on confidently-negative bins (mean p ≈ 0 vs accuracy ≈ 1),
    producing implausible ECE values of 0.6-0.9 for well-calibrated models.
    """
    y_true = np.asarray(y_true)
    y_prob_pos = np.asarray(y_prob_pos, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")

    y_pred = (y_prob_pos >= 0.5).astype(int)
    confidence = np.maximum(y_prob_pos, 1.0 - y_prob_pos)  # in [0.5, 1.0]
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi) if i < n_bins - 1 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(confidence[mask]))
        ece += abs(acc - conf) * (int(np.sum(mask)) / n)
    return float(ece)


def _binary_metric_validity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Describe when positive-class slice metrics are semantically valid.

    Why:
    - A zero-valued metric can mean "bad performance" or "undefined for this slice".
    - Diagnostics must distinguish those cases so single-class slices do not create false alarms.
    """
    positives = int(np.sum(y_true == 1))
    predicted_positives = int(np.sum(y_pred == 1))
    unique_classes = np.unique(y_true)

    reasons: List[str] = []
    if len(unique_classes) <= 1:
        reasons.append("single_class_slice")
    if positives == 0:
        reasons.append("no_positive_labels")
    if predicted_positives == 0:
        reasons.append("no_predicted_positives")

    single_class_slice = len(unique_classes) <= 1
    recall_valid = (positives > 0) and not single_class_slice
    precision_valid = (predicted_positives > 0) and not single_class_slice
    f1_valid = recall_valid and precision_valid
    return {
        "precision_valid": bool(precision_valid),
        "recall_valid": bool(recall_valid),
        "f1_valid": bool(f1_valid),
        "reasons": reasons,
    }

def compute_binary_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float, n_bins_ece: int) -> Dict[str, Any]:
    if len(y_true) == 0 or len(y_prob_pos) == 0:
        return {
            "threshold": float(threshold),
            "accuracy": float("nan"),
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "avg_confidence": float("nan"),
            "metrics_validity": {
                "precision_valid": False,
                "recall_valid": False,
                "f1_valid": False,
                "reasons": ["empty_slice"],
            },
        }

    y_pred = (y_prob_pos >= threshold).astype(int)
    validity = _binary_metric_validity(y_true, y_pred)
    out: Dict[str, Any] = {}
    out["threshold"] = float(threshold)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = (
        float(precision_score(y_true, y_pred, zero_division=0))
        if validity["precision_valid"]
        else None
    )
    out["recall"] = (
        float(recall_score(y_true, y_pred, zero_division=0))
        if validity["recall_valid"]
        else None
    )
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0)) if validity["f1_valid"] else None
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
    out["metrics_validity"] = validity
    return out

def compute_multiclass_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    y_pred = np.argmax(y_proba, axis=1)
    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return out

from __future__ import annotations
from typing import Any, Dict
from src.utils.config import load_yaml

def load_costs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return load_yaml(cfg["decision"]["costs_config"])

def expected_cost_binary(y_true, y_score, threshold: float, costs: Dict[str, float]) -> float:
    # costs: TP, TN, FP, FN (TP can be negative = benefit)
    import numpy as np
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    total = tp*costs["TP"] + tn*costs["TN"] + fp*costs["FP"] + fn*costs["FN"]
    return float(total / len(y_true))

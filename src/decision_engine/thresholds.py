from __future__ import annotations
from typing import Any, Dict
import numpy as np
from src.decision_engine.costs import expected_cost_binary
from src.evaluation_engine.metrics import compute_binary_metrics

def optimize_threshold(y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray, costs: Dict[str, float]) -> Dict[str, Any]:
    best_t = None
    best_cost = float("inf")
    best_metrics = None
    for t in grid:
        c = expected_cost_binary(y_true, y_score, float(t), costs)
        if c < best_cost:
            best_cost = c
            best_t = float(t)
    # Provide metrics at chosen threshold for interpretability
    best_metrics = compute_binary_metrics(y_true, y_score, threshold=best_t, n_bins_ece=10)
    return {"threshold": best_t, "expected_cost": float(best_cost), "metrics": best_metrics}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.decision_engine.costs import expected_cost_binary


@dataclass
class CalibrationResult:
    method: str
    y_score_calibrated: np.ndarray


def calibrate_scores(y_true: np.ndarray, y_score: np.ndarray, method: str) -> CalibrationResult:
    method = (method or "none").lower()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if method == "none":
        return CalibrationResult(method="none", y_score_calibrated=np.clip(y_score, 0.0, 1.0))

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        cal = iso.fit_transform(y_score, y_true)
        return CalibrationResult(method="isotonic", y_score_calibrated=np.clip(cal, 0.0, 1.0))

    if method == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=200)
        lr.fit(y_score.reshape(-1, 1), y_true)
        cal = lr.predict_proba(y_score.reshape(-1, 1))[:, 1]
        return CalibrationResult(method="platt", y_score_calibrated=np.clip(cal, 0.0, 1.0))

    raise ValueError(f"Unsupported calibration method='{method}'")


def threshold_ci_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: np.ndarray,
    costs: Dict[str, float],
    iters: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return {"mean_threshold": float("nan"), "ci": [float("nan"), float("nan")], "samples": []}

    rng = np.random.default_rng(seed)
    chosen: List[float] = []
    grid = np.asarray(grid).astype(float)
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        ss = y_score[idx]
        vals = np.array([expected_cost_binary(yy, ss, float(t), costs) for t in grid])
        chosen.append(float(grid[int(np.argmin(vals))]))

    arr = np.array(chosen, dtype=float)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return {
        "mean_threshold": float(np.mean(arr)),
        "ci": [lo, hi],
        "samples": [float(x) for x in arr.tolist()],
    }


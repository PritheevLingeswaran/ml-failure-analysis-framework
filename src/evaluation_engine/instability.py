from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any, Tuple

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    threshold: float,
    metric_fn: Callable[[np.ndarray, np.ndarray, float], float],
    iters: int,
    alpha: float,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_prob_pos[idx], threshold))
    lo = float(np.quantile(vals, alpha/2))
    hi = float(np.quantile(vals, 1 - alpha/2))
    return lo, hi

def instability_flag(sample_count: int, min_count: int, ci_width: float, max_width: float = 0.1) -> Dict[str, Any]:
    # Heuristic: low N or wide CI => unstable.
    return {
        "unstable": bool(sample_count < min_count or ci_width > max_width),
        "reason": (
            "low_sample_count" if sample_count < min_count
            else "wide_confidence_interval" if ci_width > max_width
            else "stable"
        ),
        "sample_count": int(sample_count),
        "ci_width": float(ci_width),
    }

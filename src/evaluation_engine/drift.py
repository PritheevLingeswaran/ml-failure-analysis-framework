from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _psi_numeric(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    t = pd.to_numeric(train, errors="coerce").dropna()
    s = pd.to_numeric(test, errors="coerce").dropna()
    if len(t) == 0 or len(s) == 0:
        return float("nan")

    edges = np.quantile(t, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    t_hist, _ = np.histogram(t, bins=edges)
    s_hist, _ = np.histogram(s, bins=edges)
    t_pct = np.clip(t_hist / max(1, len(t)), 1e-6, 1.0)
    s_pct = np.clip(s_hist / max(1, len(s)), 1e-6, 1.0)
    return float(np.sum((s_pct - t_pct) * np.log(s_pct / t_pct)))


def _tv_categorical(train: pd.Series, test: pd.Series) -> float:
    t = train.fillna("__nan__").astype(str)
    s = test.fillna("__nan__").astype(str)
    cats = sorted(set(t.unique()).union(set(s.unique())))
    if not cats:
        return 0.0
    t_dist = t.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    s_dist = s.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return float(0.5 * np.abs(t_dist - s_dist).sum())


def compute_drift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: List[str],
    psi_warn: float = 0.2,
    tv_warn: float = 0.2,
) -> Dict[str, Any]:
    shared = [c for c in train_df.columns if c in test_df.columns and c not in set(exclude_cols)]
    per_feature: List[Dict[str, Any]] = []
    for c in shared:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            score = _psi_numeric(train_df[c], test_df[c], bins=10)
            per_feature.append(
                {"feature": c, "type": "numeric", "metric": "psi", "score": score, "drifted": bool(score >= psi_warn)}
            )
        else:
            score = _tv_categorical(train_df[c], test_df[c])
            per_feature.append(
                {"feature": c, "type": "categorical", "metric": "tv_distance", "score": score, "drifted": bool(score >= tv_warn)}
            )

    per_feature = sorted(
        per_feature,
        key=lambda r: -1.0 * (r["score"] if isinstance(r.get("score"), (int, float)) and not np.isnan(r["score"]) else -1.0),
    )
    return {
        "num_features_checked": len(per_feature),
        "num_drifted": int(sum(1 for r in per_feature if r["drifted"])),
        "top_drifted": [r for r in per_feature if r["drifted"]][:10],
        "all_features": per_feature,
    }


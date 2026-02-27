from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.evaluation_engine.metrics import compute_binary_metrics
from src.decision_engine.costs import load_costs, expected_cost_binary
from src.decision_engine.thresholds import optimize_threshold

logger = logging.getLogger(__name__)

def _safe_query(df: pd.DataFrame, query: str) -> pd.Series:
    """Evaluate slice query using pandas query.

    Safety rationale:
    - Using pandas query is relatively constrained vs eval/exec.
    - We still treat slice definitions as config-controlled and code-reviewed.
    """
    return df.eval(query) if ("@" in query) else df.query(query).index  # not used

def _apply_query(df: pd.DataFrame, query: str) -> pd.Series:
    # Use pandas query to get indices; fall back to eval for some expressions.
    try:
        idx = df.query(query).index
        mask = df.index.isin(idx)
        return pd.Series(mask, index=df.index)
    except Exception:
        # fallback to eval-based boolean series
        try:
            mask = df.eval(query)
            if not isinstance(mask, pd.Series):
                raise ValueError("Slice query did not produce a boolean Series")
            return mask.astype(bool)
        except Exception as e:
            raise ValueError(f"Failed to evaluate slice query='{query}': {e}")

def build_slices(cfg: Dict[str, Any], df_features: pd.DataFrame, df_pred: pd.DataFrame, model_name: str, split: str) -> List[Dict[str, Any]]:
    label_col = cfg["data"]["dataset"]["label_col"]
    text_col = cfg["data"]["dataset"].get("text_col")
    costs = load_costs(cfg)
    use_case = cfg["decision"]["default_use_case"]
    use_case_cfg = costs["use_cases"][use_case]["binary"]

    # Join feature columns needed for slicing with predictions
    df = df_features.copy()
    # Ensure label present from predictions (authoritative)
    df[label_col] = df_pred[label_col].values
    df["y_score"] = df_pred["y_score"].values

    slices_cfg_path = cfg["slicing"]["slices_config"]
    slices_cfg = load_yaml(slices_cfg_path)
    slices = []

    # Rule-based slices
    for s in (slices_cfg.get("slices", []) or []):
        name = s["name"]
        query = s["query"]
        desc = s.get("description", "")
        # skip if referenced columns missing (common in multi-domain reuse)
        missing_cols = _missing_columns_for_query(query, df.columns)
        if missing_cols:
            logger.info("Skipping slice '%s' (missing columns: %s)", name, missing_cols)
            continue

        mask = _apply_query(df, query)
        slices.append(_slice_metrics(cfg, df, mask, slice_name=name, description=desc, model_name=model_name, split=split, use_case_cfg=use_case_cfg))

    # Auto slices
    if cfg["slicing"]["auto_slices"].get("enabled", True):
        slices.extend(_auto_slices(cfg, df, model_name=model_name, split=split, use_case_cfg=use_case_cfg))

    # Remove empty slices
    slices = [s for s in slices if s["count"] > 0]
    return slices

def _missing_columns_for_query(query: str, cols) -> List[str]:
    # Heuristic: extract tokens that look like columns (very conservative).
    # If user wants strict parsing, replace with a real expression parser.
    import re
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
    keywords = set(["and", "or", "not", "in", "True", "False", "None"])
    cols_set = set(cols)
    # Filter obvious python/pandas tokens
    candidates = [t for t in tokens if t not in keywords]
    # Keep those not in cols and not known builtins
    builtins = set(["abs", "min", "max", "len"])
    missing = sorted({t for t in candidates if (t not in cols_set and t not in builtins)})
    # This can false-positive on string constants. We tolerate and skip only if clearly missing.
    # So: only treat as missing if token appears unquoted in query.
    missing_strict = []
    for t in missing:
        if f"'{t}'" in query or f'"{t}"' in query:
            continue
        missing_strict.append(t)
    return missing_strict

def _slice_metrics(cfg: Dict[str, Any], df: pd.DataFrame, mask: pd.Series, slice_name: str, description: str, model_name: str, split: str, use_case_cfg: Dict[str, float]) -> Dict[str, Any]:
    n_bins = int(cfg["evaluation"]["calibration"]["n_bins"])
    threshold = 0.5  # slice metrics will include cost-opt threshold

    sdf = df.loc[mask].copy()
    y_true = sdf[cfg["data"]["dataset"]["label_col"]].to_numpy()
    y_score = sdf["y_score"].to_numpy()

    base_metrics = compute_binary_metrics(y_true, y_score, threshold=0.5, n_bins_ece=n_bins)

    # Decision metrics per slice (opt threshold by cost)
    grid = _threshold_grid(cfg)
    best = optimize_threshold(y_true, y_score, grid, use_case_cfg)

    # Instability detection (simple: count-based; CI-based is costly for per-slice)
    inst = {
        "unstable": bool(len(sdf) < int(cfg["evaluation"]["instability"]["min_count"])),
        "reason": "low_sample_count" if len(sdf) < int(cfg["evaluation"]["instability"]["min_count"]) else "stable",
        "sample_count": int(len(sdf)),
    }

    return {
        "slice_name": slice_name,
        "description": description,
        "model_name": model_name,
        "split": split,
        "count": int(len(sdf)),
        "metrics": base_metrics,
        "decision": {
            "best_threshold": best["threshold"],
            "expected_cost": best["expected_cost"],
            "costs": use_case_cfg,
        },
        "instability": inst,
    }

def _threshold_grid(cfg: Dict[str, Any]) -> np.ndarray:
    g = cfg["evaluation"]["threshold_grid"]
    start, stop, step = float(g["start"]), float(g["stop"]), float(g["step"])
    return np.round(np.arange(start, stop + 1e-12, step), 6)

def _auto_slices(cfg: Dict[str, Any], df: pd.DataFrame, model_name: str, split: str, use_case_cfg: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    label_col = cfg["data"]["dataset"]["label_col"]
    text_col = cfg["data"]["dataset"].get("text_col")
    auto = cfg["slicing"]["auto_slices"]

    # 1) short vs long input (if text exists)
    if text_col and text_col in df.columns:
        bins = auto["text_length_bins"]
        lengths = df[text_col].fillna("").astype(str).str.len().to_numpy()
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i+1]
            mask = (lengths >= lo) & (lengths < hi)
            out.append(_slice_metrics(cfg, df, pd.Series(mask, index=df.index), f"text_len[{lo},{hi})", "Auto: text length bin", model_name, split, use_case_cfg))

    # 2) high vs low confidence (based on score distance from 0.5)
    conf = np.maximum(df["y_score"].to_numpy(), 1 - df["y_score"].to_numpy())
    cbins = auto["confidence_bins"]
    for i in range(len(cbins) - 1):
        lo, hi = cbins[i], cbins[i+1]
        mask = (conf >= lo) & (conf < hi)
        out.append(_slice_metrics(cfg, df, pd.Series(mask, index=df.index), f"confidence[{lo},{hi})", "Auto: confidence bin", model_name, split, use_case_cfg))

    # 3) rare vs frequent labels (relative frequency)
    counts = df[label_col].value_counts(normalize=True)
    freq = df[label_col].map(counts).to_numpy()
    fbins = auto["label_freq_bins"]
    for i in range(len(fbins) - 1):
        lo, hi = fbins[i], fbins[i+1]
        mask = (freq >= lo) & (freq < hi)
        out.append(_slice_metrics(cfg, df, pd.Series(mask, index=df.index), f"label_freq[{lo},{hi})", "Auto: label frequency bin", model_name, split, use_case_cfg))

    # 4) easy vs hard examples (ensemble disagreement)
    if auto.get("difficulty", {}).get("enabled", True):
        # This requires precomputed column 'difficulty_hard'. If absent, compute naive proxy:
        # hard if score near 0.5 (uncertain)
        if "difficulty_hard" in df.columns:
            hard_mask = df["difficulty_hard"].astype(bool).to_numpy()
        else:
            hard_mask = (np.abs(df["y_score"].to_numpy() - 0.5) < 0.15)
        out.append(_slice_metrics(cfg, df, pd.Series(~hard_mask, index=df.index), "easy_examples", "Auto: easy examples", model_name, split, use_case_cfg))
        out.append(_slice_metrics(cfg, df, pd.Series(hard_mask, index=df.index), "hard_examples", "Auto: hard examples", model_name, split, use_case_cfg))

    return out

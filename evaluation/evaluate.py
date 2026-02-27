from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from src.datasets.csv_classification import CSVClassificationDataset
from src.datasets.splits import make_splits
from src.models.registry import build_models
from src.evaluation_engine.predictions import save_predictions, load_predictions, build_run_id
from src.evaluation_engine.evaluator import Evaluator
from src.utils.paths import ensure_dir
from src.utils.io import write_json

logger = logging.getLogger(__name__)

def _feature_columns(df: pd.DataFrame, label_col: str, id_col: str, text_col: str | None) -> Tuple[pd.DataFrame, pd.Series]:
    # Use all non-label columns except text/id. Text is kept separately for slicing/error surfacing.
    drop = [label_col]
    if id_col in df.columns:
        drop.append(id_col)
    if text_col and text_col in df.columns:
        drop.append(text_col)
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df[label_col].astype(int)
    return X, y

def run_evaluate(cfg: Dict[str, Any]) -> None:
    """Main evaluation runner that expects trained models + predictions."""
    result = run_evaluate_in_memory(cfg, split="test", use_case=cfg["decision"]["default_use_case"])

    outputs = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(outputs / "reports")
    run_id = build_run_id(cfg)

    write_json(outputs / "reports" / f"api_payload__{run_id}.json", result)
    logger.info("Evaluation complete. Outputs written to %s", outputs)

def run_evaluate_in_memory(cfg: Dict[str, Any], split: str, use_case: str) -> Dict[str, Any]:
    dataset_cfg = cfg["data"]["dataset"]
    loader = CSVClassificationDataset(
        path=cfg["paths"]["data_raw"],
        label_col=dataset_cfg["label_col"],
        id_col=dataset_cfg["id_col"],
        text_col=dataset_cfg.get("text_col"),
    )
    df = loader.load()
    bundle = make_splits(
        df=df,
        label_col=dataset_cfg["label_col"],
        id_col=dataset_cfg["id_col"],
        text_col=dataset_cfg.get("text_col"),
        seed=int(cfg["data"]["split"]["seed"]),
        test_size=float(cfg["data"]["split"]["test_size"]),
        val_size=float(cfg["data"]["split"]["val_size"]),
    )

    splits = {"train": bundle.train, "val": bundle.val, "test": bundle.test}
    df_split = splits[split].reset_index(drop=True)

    X, y = _feature_columns(df_split, dataset_cfg["label_col"], dataset_cfg["id_col"], dataset_cfg.get("text_col"))
    # Features for slicing/error surfacing include id + optional text + any metadata columns
    feat_cols = [dataset_cfg["id_col"]]
    if dataset_cfg.get("text_col"):
        feat_cols.append(dataset_cfg["text_col"])
    # Keep any "metadata" columns (non-numeric are fine for slicing rules)
    meta_cols = [c for c in df_split.columns if c not in set(X.columns.tolist() + [dataset_cfg["label_col"]])]
    # Avoid duplicates
    feat_cols = list(dict.fromkeys(feat_cols + meta_cols))
    features = df_split[feat_cols].copy()

    # Load predictions (must exist)
    models = build_models(cfg)
    predictions_by_model = {}
    for m in models:
        df_pred = load_predictions(cfg, m.name, split=split)
        predictions_by_model[m.name] = df_pred

    # Add a simple difficulty proxy column if we have >=2 models:
    # hard if models disagree on class at 0.5 threshold
    if len(predictions_by_model) >= 2:
        preds = []
        for mname, dfp in predictions_by_model.items():
            preds.append((dfp["y_score"].to_numpy() >= 0.5).astype(int))
        disagree = np.std(np.vstack(preds), axis=0) > 0.0
        features["difficulty_hard"] = disagree

    evaluator = Evaluator(cfg)

    # For visualization cost curve, compute per-model cost curve here
    from src.decision_engine.costs import load_costs, expected_cost_binary
    from src.decision_engine.thresholds import optimize_threshold
    grid = _threshold_grid(cfg)
    costs = load_costs(cfg)["use_cases"][use_case]["binary"]
    per_model_payload = {}
    for mname, dfp in predictions_by_model.items():
        y_true = dfp[dataset_cfg["label_col"]].to_numpy()
        y_score = dfp["y_score"].to_numpy()
        expected_costs = [expected_cost_binary(y_true, y_score, float(t), costs) for t in grid]
        payload = evaluator.evaluate_predictions(mname, dfp, features, split=split, use_case=use_case)
        payload["cost_curve"] = {"thresholds": grid.tolist(), "expected_costs": expected_costs}
        per_model_payload[mname] = payload

    result = evaluator.run_full_evaluation(
        predictions_by_model=predictions_by_model,
        features_by_split={split: features},
        split=split,
        use_case=use_case,
    )
    return {
        "comparison": result.overall["comparison"],
        "per_model": per_model_payload,
        "slice_table": result.slices,
        "decision": result.decision,
        "errors": result.errors,
        "run_id": result.run_id,
        "split": split,
    }

def _threshold_grid(cfg: Dict[str, Any]) -> np.ndarray:
    g = cfg["evaluation"]["threshold_grid"]
    start, stop, step = float(g["start"]), float(g["stop"]), float(g["step"])
    return np.round(np.arange(start, stop + 1e-12, step), 6)

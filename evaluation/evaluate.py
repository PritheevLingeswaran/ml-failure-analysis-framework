from __future__ import annotations
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from src.datasets.csv_classification import CSVClassificationDataset
from src.datasets.splits import make_splits
from src.models.registry import build_models
from src.evaluation_engine.predictions import save_predictions, load_predictions, build_run_id
from src.evaluation_engine.evaluator import Evaluator
from src.evaluation_engine.eval_quality import compute_eval_quality, render_eval_quality_markdown
from src.evaluation_engine.drift import compute_drift_report
from src.decision_engine.sensitivity import build_cost_scenarios, run_cost_sensitivity
from src.utils.paths import ensure_dir
from src.utils.io import write_json
from src.utils.tracking import write_experiment_record
from src.utils.timing import TimingCollector

logger = logging.getLogger(__name__)

def _feature_columns(df: pd.DataFrame, label_col: str, id_col: str, text_col: str | None) -> Tuple[pd.DataFrame, pd.Series]:
    # Use all non-label columns except text/id. Text is kept separately for slicing/error surfacing.
    drop = [label_col]
    if id_col in df.columns:
        drop.append(id_col)
    if text_col and text_col in df.columns:
        drop.append(text_col)
    X = df.drop(columns=[c for c in drop if c in df.columns])
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        X = X.drop(columns=dt_cols, errors="ignore")
    y = df[label_col].astype(int)
    return X, y

def run_evaluate(cfg: Dict[str, Any]) -> None:
    """Main evaluation runner that expects trained models + predictions."""
    result = run_evaluate_in_memory(cfg, split="test", use_case=cfg["decision"]["default_use_case"])

    outputs = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(outputs / "reports")
    run_id = build_run_id(cfg)

    write_json(outputs / "reports" / f"api_payload__{run_id}.json", result)
    write_json(
        outputs / "reports" / f"deployment_policy__{run_id}.json",
        {
            "run_id": run_id,
            "use_case": cfg["decision"]["default_use_case"],
            "recommended_model": result["decision"].get("recommended_model"),
            "recommended_threshold": result["decision"].get("recommended_threshold"),
            "notes": "Deploy this tuple together with the exact cost matrix and slice policy.",
        },
    )
    write_experiment_record(
        outputs,
        run_id,
        {
            "run_id": run_id,
            "split": result["split"],
            "use_case": cfg["decision"]["default_use_case"],
            "winner": result["comparison"].get("winner"),
            "recommended_model": result["decision"].get("recommended_model"),
            "recommended_threshold": result["decision"].get("recommended_threshold"),
        },
    )
    logger.info("Evaluation complete. Outputs written to %s", outputs)

def run_evaluate_in_memory(cfg: Dict[str, Any], split: str, use_case: str) -> Dict[str, Any]:
    run_start = perf_counter()
    timings = TimingCollector()
    dataset_cfg = cfg["data"]["dataset"]
    loader = CSVClassificationDataset(
        path=cfg["paths"]["data_raw"],
        label_col=dataset_cfg["label_col"],
        id_col=dataset_cfg["id_col"],
        text_col=dataset_cfg.get("text_col"),
        time_col=dataset_cfg.get("time_col"),
    )
    with timings.stage("load_data"):
        df = loader.load()
    with timings.stage("make_splits"):
        bundle = make_splits(
            df=df,
            label_col=dataset_cfg["label_col"],
            id_col=dataset_cfg["id_col"],
            text_col=dataset_cfg.get("text_col"),
            seed=int(cfg["data"]["split"]["seed"]),
            test_size=float(cfg["data"]["split"]["test_size"]),
            val_size=float(cfg["data"]["split"]["val_size"]),
            strategy=str(cfg["data"]["split"].get("strategy", "random")),
            time_col=cfg["data"]["split"].get("time_col"),
        )

    splits = {"train": bundle.train, "val": bundle.val, "test": bundle.test}
    df_split = splits[split].reset_index(drop=True)

    # Slicing / error-surfacing frame: keep every raw column except the label
    # (id, text, and business metadata such as region / amount / event_time) so
    # rule-based slices and fairness analysis can reference them directly. The
    # model's numeric feature matrix is separate and already baked into the saved
    # predictions, so it does not need to be reconstructed here.
    features = df_split.drop(columns=[dataset_cfg["label_col"]]).copy()

    # Load predictions (must exist)
    with timings.stage("load_predictions"):
        models = build_models(cfg)
        predictions_by_model = {}
        for m in models:
            df_pred = load_predictions(cfg, m.name, split=split)
            predictions_by_model[m.name] = df_pred

    # Optional lightweight model stack: mean-probability ensemble over available models.
    ens_cfg = cfg.get("advanced", {}).get("model_stack", {}).get("ensemble_avg", {})
    if ens_cfg.get("enabled", True) and len(predictions_by_model) >= 2:
        first = next(iter(predictions_by_model.values())).copy()
        stacked = np.vstack([dfp["y_score"].to_numpy() for dfp in predictions_by_model.values()])
        first["y_score"] = np.mean(stacked, axis=0)
        predictions_by_model["ensemble_avg"] = first

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
    model_scores = {}
    for mname, dfp in predictions_by_model.items():
        y_true = dfp[dataset_cfg["label_col"]].to_numpy()
        y_score = dfp["y_score"].to_numpy()
        model_scores[mname] = {"y_true": y_true, "y_score": y_score}
        expected_costs = [expected_cost_binary(y_true, y_score, float(t), costs) for t in grid]
        payload = evaluator.evaluate_predictions(
            mname,
            dfp,
            features,
            split=split,
            use_case=use_case,
            timing_collector=timings,
        )
        payload["cost_curve"] = {"thresholds": grid.tolist(), "expected_costs": expected_costs}
        per_model_payload[mname] = payload

    result = evaluator.run_full_evaluation(
        predictions_by_model=predictions_by_model,
        features_by_split={split: features},
        split=split,
        use_case=use_case,
        per_model=per_model_payload,
        timing_collector=timings,
    )

    # Drift report: compare train vs selected split features.
    drift = compute_drift_report(
        train_df=bundle.train,
        test_df=df_split,
        exclude_cols=[
            dataset_cfg["label_col"],
            dataset_cfg["id_col"],
            dataset_cfg.get("text_col", ""),
            dataset_cfg.get("time_col", ""),
        ],
        psi_warn=float(cfg.get("advanced", {}).get("drift", {}).get("psi_warn", 0.2)),
        tv_warn=float(cfg.get("advanced", {}).get("drift", {}).get("tv_warn", 0.2)),
    )

    # Cost sensitivity: winner robustness across FP/FN multipliers.
    sens_cfg = cfg.get("advanced", {}).get("cost_sensitivity", {})
    fp_mults = sens_cfg.get("fp_multipliers", [0.5, 1.0, 1.5, 2.0])
    fn_mults = sens_cfg.get("fn_multipliers", [0.5, 1.0, 1.5, 2.0])
    scenarios = build_cost_scenarios(costs, fp_mults=fp_mults, fn_mults=fn_mults)
    model_arrays = {
        mname: {
            "y_true": dfp[dataset_cfg["label_col"]].to_numpy(),
            "y_score": dfp["y_score"].to_numpy(),
        }
        for mname, dfp in predictions_by_model.items()
    }
    sensitivity = run_cost_sensitivity(model_arrays, grid=grid, scenarios=scenarios)

    eval_quality = compute_eval_quality(
        cfg,
        run_id=result.run_id,
        split=split,
        use_case=use_case,
        split_count=len(df_split),
        total_count=len(df),
        per_model=per_model_payload,
        model_scores=model_scores,
        decision=result.decision,
        runtime={"total_sec": 0.0, "stages": timings.stage_totals()},
    )

    outputs = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(outputs / "metrics")
    ensure_dir(outputs / "reports")
    with timings.stage("write_outputs"):
        write_json(outputs / "metrics" / f"eval_quality__{result.run_id}__{split}.json", eval_quality)
        (outputs / "reports" / f"eval_quality__{result.run_id}__{split}.md").write_text(
            render_eval_quality_markdown(eval_quality),
            encoding="utf-8",
        )

    total_runtime_sec = round(perf_counter() - run_start, 6)
    runtime = {
        "total_sec": total_runtime_sec,
        "stages": timings.stage_totals(),
    }
    eval_quality["runtime"] = runtime
    timings_payload = {
        "run_id": result.run_id,
        "split": split,
        "total_runtime_sec": total_runtime_sec,
        "stage_totals": runtime["stages"],
        "events": timings.events_payload(),
    }
    write_json(outputs / "metrics" / f"eval_quality__{result.run_id}__{split}.json", eval_quality)
    (outputs / "reports" / f"eval_quality__{result.run_id}__{split}.md").write_text(
        render_eval_quality_markdown(eval_quality),
        encoding="utf-8",
    )
    write_json(outputs / "metrics" / f"timings__{result.run_id}__{split}.json", timings_payload)
    logger.info(
        "total_runtime_sec=%.6f stage_breakdown=%s",
        total_runtime_sec,
        runtime["stages"],
    )
    logger.info(
        "eval_quality run_id=%s split=%s dataset_size=%s models_compared=%s slice_diagnostics=%s reduction_pct=%.6f",
        result.run_id,
        split,
        eval_quality["dataset_size"],
        eval_quality["models_compared"],
        eval_quality["slice_diagnostics"]["diagnostics_found"],
        eval_quality["business_loss_reduction"]["reduction_pct"],
    )

    return {
        "comparison": result.overall["comparison"],
        "per_model": per_model_payload,
        "slice_table": result.slices,
        "decision": result.decision,
        "errors": result.errors,
        "drift": drift,
        "cost_sensitivity": sensitivity,
        "eval_quality": eval_quality,
        "run_id": result.run_id,
        "split": split,
    }

def _threshold_grid(cfg: Dict[str, Any]) -> np.ndarray:
    g = cfg["evaluation"]["threshold_grid"]
    start, stop, step = float(g["start"]), float(g["stop"]), float(g["step"])
    return np.round(np.arange(start, stop + 1e-12, step), 6)

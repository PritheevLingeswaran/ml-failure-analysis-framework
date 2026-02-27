from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation_engine.metrics import compute_binary_metrics, compute_multiclass_metrics
from src.evaluation_engine.instability import bootstrap_ci, instability_flag
from src.slicing.builder import build_slices
from src.decision_engine.costs import load_costs, expected_cost_binary
from src.decision_engine.thresholds import optimize_threshold
from src.error_analysis.analyzer import analyze_errors
from src.visualization.plots import plot_all

from src.utils.io import write_json
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    run_id: str
    overall: Dict[str, Any]
    slices: List[Dict[str, Any]]
    decision: Dict[str, Any]
    errors: Dict[str, Any]

class Evaluator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def evaluate_predictions(
        self,
        model_name: str,
        df_pred: pd.DataFrame,
        df_features: pd.DataFrame,
        split: str,
        use_case: str,
    ) -> Dict[str, Any]:
        label_col = self.cfg["data"]["dataset"]["label_col"]
        y_true = df_pred[label_col].to_numpy()
        y_score = df_pred["y_score"].to_numpy()

        n_bins = int(self.cfg["evaluation"]["calibration"]["n_bins"])
        threshold = float(df_pred.get("threshold", pd.Series([0.5])).iloc[0]) if "threshold" in df_pred.columns else 0.5

        # Threshold-free metrics exist, but decisions are thresholded.
        overall = compute_binary_metrics(y_true, y_score, threshold=threshold, n_bins_ece=n_bins)
        overall["model_name"] = model_name
        overall["split"] = split
        overall["count"] = int(len(df_pred))

        # Slice metrics
        slices = build_slices(self.cfg, df_features, df_pred, model_name=model_name, split=split)

        # Decision metrics / threshold optimization
        costs = load_costs(self.cfg)
        use_case_cfg = costs["use_cases"][use_case]["binary"]

        grid = self._threshold_grid()
        best = optimize_threshold(y_true, y_score, grid, use_case_cfg)
        overall["best_threshold"] = best["threshold"]
        overall["expected_cost_at_best_threshold"] = best["expected_cost"]

        return {"overall": overall, "slices": slices, "best": best}

    def _threshold_grid(self) -> np.ndarray:
        g = self.cfg["evaluation"]["threshold_grid"]
        start, stop, step = float(g["start"]), float(g["stop"]), float(g["step"])
        return np.round(np.arange(start, stop + 1e-12, step), 6)

    def run_full_evaluation(
        self,
        predictions_by_model: Dict[str, pd.DataFrame],
        features_by_split: Dict[str, pd.DataFrame],
        split: str,
        use_case: str,
    ) -> EvaluationResult:
        from src.evaluation_engine.predictions import build_run_id
        run_id = build_run_id(self.cfg)

        per_model = {}
        for model_name, df_pred in predictions_by_model.items():
            per_model[model_name] = self.evaluate_predictions(
                model_name=model_name,
                df_pred=df_pred,
                df_features=features_by_split[split],
                split=split,
                use_case=use_case,
            )

        # Compare models by expected cost at their optimized thresholds
        comparison = self._compare_models(per_model)

        # Decision recommendation (overall + per-slice)
        decision = self._recommend(per_model, use_case=use_case)

        # Error analysis (top errors, clusters)
        errors = analyze_errors(self.cfg, predictions_by_model, features_by_split[split], use_case=use_case)

        # Visualization + report outputs
        if self.cfg.get("visualization", {}).get("enabled", True):
            plot_all(self.cfg, per_model=per_model, comparison=comparison, decision=decision)

        # Persist machine-readable summary
        outputs = Path(self.cfg["paths"]["outputs_dir"])
        ensure_dir(outputs / "metrics")
        ensure_dir(outputs / "reports")

        write_json(outputs / "metrics" / f"comparison__{run_id}__{split}.json", comparison)
        write_json(outputs / "metrics" / f"decision__{run_id}__{split}.json", decision)
        write_json(outputs / "reports" / f"errors__{run_id}__{split}.json", errors)

        overall = {"run_id": run_id, "split": split, "comparison": comparison}
        slices = self._collect_slice_table(per_model)
        return EvaluationResult(run_id=run_id, overall=overall, slices=slices, decision=decision, errors=errors)

    def _compare_models(self, per_model: Dict[str, Any]) -> Dict[str, Any]:
        rows = []
        for model_name, payload in per_model.items():
            best = payload["best"]
            rows.append({
                "model": model_name,
                "best_threshold": best["threshold"],
                "expected_cost": best["expected_cost"],
                "f1_at_best": best["metrics"]["f1"],
                "roc_auc": payload["overall"]["roc_auc"],
                "pr_auc": payload["overall"]["pr_auc"],
                "brier": payload["overall"]["brier"],
                "ece": payload["overall"]["ece"],
                "count": payload["overall"]["count"],
            })
        # Rank by expected cost
        rows_sorted = sorted(rows, key=lambda r: r["expected_cost"])
        return {"ranking": rows_sorted, "winner": rows_sorted[0]["model"] if rows_sorted else None}

    def _collect_slice_table(self, per_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Flatten slice results for UI/API
        out = []
        for model_name, payload in per_model.items():
            for s in payload["slices"]:
                out.append({**s, "model": model_name})
        return out

    def _recommend(self, per_model: Dict[str, Any], use_case: str) -> Dict[str, Any]:
        # Overall recommendation: minimize expected cost at optimal threshold
        ranking = self._compare_models(per_model)["ranking"]
        if not ranking:
            return {"recommended_model": None, "rationale": {"error": "no_models"}}
        best_model = ranking[0]["model"]
        best_threshold = ranking[0]["best_threshold"]

        # Per-slice recs: choose model with lowest slice expected cost (optionally)
        per_slice = []
        if self.cfg["decision"].get("per_slice_optimization", True):
            # Collect slices across models by slice_name
            by_slice = {}
            for model_name, payload in per_model.items():
                for s in payload["slices"]:
                    key = s["slice_name"]
                    by_slice.setdefault(key, []).append((model_name, s))
            for slice_name, entries in by_slice.items():
                # Only compare if each has cost computed
                entries = [e for e in entries if "decision" in e[1] and "expected_cost" in e[1]["decision"]]
                if not entries:
                    continue
                best = min(entries, key=lambda t: t[1]["decision"]["expected_cost"])
                per_slice.append({
                    "slice_name": slice_name,
                    "recommended_model": best[0],
                    "recommended_threshold": best[1]["decision"]["best_threshold"],
                    "expected_cost": best[1]["decision"]["expected_cost"],
                    "count": best[1]["count"],
                    "unstable": best[1].get("instability", {}).get("unstable", False),
                })

        return {
            "recommended_model": best_model,
            "recommended_threshold": float(best_threshold),
            "rationale": {
                "objective": "min_expected_cost",
                "use_case": use_case,
                "ranking": ranking[:5],
                "note": "This recommendation is context-dependent. Change the cost matrix or slices and the winner can change.",
            },
            "per_slice_recommendations": per_slice,
        }

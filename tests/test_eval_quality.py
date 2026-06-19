from __future__ import annotations

import numpy as np

from src.evaluation_engine.eval_quality import (
    compute_business_loss_reduction,
    compute_cost_increase,
    compute_eval_quality,
    compute_slice_diagnostics,
)
from src.evaluation_engine.metrics import compute_binary_metrics


def _cfg(tmp_path):
    costs_path = tmp_path / "costs.yaml"
    costs_path.write_text(
        "\n".join(
            [
                "use_cases:",
                "  fraud_strict:",
                "    binary:",
                "      TP: 0.0",
                "      TN: 0.0",
                "      FP: 5.0",
                "      FN: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "decision": {"costs_config": str(costs_path)},
        "evaluation": {"instability": {"min_count": 10}},
        "evaluation_quality": {
            "slice_drop_threshold": 0.35,
            "slice_cost_increase_threshold": 0.20,
            "baseline_threshold": 0.5,
            "cost_normalization_eps": 1e-9,
        },
    }


def test_eval_quality_counts_dataset_and_models(tmp_path):
    cfg = _cfg(tmp_path)
    per_model = {
        "winner": {
            "overall": {
                "precision": 0.8,
                "recall": 0.9,
                "f1": 0.85,
                "expected_cost_at_best_threshold": 0.2,
            },
            "slices": [],
            "best": {"threshold": 0.7, "expected_cost": 0.2},
        },
        "runnerup": {
            "overall": {
                "precision": 0.75,
                "recall": 0.82,
                "f1": 0.78,
                "expected_cost_at_best_threshold": 0.35,
            },
            "slices": [],
            "best": {"threshold": 0.6, "expected_cost": 0.35},
        },
    }
    model_scores = {
        "winner": {
            "y_true": np.array([0, 1, 1, 0]),
            "y_score": np.array([0.55, 0.52, 0.9, 0.2]),
        },
        "runnerup": {
            "y_true": np.array([0, 1, 1, 0]),
            "y_score": np.array([0.3, 0.6, 0.8, 0.4]),
        },
    }
    decision = {
        "recommended_model": "winner",
        "rationale": {
            "ranking": [
                {"model": "winner", "expected_cost": 0.2},
                {"model": "runnerup", "expected_cost": 0.35},
            ]
        },
    }

    quality = compute_eval_quality(
        cfg,
        run_id="run-1",
        split="test",
        use_case="fraud_strict",
        split_count=4,
        total_count=20,
        per_model=per_model,
        model_scores=model_scores,
        decision=decision,
        runtime={"total_sec": 12.3, "stages": {"load_data": 0.1}},
    )

    assert quality["dataset_size"] == {"split_count": 4, "total_count_if_available": 20}
    assert quality["models_compared"] == 2
    assert quality["runtime"]["total_sec"] == 12.3


def test_business_loss_reduction_computation(tmp_path):
    cfg = _cfg(tmp_path)
    per_model = {
        "winner": {
            "best": {"threshold": 0.6, "expected_cost": 0.25},
        },
    }
    model_scores = {
        "winner": {
            "y_true": np.array([0, 1, 1, 0]),
            "y_score": np.array([0.55, 0.52, 0.9, 0.2]),
        }
    }
    decision = {
        "recommended_model": "winner",
        "rationale": {
            "ranking": [
                {"model": "winner", "expected_cost": 0.25},
                {"model": "runnerup", "expected_cost": 0.4},
            ]
        },
    }

    business = compute_business_loss_reduction(
        cfg,
        use_case="fraud_strict",
        decision=decision,
        per_model=per_model,
        model_scores=model_scores,
        baseline_threshold=0.5,
    )

    assert abs(business["baseline_expected_cost"] - 1.25) < 1e-9
    assert abs(business["optimized_expected_cost"] - 0.25) < 1e-9
    assert abs(business["reduction_pct"] - 0.8) < 1e-9
    assert abs(business["winner_vs_runnerup_cost_delta"] - 0.15) < 1e-9


def test_slice_diagnostic_detection_thresholds(tmp_path):
    cfg = _cfg(tmp_path)
    overall = {
        "precision": 0.8,
        "recall": 0.9,
        "f1": 0.85,
        "expected_cost_at_best_threshold": 0.5,
    }
    slices = [
        {
            "slice_name": "low_recall",
            "count": 20,
            "metrics": {"precision": 0.79, "recall": 0.5, "f1": 0.6},
            "metrics_validity": {"precision_valid": True, "recall_valid": True, "f1_valid": True, "reasons": []},
            "decision": {"expected_cost": 0.52},
            "instability": {"unstable": False},
        },
        {
            "slice_name": "cost_spike",
            "count": 25,
            "metrics": {"precision": 0.78, "recall": 0.88, "f1": 0.84},
            "metrics_validity": {"precision_valid": True, "recall_valid": True, "f1_valid": True, "reasons": []},
            "decision": {"expected_cost": 0.7},
            "instability": {"unstable": False},
        },
        {
            "slice_name": "unstable_bad_slice",
            "count": 5,
            "metrics": {"precision": 0.1, "recall": 0.1, "f1": 0.1},
            "metrics_validity": {"precision_valid": True, "recall_valid": True, "f1_valid": True, "reasons": []},
            "decision": {"expected_cost": 1.0},
            "instability": {"unstable": True},
        },
    ]

    diagnostics = compute_slice_diagnostics(cfg, overall=overall, slices=slices)

    assert diagnostics["diagnostics_found"] == 2
    assert [row["slice_name"] for row in diagnostics["top_diagnostics"]] == ["cost_spike", "low_recall"]
    assert diagnostics["top_diagnostics"][0]["triggered_by"] == "cost_increase"
    assert diagnostics["top_diagnostics"][0]["metric_used"] is None
    assert diagnostics["top_diagnostics"][1]["metric_used"] == "recall"


def test_no_false_diagnostic_when_metric_invalid():
    metrics = compute_binary_metrics(
        y_true=np.array([0, 0, 0, 0]),
        y_prob_pos=np.array([0.1, 0.2, 0.3, 0.4]),
        threshold=0.5,
        n_bins_ece=5,
    )

    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["metrics_validity"]["recall_valid"] is False
    assert "no_positive_labels" in metrics["metrics_validity"]["reasons"]

    cfg = {
        "evaluation": {"instability": {"min_count": 1}},
        "evaluation_quality": {
            "slice_drop_threshold": 0.35,
            "slice_cost_increase_threshold": 10.0,
            "cost_normalization_eps": 1e-9,
        },
    }
    diagnostics = compute_slice_diagnostics(
        cfg,
        overall={
            "precision": 0.8,
            "recall": 0.9,
            "f1": 0.85,
            "expected_cost_at_best_threshold": 0.5,
        },
        slices=[
            {
                "slice_name": "label_negative",
                "count": 4,
                "metrics": metrics,
                "metrics_validity": metrics["metrics_validity"],
                "decision": {"expected_cost": 0.55},
                "instability": {"unstable": False},
            }
        ],
    )

    assert diagnostics["diagnostics_found"] == 0


def test_cost_increase_pct_computation():
    positive = compute_cost_increase(overall_cost=0.5, slice_cost=0.7, eps=1e-9)
    negative = compute_cost_increase(overall_cost=-2.0, slice_cost=-1.0, eps=1e-9)

    assert abs(positive["cost_increase_abs"] - 0.2) < 1e-9
    assert abs(positive["cost_increase_pct"] - 0.4) < 1e-9
    assert abs(negative["cost_increase_abs"] - 1.0) < 1e-9
    assert abs(negative["cost_increase_pct"] - 0.5) < 1e-9


def test_diagnostic_trigger_fields_present(tmp_path):
    cfg = _cfg(tmp_path)
    metrics = compute_binary_metrics(
        y_true=np.array([0, 1, 1, 0]),
        y_prob_pos=np.array([0.2, 0.3, 0.4, 0.1]),
        threshold=0.5,
        n_bins_ece=5,
    )

    diagnostics = compute_slice_diagnostics(
        cfg,
        overall={
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "expected_cost_at_best_threshold": 0.5,
        },
        slices=[
            {
                "slice_name": "metric_drop_slice",
                "count": 10,
                "metrics": metrics,
                "metrics_validity": metrics["metrics_validity"],
                "decision": {"expected_cost": 0.55},
                "instability": {"unstable": False},
            }
        ],
    )

    row = diagnostics["top_diagnostics"][0]
    assert "triggered_by" in row
    assert "metric_used" in row
    assert "metrics_validity" in row
    assert row["triggered_by"] == "metric_drop"

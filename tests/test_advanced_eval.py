import numpy as np

from src.decision_engine.sensitivity import build_cost_scenarios, run_cost_sensitivity
from src.evaluation_engine.advanced import calibrate_scores, threshold_ci_bootstrap


def test_calibrate_scores_platt_shape():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    out = calibrate_scores(y_true, y_score, method="platt")
    assert out.method == "platt"
    assert len(out.y_score_calibrated) == len(y_score)
    assert np.all((out.y_score_calibrated >= 0.0) & (out.y_score_calibrated <= 1.0))


def test_threshold_ci_bootstrap_basic():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.2, 0.9, 0.85, 0.7, 0.3, 0.8, 0.4])
    grid = np.array([0.2, 0.4, 0.6, 0.8])
    costs = {"TP": -1.0, "TN": 0.0, "FP": 2.0, "FN": 5.0}
    ci = threshold_ci_bootstrap(y_true, y_score, grid, costs, iters=50, alpha=0.1, seed=1)
    assert "ci" in ci and len(ci["ci"]) == 2
    assert ci["ci"][0] <= ci["ci"][1]


def test_cost_sensitivity_runs():
    grid = np.array([0.2, 0.4, 0.6])
    scenarios = build_cost_scenarios({"TP": -1.0, "TN": 0.0, "FP": 2.0, "FN": 5.0}, [1.0], [1.0, 2.0])
    preds = {
        "m1": {"y_true": np.array([0, 1, 0, 1]), "y_score": np.array([0.1, 0.8, 0.3, 0.7])},
        "m2": {"y_true": np.array([0, 1, 0, 1]), "y_score": np.array([0.2, 0.6, 0.4, 0.55])},
    }
    out = run_cost_sensitivity(preds, grid, scenarios)
    assert len(out["scenarios"]) == 2
    assert "winner_robustness" in out


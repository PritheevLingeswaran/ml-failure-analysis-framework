import numpy as np
import pandas as pd

from src.evaluation_engine.drift import compute_drift_report


def _frame(x, c):
    return pd.DataFrame({"x": x, "c": c, "label": 0, "id": range(len(x))})


def test_no_drift_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=800)
    c = rng.choice(["a", "b", "c"], size=800)
    train = _frame(x, c)
    test = _frame(x.copy(), c.copy())
    report = compute_drift_report(train, test, exclude_cols=["label", "id"], psi_warn=0.2, tv_warn=0.2)
    assert report["num_features_checked"] == 2
    assert report["num_drifted"] == 0
    psi = next(f["score"] for f in report["all_features"] if f["feature"] == "x")
    assert psi < 0.1


def test_numeric_shift_is_flagged():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=800)
    train = _frame(x, ["a"] * 800)
    test = _frame(x + 5.0, ["a"] * 800)  # large mean shift
    report = compute_drift_report(train, test, exclude_cols=["label", "id"], psi_warn=0.2, tv_warn=0.2)
    xf = next(f for f in report["all_features"] if f["feature"] == "x")
    assert xf["metric"] == "psi"
    assert xf["drifted"] is True
    assert xf["score"] >= 0.2


def test_categorical_shift_is_flagged():
    train = _frame(np.zeros(400), ["a"] * 400)
    test = _frame(np.zeros(400), ["b"] * 400)  # completely different category
    report = compute_drift_report(train, test, exclude_cols=["label", "id", "x"], psi_warn=0.2, tv_warn=0.2)
    cf = next(f for f in report["all_features"] if f["feature"] == "c")
    assert cf["metric"] == "tv_distance"
    assert cf["drifted"] is True
    assert cf["score"] > 0.9


def test_excluded_columns_are_skipped():
    train = _frame(np.zeros(10), ["a"] * 10)
    test = _frame(np.zeros(10), ["a"] * 10)
    report = compute_drift_report(train, test, exclude_cols=["label", "id", "x", "c"], psi_warn=0.2, tv_warn=0.2)
    assert report["num_features_checked"] == 0

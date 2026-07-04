import numpy as np
from src.evaluation_engine.metrics import compute_binary_metrics, expected_calibration_error


def test_binary_metrics_shape():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.8, 0.6, 0.4])
    m = compute_binary_metrics(y_true, y_score, threshold=0.5, n_bins_ece=5)
    assert "accuracy" in m and "confusion" in m


def test_ece_perfectly_calibrated_is_near_zero():
    # Confident and correct: confidence 0.99 everywhere, all predictions correct.
    y_true = np.array([1] * 50 + [0] * 50)
    y_score = np.array([0.99] * 50 + [0.01] * 50)
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    assert ece < 0.02


def test_ece_overconfident_is_large_and_bounded():
    # Confidence 0.99 but only half correct -> gap ~0.49.
    y_true = np.array([1] * 50 + [0] * 50)
    y_score = np.array([0.99] * 100)  # always predicts positive with high confidence
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    assert 0.45 < ece <= 0.5


def test_ece_is_in_valid_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_score = rng.uniform(0, 1, size=500)
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    # ECE (confidence form) is a weighted average of |acc - conf|, so in [0, 1].
    assert 0.0 <= ece <= 1.0


def test_ece_not_inflated_by_confident_negatives():
    # Regression test for the old bug: a well-separated, well-calibrated set of
    # mostly-negative predictions must NOT report a huge ECE.
    y_true = np.array([0] * 90 + [1] * 10)
    y_score = np.array([0.03] * 90 + [0.97] * 10)  # all correct, well calibrated
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    assert ece < 0.05


def test_ece_empty_is_nan():
    ece = expected_calibration_error(np.array([]), np.array([]), n_bins=10)
    assert np.isnan(ece)

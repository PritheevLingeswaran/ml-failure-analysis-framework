import numpy as np
from src.evaluation_engine.metrics import compute_binary_metrics

def test_binary_metrics_shape():
    y_true = np.array([0,1,0,1])
    y_score = np.array([0.2,0.8,0.6,0.4])
    m = compute_binary_metrics(y_true, y_score, threshold=0.5, n_bins_ece=5)
    assert "accuracy" in m and "confusion" in m

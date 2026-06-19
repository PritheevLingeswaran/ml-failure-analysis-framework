import numpy as np
from src.decision_engine.costs import expected_cost_binary

def test_expected_cost_binary_basic():
    y_true = np.array([0,0,1,1])
    y_score = np.array([0.1,0.9,0.2,0.8])
    costs = {"TP": -1.0, "TN": 0.0, "FP": 2.0, "FN": 5.0}
    # threshold 0.5 => preds: [0,1,0,1] => tn=1 fp=1 fn=1 tp=1
    # total cost = 1*0 + 1*2 + 1*5 + 1*(-1) = 6; mean=1.5
    c = expected_cost_binary(y_true, y_score, 0.5, costs)
    assert abs(c - 1.5) < 1e-9

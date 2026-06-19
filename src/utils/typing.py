from __future__ import annotations
from typing import Dict, Any, TypedDict, Optional, List

class Predictions(TypedDict):
    id: List[str]
    y_true: List[int]
    y_pred: List[int]
    y_score: List[float]
    model_name: str
    split: str

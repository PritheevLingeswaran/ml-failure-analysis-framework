from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class ModelArtifacts:
    name: str
    version: str
    path: str

class BaseModel:
    """Unified interface.

    Why:
    - Evaluation needs a consistent contract; otherwise you compare apples to oranges.
    - This keeps training/inference aligned across models.
    """
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        return np.argmax(proba, axis=1)

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, name: str, path: str) -> "BaseModel":
        raise NotImplementedError

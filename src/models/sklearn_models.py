from __future__ import annotations
import logging
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

class SklearnLogReg(BaseModel):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = LogisticRegression(**self.params)
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, name: str, path: str) -> "SklearnLogReg":
        inst = cls(name=name, params={})
        inst._model = joblib.load(path)
        return inst

class SklearnRandomForest(BaseModel):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = RandomForestClassifier(**self.params)
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, name: str, path: str) -> "SklearnRandomForest":
        inst = cls(name=name, params={})
        inst._model = joblib.load(path)
        return inst

def try_build_xgboost(name: str, params: Dict[str, Any]):
    """Optional dependency. We keep a clean fallback for environments without xgboost."""
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        logger.warning("xgboost not installed; skipping xgb model. (%s)", e)
        return None

    class XGBoostClassifierModel(BaseModel):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            self._model = xgb.XGBClassifier(**params)
            self._model.fit(X, y)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return self._model.predict_proba(X)

        def save(self, path: str) -> None:
            joblib.dump(self._model, path)

        @classmethod
        def load(cls, name: str, path: str) -> "XGBoostClassifierModel":
            inst = cls(name=name, params={})
            inst._model = joblib.load(path)
            return inst

    return XGBoostClassifierModel(name=name, params=params)

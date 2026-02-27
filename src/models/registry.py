from __future__ import annotations
import logging
from typing import Any, Dict, List

from src.models.base import BaseModel
from src.models.sklearn_models import SklearnLogReg, SklearnRandomForest, try_build_xgboost

logger = logging.getLogger(__name__)

def build_models(cfg: Dict[str, Any]) -> List[BaseModel]:
    out: List[BaseModel] = []
    for item in cfg["models"]["registry"]:
        if item.get("enabled") is False:
            continue
        name = item["name"]
        mtype = item["type"]
        params = item.get("params", {}) or {}
        if mtype == "sklearn_logreg":
            out.append(SklearnLogReg(name, params))
        elif mtype == "sklearn_random_forest":
            out.append(SklearnRandomForest(name, params))
        elif mtype == "xgboost_classifier":
            model = try_build_xgboost(name, params)
            if model is not None:
                out.append(model)
        else:
            raise ValueError(f"Unknown model type: {mtype}")
    if len(out) < 2:
        logger.warning("Only %s models enabled. Requirements expect 2–3.", len(out))
    return out

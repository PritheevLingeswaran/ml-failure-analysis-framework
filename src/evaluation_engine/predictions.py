from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from src.utils.paths import ensure_dir
from src.utils.io import write_csv, read_csv

def _hash_config(cfg: Dict[str, Any]) -> str:
    stable = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(stable).hexdigest()[:12]

def predictions_dir(cfg: Dict[str, Any]) -> Path:
    base = Path(cfg["paths"]["data_processed_dir"]) / "predictions"
    ensure_dir(base)
    return base

def build_run_id(cfg: Dict[str, Any]) -> str:
    # Run id must be deterministic for audit, but unique across config changes.
    return _hash_config({
        "data": cfg.get("data", {}),
        "models": cfg.get("models", {}),
        "evaluation": cfg.get("evaluation", {}),
        "slicing": cfg.get("slicing", {}),
        "decision": cfg.get("decision", {}),
    })

def save_predictions(cfg: Dict[str, Any], df: pd.DataFrame, model_name: str, split: str) -> Path:
    run_id = build_run_id(cfg)
    outdir = predictions_dir(cfg) / run_id
    ensure_dir(outdir)
    path = outdir / f"{model_name}__{split}.csv"
    write_csv(path, df)
    return path

def load_predictions(cfg: Dict[str, Any], model_name: str, split: str) -> pd.DataFrame:
    run_id = build_run_id(cfg)
    path = predictions_dir(cfg) / run_id / f"{model_name}__{split}.csv"
    return read_csv(path)

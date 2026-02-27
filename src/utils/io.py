from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

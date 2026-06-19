from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets.base import DatasetBundle
from src.utils.random import set_global_seed

logger = logging.getLogger(__name__)

def make_splits(
    df: pd.DataFrame,
    label_col: str,
    id_col: str,
    text_col: Optional[str],
    seed: int,
    test_size: float,
    val_size: float,
    strategy: str = "random",
    time_col: Optional[str] = None,
) -> DatasetBundle:
    """Deterministic splits.

    Why:
    - Without deterministic splits, comparing models is meaningless and not auditable.
    - You need repeatability for decision logs, regression checks, and incident retrospectives.
    """
    set_global_seed(seed)

    if strategy == "time":
        if not time_col or time_col not in df.columns:
            raise ValueError(f"time split requested but time_col='{time_col}' missing from dataset")
        ordered = df.copy()
        ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
        ordered = ordered.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        n = len(ordered)
        n_test = max(1, int(round(n * test_size)))
        n_train_val = max(1, n - n_test)
        n_val = max(1, int(round(n_train_val * (val_size / max(1e-12, (1.0 - test_size))))))
        n_train = max(1, n_train_val - n_val)
        train = ordered.iloc[:n_train].copy()
        val = ordered.iloc[n_train:n_train + n_val].copy()
        test = ordered.iloc[n_train + n_val:].copy()
    else:
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[label_col] if df[label_col].nunique() > 1 else None,
        )
        # val_size is fraction of remaining
        val_fraction = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_fraction,
            random_state=seed,
            stratify=train_val[label_col] if train_val[label_col].nunique() > 1 else None,
        )

    logger.info("Splits(strategy=%s): train=%s val=%s test=%s", strategy, len(train), len(val), len(test))
    return DatasetBundle(train=train, val=val, test=test, label_col=label_col, id_col=id_col, text_col=text_col)

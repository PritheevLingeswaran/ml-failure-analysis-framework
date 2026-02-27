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
    val_size: float
) -> DatasetBundle:
    """Deterministic splits.

    Why:
    - Without deterministic splits, comparing models is meaningless and not auditable.
    - You need repeatability for decision logs, regression checks, and incident retrospectives.
    """
    set_global_seed(seed)

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

    logger.info("Splits: train=%s val=%s test=%s", len(train), len(val), len(test))
    return DatasetBundle(train=train, val=val, test=test, label_col=label_col, id_col=id_col, text_col=text_col)

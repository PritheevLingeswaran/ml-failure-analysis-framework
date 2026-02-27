from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# --- Ensure repo root is on sys.path so `import src...` works reliably ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.paths import ensure_dir

from src.datasets.csv_classification import CSVClassificationDataset
from src.datasets.splits import make_splits
from src.models.registry import build_models
from src.evaluation_engine.predictions import save_predictions

logger = logging.getLogger(__name__)


def _drop_cols(df: pd.DataFrame, label_col: str, id_col: str, text_col: str | None) -> pd.DataFrame:
    """Drop non-feature columns (label/id/text)."""
    cols = [label_col, id_col]
    if text_col:
        cols.append(text_col)
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore").copy()


def _make_feature_schema(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    id_col: str,
    text_col: str | None,
) -> pd.Index:
    """
    Build a stable feature schema (columns) across all splits.

    Why:
    - pd.get_dummies() can create different columns in different splits.
    - Models require the same feature columns at train and inference.
    """
    all_raw = pd.concat(
        [
            _drop_cols(train_df, label_col, id_col, text_col),
            _drop_cols(val_df, label_col, id_col, text_col),
            _drop_cols(test_df, label_col, id_col, text_col),
        ],
        axis=0,
        ignore_index=True,
    )
    all_encoded = pd.get_dummies(all_raw, drop_first=False)
    return all_encoded.columns


def _feature_columns(
    df: pd.DataFrame,
    label_col: str,
    id_col: str,
    text_col: str | None,
    feature_cols: pd.Index,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create numeric feature matrix + label vector.

    - One-hot encodes categoricals (region etc.)
    - Reindexes to a stable schema
    """
    X_raw = _drop_cols(df, label_col, id_col, text_col)
    X_enc = pd.get_dummies(X_raw, drop_first=False)

    # Align columns to the global schema
    X = X_enc.reindex(columns=feature_cols, fill_value=0)

    # Ensure everything is numeric
    # (If this fails, you still have a non-encoded column leaking in.)
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns leaked into features: {non_numeric}")

    y = df[label_col].astype(int)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    dataset_cfg = cfg["data"]["dataset"]
    label_col = dataset_cfg["label_col"]
    id_col = dataset_cfg["id_col"]
    text_col = dataset_cfg.get("text_col")

    loader = CSVClassificationDataset(
        path=cfg["paths"]["data_raw"],
        label_col=label_col,
        id_col=id_col,
        text_col=text_col,
    )
    df = loader.load()

    bundle = make_splits(
        df=df,
        label_col=label_col,
        id_col=id_col,
        text_col=text_col,
        seed=int(cfg["data"]["split"]["seed"]),
        test_size=float(cfg["data"]["split"]["test_size"]),
        val_size=float(cfg["data"]["split"]["val_size"]),
    )

    # --- Build stable feature schema across splits (critical) ---
    feature_cols = _make_feature_schema(bundle.train, bundle.val, bundle.test, label_col, id_col, text_col)

    models = build_models(cfg)
    processed_dir = Path(cfg["paths"]["data_processed_dir"])
    models_dir = ensure_dir(processed_dir / "models")

    for m in models:
        Xtr, ytr = _feature_columns(bundle.train, label_col, id_col, text_col, feature_cols)
        Xva, yva = _feature_columns(bundle.val, label_col, id_col, text_col, feature_cols)
        Xte, yte = _feature_columns(bundle.test, label_col, id_col, text_col, feature_cols)

        logger.info("Training model=%s on X=%s", m.name, Xtr.shape)
        m.fit(Xtr, ytr)

        model_path = models_dir / f"{m.name}.joblib"
        m.save(str(model_path))
        logger.info("Saved model=%s -> %s", m.name, model_path)

        # Generate predictions for each split
        for split_name, (X, split_df) in {
            "train": (Xtr, bundle.train),
            "val": (Xva, bundle.val),
            "test": (Xte, bundle.test),
        }.items():
            proba = m.predict_proba(X)

            if proba.shape[1] == 2:
                y_score = proba[:, 1]
            else:
                # multiclass: store max prob as confidence proxy
                y_score = np.max(proba, axis=1)

            df_pred = pd.DataFrame(
                {
                    id_col: split_df[id_col].values,
                    label_col: split_df[label_col].values,
                    "y_score": y_score,
                }
            )

            save_path = save_predictions(cfg, df_pred, model_name=m.name, split=split_name)
            logger.info("Saved predictions: %s", save_path)


if __name__ == "__main__":
    main()

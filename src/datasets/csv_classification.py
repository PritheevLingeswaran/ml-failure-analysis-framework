from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from src.datasets.base import DatasetLoader
from src.utils.io import read_csv

logger = logging.getLogger(__name__)

class CSVClassificationDataset(DatasetLoader):
    def __init__(self, path: str, label_col: str, id_col: str, text_col: str | None = None):
        self.path = path
        self.label_col = label_col
        self.id_col = id_col
        self.text_col = text_col

    def load(self) -> pd.DataFrame:
        df = read_csv(self.path)

        # Minimal hygiene; production pipelines should do more
        df = df.drop_duplicates()
        if self.id_col not in df.columns:
            df[self.id_col] = [f"row_{i}" for i in range(len(df))]
        if self.label_col not in df.columns:
            raise ValueError(f"label_col '{self.label_col}' not in dataset columns={list(df.columns)}")

        # Remove rows with missing labels
        df = df.dropna(subset=[self.label_col]).copy()

        # Convert label to int if possible
        df[self.label_col] = df[self.label_col].astype(int)

        logger.info("Loaded dataset: %s rows, %s cols", len(df), df.shape[1])
        return df

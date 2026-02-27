from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_col: str
    id_col: str
    text_col: Optional[str] = None

class DatasetLoader:
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

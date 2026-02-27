from __future__ import annotations
import numpy as np
import random

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    out_path = Path(cfg["paths"]["data_raw"])
    ensure_dir(out_path.parent)

    # Synthetic dataset designed to demonstrate slices + text length effects.
    X, y = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.75, 0.25],
        class_sep=1.0,
        random_state=int(cfg["data"]["split"]["seed"]),
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y.astype(int)
    df["id"] = [f"ex_{i}" for i in range(len(df))]

    # Add metadata columns used for rule slices (optional)
    rng = np.random.default_rng(42)
    df["region"] = rng.choice(["US", "IN", "EU"], size=len(df), p=[0.55, 0.35, 0.10])
    df["amount"] = np.round(rng.lognormal(mean=6.5, sigma=0.6, size=len(df)), 2)

    # Create a text column with different lengths correlated with label and region
    base_phrases = [
        "payment failed",
        "account locked",
        "refund requested",
        "chargeback dispute",
        "address mismatch",
        "login unusual activity",
    ]
    texts = []
    for i in range(len(df)):
        phrase = rng.choice(base_phrases)
        noise_len = int(rng.integers(5, 200))
        # Harder positives get longer text
        extra = " details" * (1 + int(df.loc[i, "label"] == 1) + int(df.loc[i, "region"] == "IN"))
        text = (phrase + extra + " ") + ("x" * noise_len)
        texts.append(text)
    df["text"] = texts

    df.to_csv(out_path, index=False)
    logger.info("Wrote synthetic dataset to %s (rows=%s)", out_path, len(df))

if __name__ == "__main__":
    main()

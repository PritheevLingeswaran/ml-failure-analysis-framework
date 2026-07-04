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

def build_real_breast_cancer(seed: int) -> pd.DataFrame:
    """Real, bundled dataset: Breast Cancer Wisconsin (569 cases, 30 features).

    Positive class = malignant (target==0 in sklearn), i.e. the costly-to-miss
    outcome. No text/region/time columns — the framework degrades gracefully
    (text/fairness slices simply do not apply).
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Sanitize feature names (spaces -> underscores) so slice queries are simple.
    feature_cols = {c: c.replace(" ", "_") for c in data.feature_names}
    df = df.rename(columns=feature_cols)
    df["label"] = (data.target == 0).astype(int)  # 1 = malignant
    df = df.drop(columns=["target"])
    df["id"] = [f"case_{i}" for i in range(len(df))]
    # Deterministic shuffle for a stable, well-mixed split.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_synthetic(seed: int) -> pd.DataFrame:
    # Synthetic dataset designed to demonstrate slices + text length effects.
    X, y = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.75, 0.25],
        class_sep=1.0,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y.astype(int)
    df["id"] = [f"ex_{i}" for i in range(len(df))]

    # Add metadata columns used for rule slices (optional)
    rng = np.random.default_rng(42)
    df["region"] = rng.choice(["US", "IN", "EU"], size=len(df), p=[0.55, 0.35, 0.10])
    df["amount"] = np.round(rng.lognormal(mean=6.5, sigma=0.6, size=len(df)), 2)
    # Deterministic synthetic event time for temporal split experiments.
    base_ts = pd.Timestamp("2024-01-01")
    df["event_time"] = base_ts + pd.to_timedelta(np.arange(len(df)), unit="h")

    # Create a text column with different lengths correlated with label and region.
    base_phrases = [
        "payment failed",
        "account locked",
        "refund requested",
        "chargeback dispute",
        "address mismatch",
        "login unusual activity",
    ]
    # Realistic filler vocabulary so error examples and failure clusters are
    # human-readable (rather than opaque "xxxx" padding).
    filler_vocab = [
        "customer", "transaction", "reported", "system", "reference", "number",
        "attempt", "verification", "pending", "review", "support", "ticket",
        "urgent", "flagged", "balance", "transfer", "merchant", "declined",
        "gateway", "timeout", "retry", "session", "device", "location",
        "email", "phone", "update", "confirm", "status", "reason", "notes",
        "amount", "history", "profile", "risk", "score", "manual", "queue",
    ]
    texts = []
    for i in range(len(df)):
        phrase = rng.choice(base_phrases)
        # Harder positives get longer text
        extra = " details" * (1 + int(df.loc[i, "label"] == 1) + int(df.loc[i, "region"] == "IN"))
        n_words = int(rng.integers(3, 40))
        noise = " ".join(rng.choice(filler_vocab, size=n_words))
        text = f"{phrase}{extra} {noise}"
        texts.append(text)
    df["text"] = texts
    return df


_BUILDERS = {
    "synthetic": build_synthetic,
    "sklearn_breast_cancer": build_real_breast_cancer,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    out_path = Path(cfg["paths"]["data_raw"])
    ensure_dir(out_path.parent)

    source = cfg.get("data", {}).get("source", "synthetic")
    if source not in _BUILDERS:
        raise ValueError(f"Unknown data.source='{source}'. Options: {sorted(_BUILDERS)}")

    seed = int(cfg["data"]["split"]["seed"])
    df = _BUILDERS[source](seed)

    df.to_csv(out_path, index=False)
    logger.info("Wrote %s dataset to %s (rows=%s, cols=%s)", source, out_path, len(df), df.shape[1])


if __name__ == "__main__":
    main()

import pandas as pd

from src.datasets.splits import make_splits


def test_make_splits_time_strategy():
    df = pd.DataFrame(
        {
            "id": [f"i{i}" for i in range(20)],
            "label": [i % 2 for i in range(20)],
            "event_time": pd.date_range("2024-01-01", periods=20, freq="D"),
            "x": list(range(20)),
        }
    )
    bundle = make_splits(
        df=df,
        label_col="label",
        id_col="id",
        text_col=None,
        seed=42,
        test_size=0.2,
        val_size=0.1,
        strategy="time",
        time_col="event_time",
    )
    assert len(bundle.train) > 0 and len(bundle.val) > 0 and len(bundle.test) > 0
    assert bundle.train["event_time"].max() <= bundle.val["event_time"].min()
    assert bundle.val["event_time"].max() <= bundle.test["event_time"].min()


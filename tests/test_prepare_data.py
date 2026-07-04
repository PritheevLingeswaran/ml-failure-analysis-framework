from scripts.prepare_data import build_real_breast_cancer, build_synthetic


def test_real_builder_produces_clean_real_dataset():
    df = build_real_breast_cancer(seed=42)
    assert len(df) == 569
    assert "label" in df.columns and "id" in df.columns
    assert set(df["label"].unique()) <= {0, 1}
    assert df["label"].sum() > 0  # has malignant (positive) cases
    # Feature names must be query-safe (no spaces) for slice rules.
    assert all(" " not in c for c in df.columns)
    # Deterministic for a fixed seed.
    assert df["id"].iloc[0] == build_real_breast_cancer(seed=42)["id"].iloc[0]


def test_synthetic_builder_has_expected_columns_and_readable_text():
    df = build_synthetic(seed=42)
    assert len(df) == 6000
    for col in ["label", "id", "region", "amount", "event_time", "text"]:
        assert col in df.columns
    # Regression: text must be readable filler, not "xxxx" padding.
    sample = " ".join(df["text"].head(20).tolist()).lower()
    assert "xxxx" not in sample

import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.utils.config import load_config
    from src.api.app import create_app
    from src.api import routes
    from src.db import base as db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("MLFA_API_KEY", raising=False)
    db.reset_engine()
    routes.reset_cache()

    cfg = load_config("configs/dev.yaml")
    cfg["paths"]["data_processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    yield TestClient(create_app(cfg))
    db.reset_engine()


def _csv(n=300, n_classes=2):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, n_classes=n_classes, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["label"] = y
    return df.to_csv(index=False).encode()


def test_upload_run_and_history(client):
    up = client.post("/datasets/upload", files={"file": ("d.csv", _csv(), "text/csv")}, data={"label_col": "label"})
    assert up.status_code == 200, up.text
    ds = up.json()
    assert ds["n_rows"] == 300 and ds["label_col"] == "label" and ds["n_cols"] == 9

    assert len(client.get("/datasets").json()["datasets"]) == 1
    assert client.get(f"/datasets/{ds['id']}").json()["id"] == ds["id"]

    run = client.post("/runs", json={"dataset_id": ds["id"], "use_case": "default"})
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["status"] == "completed"
    assert body["winner_model"] in {"logreg", "rf", "ensemble_avg"}
    assert body["recommended_threshold"] is not None
    assert body["summary"]["ranking"]

    runs = client.get("/runs").json()["runs"]
    assert len(runs) == 1 and runs[0]["id"] == body["id"]
    assert client.get(f"/runs/{body['id']}").json()["id"] == body["id"]


def test_upload_rejects_non_binary_label(client):
    r = client.post("/datasets/upload", files={"file": ("d.csv", _csv(n=90, n_classes=3), "text/csv")})
    assert r.status_code == 422


def test_upload_rejects_non_csv_content_type(client):
    r = client.post("/datasets/upload", files={"file": ("d.json", b"{}", "application/json")})
    assert r.status_code == 415


def test_upload_rejects_missing_label_column(client):
    r = client.post(
        "/datasets/upload",
        files={"file": ("d.csv", _csv(), "text/csv")},
        data={"label_col": "nonexistent"},
    )
    assert r.status_code == 400


def test_get_missing_dataset_404(client):
    assert client.get("/datasets/nope").status_code == 404
    assert client.get("/runs/nope").status_code == 404

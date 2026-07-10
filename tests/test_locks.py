import threading
import time

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.api.locks import LocalLockManager, LockBusy


def test_same_key_serializes_and_times_out():
    m = LocalLockManager()
    order = []
    held = threading.Event()

    def holder():
        with m.lock("k", blocking_timeout=5):
            order.append("in")
            held.set()
            time.sleep(0.3)
            order.append("out")

    t = threading.Thread(target=holder)
    t.start()
    held.wait(2)
    # While the key is held, a waiter with a tiny timeout must give up.
    with pytest.raises(LockBusy):
        with m.lock("k", blocking_timeout=0.05):
            pass
    t.join()
    assert order == ["in", "out"]


def test_different_keys_do_not_block():
    m = LocalLockManager()
    with m.lock("k1", blocking_timeout=1):
        with m.lock("k2", blocking_timeout=1):  # different key -> no contention
            pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.utils.config import load_config
    from src.api.app import create_app
    from src.api import routes, locks
    from src.db import base as db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("MLFA_API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    db.reset_engine()
    routes.reset_cache()
    locks.reset_lock_manager()
    cfg = load_config("configs/dev.yaml")
    cfg["paths"]["data_processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    yield TestClient(create_app(cfg))
    db.reset_engine()


def _upload(client):
    X, y = make_classification(n_samples=250, n_features=6, n_informative=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["label"] = y
    r = client.post("/datasets/upload", files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")})
    return r.json()["id"]


def test_concurrent_runs_same_dataset_are_serialized(client):
    ds_id = _upload(client)
    results = []

    def run():
        r = client.post("/runs", json={"dataset_id": ds_id, "use_case": "default"})
        results.append(r.status_code)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Serialized by the lock: both complete successfully (without the lock they
    # would race on the shared predictions dir). No 500s.
    assert results == [200, 200]
    assert len(client.get("/runs").json()["runs"]) == 2

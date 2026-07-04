import threading
import time

import pytest

from src.utils.config import load_config
from src.api import routes


def _canned_result():
    return {
        "comparison": {"ranking": [{"model": "rf", "expected_cost": 0.01}], "winner": "rf"},
        "per_model": {"rf": {"overall": {}, "slices": [], "cost_curve": {"thresholds": [], "expected_costs": []}}},
        "slice_table": [],
        "decision": {
            "recommended_model": "rf",
            "recommended_threshold": 0.14,
            "rationale": {"objective": "min_expected_cost", "use_case": "fraud_strict", "ranking": [], "note": "n"},
            "per_slice_recommendations": [],
        },
        "errors": {"top_false_positives": {}, "top_false_negatives": {}, "clusters": {}},
        "drift": {},
        "cost_sensitivity": {},
        "eval_quality": {"quality": "ok"},
        "run_id": "testrun",
        "split": "test",
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    routes._CACHE.clear()
    routes._KEY_LOCKS.clear()
    yield
    routes._CACHE.clear()
    routes._KEY_LOCKS.clear()


def test_single_flight_computes_once_under_concurrency(monkeypatch):
    cfg = load_config("configs/dev.yaml")
    calls = {"n": 0}

    def slow_eval(cfg, split, use_case):
        calls["n"] += 1  # only the single-flight winner runs this
        time.sleep(0.3)
        return _canned_result()

    monkeypatch.setattr(routes, "run_evaluate_in_memory", slow_eval)

    use_case = cfg["decision"]["default_use_case"]
    results = []

    def worker():
        results.append(routes._evaluate_cached(cfg, use_case=use_case, split="test"))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1  # 10 concurrent callers, exactly one computation
    assert len(results) == 10
    assert all(r["run_id"] == "testrun" for r in results)


def test_cache_hit_avoids_recompute(monkeypatch):
    cfg = load_config("configs/dev.yaml")
    calls = {"n": 0}

    def counting_eval(cfg, split, use_case):
        calls["n"] += 1
        return _canned_result()

    monkeypatch.setattr(routes, "run_evaluate_in_memory", counting_eval)
    use_case = cfg["decision"]["default_use_case"]
    routes._evaluate_cached(cfg, use_case=use_case, split="test")
    routes._evaluate_cached(cfg, use_case=use_case, split="test")
    assert calls["n"] == 1


def test_compare_endpoint_shape(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    cfg = load_config("configs/dev.yaml")
    monkeypatch.setattr(routes, "run_evaluate_in_memory", lambda cfg, split, use_case: _canned_result())

    client = TestClient(create_app(cfg))
    r = client.get("/compare")
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body and "per_model" in body

    assert client.get("/health").status_code == 200
    assert client.get("/version").json()["version"] == "0.1.0"


def test_api_key_guard(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    cfg = load_config("configs/dev.yaml")
    cfg["api"]["api_key"] = "secret123"
    monkeypatch.setattr(routes, "run_evaluate_in_memory", lambda cfg, split, use_case: _canned_result())

    client = TestClient(create_app(cfg))
    # Data endpoints require the key.
    assert client.get("/compare").status_code == 401
    assert client.get("/compare", headers={"X-API-Key": "secret123"}).status_code == 200
    # Health/version are always exempt.
    assert client.get("/health").status_code == 200
    assert client.get("/version").status_code == 200

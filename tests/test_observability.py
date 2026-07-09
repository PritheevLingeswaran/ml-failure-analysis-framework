import logging

import pytest

from src.api import security, observability


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.utils.config import load_config
    from src.api.app import create_app
    from src.api import routes
    from src.db import base as db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.delenv("MLFA_API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    db.reset_engine()
    routes.reset_cache()
    cfg = load_config("configs/dev.yaml")
    cfg["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    yield TestClient(create_app(cfg))
    db.reset_engine()


def test_healthz_liveness(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "alive"


def test_readyz_checks_dependencies(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"] is True
    assert any(k.startswith("cache:") for k in body["checks"])


def test_metrics_endpoint_exposes_prometheus(client):
    client.get("/healthz")  # generate some traffic
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "http_requests_total" in r.text


def test_request_id_header_generated_and_echoed(client):
    r = client.get("/healthz")
    assert r.headers.get("X-Request-ID")  # generated
    r2 = client.get("/healthz", headers={"X-Request-ID": "trace-abc"})
    assert r2.headers["X-Request-ID"] == "trace-abc"  # propagated


def test_request_id_filter_stamps_records():
    observability.set_request_id("rid-123")
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hi", None, None)
    assert observability.RequestIdFilter().filter(rec) is True
    assert rec.request_id == "rid-123"


def test_json_formatter_emits_request_id():
    import json

    observability.set_request_id("rid-json")
    rec = logging.LogRecord("lg", logging.WARNING, __file__, 1, "boom %s", ("x",), None)
    rec.request_id = "rid-json"
    parsed = json.loads(observability.JsonFormatter().format(rec))
    assert parsed["request_id"] == "rid-json"
    assert parsed["level"] == "WARNING" and parsed["msg"] == "boom x"


def test_redis_limiter_fails_open_to_inmemory():
    # Nothing is listening on this port -> every Redis op fails, limiter must
    # fall back to the in-memory counter (which still enforces per process).
    lim = security.RedisRateLimiter("redis://127.0.0.1:6399", limit=2, window_sec=60)
    results = [lim.allow("k") for _ in range(4)]
    assert results.count(True) <= 2  # fallback still enforces the limit
    assert lim.ping() is False

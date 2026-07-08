import copy

import pytest

from src.utils.config import load_config
from src.api import security, routes


@pytest.fixture
def base_cfg():
    return load_config("configs/dev.yaml")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ["APP_ENV", "MLFA_API_KEY", "CORS_ORIGINS", "RATE_LIMIT_DEFAULT", "MAX_BODY_BYTES"]:
        monkeypatch.delenv(var, raising=False)
    routes._CACHE.clear()
    routes._KEY_LOCKS.clear()


def test_prod_without_key_fails_fast(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["app"]["env"] = "prod"
    cfg["api"]["api_key"] = None
    with pytest.raises(RuntimeError, match="requires authentication"):
        security.validate_security_config(cfg)


def test_prod_with_key_is_allowed(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["app"]["env"] = "prod"
    cfg["api"]["api_key"] = "k"
    security.validate_security_config(cfg)  # no raise


def test_local_env_needs_no_key(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["app"]["env"] = "dev"
    cfg["api"]["api_key"] = None
    security.validate_security_config(cfg)  # no raise


def test_env_var_forces_auth(base_cfg, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    cfg = copy.deepcopy(base_cfg)
    cfg["api"]["api_key"] = None
    with pytest.raises(RuntimeError):
        security.validate_security_config(cfg)


def test_cors_origins_never_wildcard_by_default(base_cfg):
    origins = security.resolve_cors_origins(base_cfg)
    assert "*" not in origins


def test_cors_origins_from_env(base_cfg, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
    assert security.resolve_cors_origins(base_cfg) == ["https://a.com", "https://b.com"]


def test_rate_limit_returns_429(base_cfg, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "2/minute")
    monkeypatch.setattr(routes, "run_evaluate_in_memory", lambda cfg, split, use_case: {
        "comparison": {"ranking": [], "winner": "rf"}, "per_model": {},
    })
    client = TestClient(create_app(base_cfg))
    codes = [client.get("/compare").status_code for _ in range(4)]
    assert 429 in codes  # limit is 2/min, 4 requests must trip it
    assert client.get("/health").status_code == 200  # exempt


def test_oversized_body_rejected(base_cfg, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api.app import create_app

    monkeypatch.setenv("MAX_BODY_BYTES", "50")
    client = TestClient(create_app(base_cfg))
    r = client.post("/evaluate", content=b"x" * 500)
    assert r.status_code == 413

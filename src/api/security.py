"""Security configuration: environment-aware auth, CORS, and rate limiting.

Resolution order for every setting is env var first, then YAML config, then a
safe default — so the same image can run locally (open) and in production
(locked down) purely through environment variables.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from starlette.requests import Request

logger = logging.getLogger(__name__)

# Environments permitted to run WITHOUT an API key. Anything else (prod,
# staging, production, ...) must have one or startup fails.
_LOCAL_ENVS = {"dev", "local", "test", "ci"}


def resolve_env(cfg: Dict[str, Any]) -> str:
    return (os.environ.get("APP_ENV") or cfg.get("app", {}).get("env") or "dev").strip().lower()


def resolve_api_key(cfg: Dict[str, Any]) -> Optional[str]:
    key = os.environ.get("MLFA_API_KEY") or cfg.get("api", {}).get("api_key")
    return key if key else None


def auth_required(env: str) -> bool:
    """Non-local environments require authentication."""
    return env not in _LOCAL_ENVS


def validate_security_config(cfg: Dict[str, Any]) -> None:
    """Fail fast at startup if a non-local environment has no API key.

    This is the difference between "auth is optional" and "auth is mandatory in
    production": a misconfigured prod deploy refuses to start rather than coming
    up wide open.
    """
    env = resolve_env(cfg)
    key = resolve_api_key(cfg)
    if auth_required(env) and not key:
        raise RuntimeError(
            f"APP_ENV='{env}' is a non-local environment and requires authentication, "
            "but no API key is configured. Set the MLFA_API_KEY environment variable "
            "(or api.api_key in config) before starting the service."
        )
    if auth_required(env):
        logger.info("Auth REQUIRED for env=%s (API key configured).", env)
    else:
        logger.info("Auth optional for local env=%s.", env)


def resolve_cors_origins(cfg: Dict[str, Any]) -> List[str]:
    """Explicit allow-list only — never a wildcard by default.

    CORS_ORIGINS (comma-separated) overrides config. Empty means 'no
    cross-origin browser access', which is the safe production default when the
    UI is served same-origin.
    """
    env_val = os.environ.get("CORS_ORIGINS")
    if env_val is not None:
        return [o.strip() for o in env_val.split(",") if o.strip()]
    return cfg.get("api", {}).get("cors_origins", ["http://localhost:5173", "http://127.0.0.1:5173"])


def rate_limit_key(request: Request) -> str:
    """Throttle per API key when present, otherwise per client IP."""
    if request.headers.get("x-api-key"):
        return "key:" + request.headers["x-api-key"]
    client = request.client
    return "ip:" + (client.host if client else "unknown")


_WINDOW_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "day": 86400}


def parse_rate(spec: str) -> Tuple[int, int]:
    """'120/minute' -> (120, 60). Raises ValueError on malformed input."""
    count_s, _, unit_s = spec.strip().partition("/")
    count = int(count_s)
    unit = unit_s.strip().rstrip("s").lower()
    if unit not in _WINDOW_SECONDS or count <= 0:
        raise ValueError(f"Invalid rate limit spec: {spec!r}")
    return count, _WINDOW_SECONDS[unit]


class FixedWindowRateLimiter:
    """Thread-safe in-memory fixed-window limiter (single process only)."""

    backend = "memory"

    def __init__(self, limit: int, window_sec: int):
        self.limit = limit
        self.window = window_sec
        self._hits: Dict[str, Tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        window_start = now - (now % self.window)
        with self._lock:
            ws, count = self._hits.get(key, (window_start, 0))
            if ws != window_start:
                ws, count = window_start, 0
            count += 1
            self._hits[key] = (ws, count)
            return count <= self.limit

    def ping(self) -> bool:
        return True


class RedisRateLimiter:
    """Distributed fixed-window limiter backed by Redis.

    Per {key, window} it does an atomic INCR (and sets EXPIRE on first hit), so
    the limit is enforced GLOBALLY across every worker/replica — unlike the
    in-memory limiter which is per process. Fails OPEN: if Redis is unreachable
    it degrades to a per-process in-memory count rather than rejecting traffic,
    because the limiter must never be the thing that takes the API down.
    """

    backend = "redis"

    def __init__(self, url: str, limit: int, window_sec: int, prefix: str = "mlfa:rl:"):
        import redis  # local import; only needed when Redis is configured

        self._client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        self.limit = limit
        self.window = window_sec
        self._prefix = prefix
        self._fallback = FixedWindowRateLimiter(limit, window_sec)

    def allow(self, key: str) -> bool:
        now = int(time.time())
        window_start = now - (now % self.window)
        redis_key = f"{self._prefix}{key}:{window_start}"
        try:
            pipe = self._client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, self.window)
            count, _ = pipe.execute()
            return int(count) <= self.limit
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis rate-limit failed, failing open to in-memory: %s", e)
            return self._fallback.allow(key)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False


def build_limiter(cfg: Dict[str, Any]):
    """Per-key/IP limiter. Redis-backed when REDIS_URL is set (shared across
    workers), else in-memory. Tune with RATE_LIMIT_DEFAULT (e.g. '120/minute')."""
    spec = os.environ.get("RATE_LIMIT_DEFAULT") or cfg.get("api", {}).get("rate_limit_default", "120/minute")
    limit, window = parse_rate(spec)
    url = os.environ.get("REDIS_URL") or cfg.get("api", {}).get("redis_url")
    if url:
        try:
            lim = RedisRateLimiter(url, limit, window)
            if lim.ping():
                logger.info("Rate limiter: redis (shared across workers)")
                return lim
            logger.warning("REDIS_URL set but Redis unreachable; rate limiter falling back to in-memory.")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not init Redis rate limiter (%s); using in-memory.", e)
    logger.info("Rate limiter: in-memory (per process)")
    return FixedWindowRateLimiter(limit, window)


def max_body_bytes(cfg: Dict[str, Any]) -> int:
    """Reject request bodies larger than this (defense against memory-abuse)."""
    return int(os.environ.get("MAX_BODY_BYTES") or cfg.get("api", {}).get("max_body_bytes", 1_048_576))

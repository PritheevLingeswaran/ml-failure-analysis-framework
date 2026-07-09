"""Pluggable evaluation cache.

In-memory by default (per-process, zero infra for local dev); Redis when
REDIS_URL is set, so multiple workers/replicas share one cache instead of each
recomputing the same evaluation.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Cache:
    def get(self, key: str) -> Optional[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def ping(self) -> bool:  # pragma: no cover
        return True

    @property
    def backend(self) -> str:  # pragma: no cover
        return "base"


class InMemoryCache(Cache):
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            hit = self._data.get(key)
            if hit and hit["expires"] > time.time():
                return hit["value"]
            if hit:
                self._data.pop(key, None)
        return None

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        with self._lock:
            self._data[key] = {"value": value, "expires": time.time() + ttl}

    @property
    def backend(self) -> str:
        return "memory"


class RedisCache(Cache):
    """JSON-serialized values in Redis. Fails soft: a Redis error degrades to a
    cache miss/no-op rather than taking down the request."""

    def __init__(self, url: str, prefix: str = "mlfa:eval:") -> None:
        import redis  # local import so the dependency is only needed when used

        self._client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        self._prefix = prefix

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._client.get(self._prefix + key)
            return json.loads(raw) if raw else None
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis GET failed, treating as miss: %s", e)
            return None

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        try:
            self._client.set(self._prefix + key, json.dumps(value), ex=ttl)
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis SET failed, skipping cache write: %s", e)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False

    @property
    def backend(self) -> str:
        return "redis"


def build_cache(cfg: Dict[str, Any]) -> Cache:
    url = os.environ.get("REDIS_URL") or cfg.get("api", {}).get("redis_url")
    if url:
        try:
            cache = RedisCache(url)
            if cache.ping():
                logger.info("Evaluation cache: redis (%s)", url)
                return cache
            logger.warning("REDIS_URL set but Redis unreachable; falling back to in-memory cache.")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not init Redis (%s); falling back to in-memory cache.", e)
    logger.info("Evaluation cache: in-memory")
    return InMemoryCache()

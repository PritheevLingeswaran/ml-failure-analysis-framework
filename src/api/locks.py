"""Distributed run lock.

Two concurrent `/runs` on the same dataset resolve to the same run-id and would
race on the shared predictions directory (each writes the same model/prediction
files). This serializes them: same key -> one runs while the other waits;
different datasets -> different keys -> still parallel.

Redis-backed when REDIS_URL is set (correct across workers/replicas); otherwise a
per-process lock (correct for a single-worker deployment). If a holder crashes,
the Redis lock auto-expires after its lease so runs don't wedge forever.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class LockBusy(Exception):
    """Could not acquire the lock within the blocking timeout."""


class LocalLockManager:
    backend = "local"

    def __init__(self) -> None:
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _get(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    @contextmanager
    def lock(self, key: str, blocking_timeout: float = 120.0, lease: float = 1800.0) -> Iterator[None]:
        lock = self._get(key)
        if not lock.acquire(timeout=blocking_timeout):
            raise LockBusy(key)
        try:
            yield
        finally:
            lock.release()


class RedisLockManager:
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis  # local import; only needed when Redis is configured

        self._client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False

    @contextmanager
    def lock(self, key: str, blocking_timeout: float = 120.0, lease: float = 1800.0) -> Iterator[None]:
        # `lease` (timeout) auto-releases if the holder dies; `blocking_timeout`
        # bounds how long a waiter blocks before giving up (-> LockBusy).
        rlock = self._client.lock(f"mlfa:runlock:{key}", timeout=lease, blocking_timeout=blocking_timeout)
        if not rlock.acquire():
            raise LockBusy(key)
        try:
            yield
        finally:
            try:
                rlock.release()
            except Exception:  # noqa: BLE001
                pass  # lease may already have expired


def build_lock_manager(cfg: Dict[str, Any]):
    url = os.environ.get("REDIS_URL") or cfg.get("api", {}).get("redis_url")
    if url:
        try:
            mgr = RedisLockManager(url)
            if mgr.ping():
                logger.info("Run lock: redis (cross-worker)")
                return mgr
            logger.warning("REDIS_URL set but Redis unreachable; run lock falling back to per-process.")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not init Redis run lock (%s); using per-process lock.", e)
    logger.info("Run lock: per-process")
    return LocalLockManager()


_lock_manager: Optional[Any] = None
_lm_guard = threading.Lock()


def get_lock_manager(cfg: Dict[str, Any]):
    global _lock_manager
    if _lock_manager is None:
        with _lm_guard:
            if _lock_manager is None:
                _lock_manager = build_lock_manager(cfg)
    return _lock_manager


def reset_lock_manager() -> None:
    """Test hook."""
    global _lock_manager
    with _lm_guard:
        _lock_manager = None

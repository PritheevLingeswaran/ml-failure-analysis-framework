"""Database engine/session management.

Pluggable by URL: defaults to a local SQLite file (zero infra for dev/demo),
uses Postgres when DATABASE_URL points at one. The same models and migrations
run against both.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DEFAULT_URL = "sqlite:///data/app.db"


class Base(DeclarativeBase):
    pass


def database_url(cfg: Optional[Dict[str, Any]] = None) -> str:
    return os.environ.get("DATABASE_URL") or (cfg or {}).get("db", {}).get("url") or DEFAULT_URL


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine(cfg: Optional[Dict[str, Any]] = None) -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = database_url(cfg)
        connect_args: Dict[str, Any] = {}
        if url.startswith("sqlite"):
            # Allow the file's parent dir to exist and share the connection across threads.
            path = url.replace("sqlite:///", "", 1)
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            connect_args["check_same_thread"] = False
        _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
        logger.info("Database engine initialized: %s", url.split("@")[-1])
    return _engine


def reset_engine() -> None:
    """Test hook: dispose and rebuild on next use."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope(cfg: Optional[Dict[str, Any]] = None) -> Iterator[Session]:
    get_engine(cfg)
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(cfg: Optional[Dict[str, Any]] = None) -> None:
    """Convenience for local/dev and tests. Production schema is managed by Alembic."""
    from src.db import models  # noqa: F401  (register mappers)

    Base.metadata.create_all(get_engine(cfg))


def ping(cfg: Optional[Dict[str, Any]] = None) -> bool:
    try:
        with get_engine(cfg).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("DB ping failed: %s", e)
        return False

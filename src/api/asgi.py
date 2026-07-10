"""ASGI entrypoint for multi-worker servers.

`uvicorn --workers N` needs an import string (it can't fork an already-built app
object), so this module builds a module-level `app`. Each worker imports it and
gets its own process-local limiter/cache — which is exactly why those are backed
by Redis: the workers share one cache and one rate-limit counter.

Migrations and any data bootstrap run once in the container entrypoint BEFORE
the workers start, so importing here does not retrigger them.
"""
from __future__ import annotations

import os

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.api.app import create_app

_cfg = load_config(os.environ.get("CONFIG_PATH", "configs/prod.yaml"))
setup_logging(_cfg)
app = create_app(_cfg)

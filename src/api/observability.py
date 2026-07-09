"""Request-scoped context + structured logging.

The request ID lives in a contextvar, so any code executing within a request —
the handler, the rate limiter, the cache layer — can read it WITHOUT it being
threaded through every function signature. A logging filter stamps it onto every
log record, which is what makes "grep one request_id to see the whole request"
actually true end to end.
"""
from __future__ import annotations

import contextvars
import json
import logging
import uuid
from datetime import datetime, timezone

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Attaches the current request_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def install_request_logging(json_logs: bool) -> None:
    """Add the request-id filter to every root handler (so propagated records
    from child loggers get tagged), and switch to JSON output when requested."""
    root = logging.getLogger()
    filt = RequestIdFilter()
    for handler in root.handlers:
        # Avoid stacking duplicate filters on repeated setup.
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(filt)
        if json_logs:
            handler.setFormatter(JsonFormatter())

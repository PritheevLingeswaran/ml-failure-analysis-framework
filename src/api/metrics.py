"""Prometheus metrics.

Kept in one module so any layer (middleware, cache, rate limiter) can record
without importing each other. Path labels use the matched *route template*
(e.g. /runs/{run_id}) not the raw URL, to keep cardinality bounded.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"])
CACHE_EVENTS = Counter("eval_cache_events_total", "Evaluation cache events", ["event"])  # hit|miss
EVAL_COMPUTATIONS = Counter("eval_computations_total", "Evaluations actually computed (cache misses that ran the pipeline)")
RATE_LIMIT_REJECTIONS = Counter("rate_limit_rejections_total", "Requests rejected by the rate limiter")


def record_request(method: str, path: str, status: int, duration_sec: float) -> None:
    REQUESTS.labels(method, path, str(status)).inc()
    LATENCY.labels(method, path).observe(duration_sec)


def cache_hit() -> None:
    CACHE_EVENTS.labels("hit").inc()


def cache_miss() -> None:
    CACHE_EVENTS.labels("miss").inc()


def eval_computed() -> None:
    EVAL_COMPUTATIONS.inc()


def rate_limited() -> None:
    RATE_LIMIT_REJECTIONS.inc()


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

from __future__ import annotations
import logging
import time
from typing import Any, Dict
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.api.persistence_routes import router as persistence_router
from src.api import security, metrics, observability, routes
from src.db import base as db

logger = logging.getLogger(__name__)

# Endpoints that never require an API key or rate limiting (probes, docs, schema).
_EXEMPT = {"/health", "/healthz", "/readyz", "/version", "/metrics", "/docs", "/redoc", "/openapi.json"}


def create_app(cfg: Dict[str, Any]) -> FastAPI:
    # Fail fast: a non-local environment with no API key must not start.
    security.validate_security_config(cfg)

    app = FastAPI(
        title="ml-failure-analysis-framework",
        version="0.1.0",
        description="Internal model evaluation, slicing, error analysis, and decision-theoretic recommendations.",
    )
    app.state.cfg = cfg

    env = security.resolve_env(cfg)
    api_key = security.resolve_api_key(cfg)
    max_body = security.max_body_bytes(cfg)
    limiter = security.build_limiter(cfg)
    app.state.limiter = limiter

    # --- CORS: explicit allow-list, never wildcard ---
    origins = security.resolve_cors_origins(cfg)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    logger.info("CORS allow-list: %s", origins or "(none)")

    # --- Single gate: request-id -> body-size -> rate-limit -> auth -> metrics.
    # The request id is set FIRST so every downstream log (rate limiter, cache,
    # handler) carries it, making a request traceable end to end.
    @app.middleware("http")
    async def gate(request: Request, call_next):
        rid = request.headers.get("x-request-id") or observability.new_request_id()
        observability.set_request_id(rid)
        start = time.perf_counter()
        path = request.url.path
        exempt = request.method == "OPTIONS" or path in _EXEMPT

        async def finalize(response: Response) -> Response:
            response.headers["X-Request-ID"] = rid
            route = request.scope.get("route")
            label = getattr(route, "path", None) or path
            metrics.record_request(request.method, label, response.status_code, time.perf_counter() - start)
            return response

        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > max_body:
                    return await finalize(JSONResponse(status_code=413, content={"error": "payload_too_large", "detail": f"Body exceeds {max_body} bytes."}))
            except ValueError:
                return await finalize(JSONResponse(status_code=400, content={"error": "bad_request", "detail": "Invalid Content-Length."}))

        if not exempt:
            if not limiter.allow(security.rate_limit_key(request)):
                metrics.rate_limited()
                logger.warning("rate limited: %s", security.rate_limit_key(request))
                return await finalize(JSONResponse(status_code=429, content={"error": "rate_limited", "detail": f"Rate limit exceeded ({limiter.limit}/{limiter.window}s)."}))
            if api_key and request.headers.get("x-api-key") != api_key:
                return await finalize(JSONResponse(status_code=401, content={"error": "unauthorized", "detail": "Missing or invalid X-API-Key header."}))

        return await finalize(await call_next(request))

    # --- Probes & metrics ---
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        # Liveness: is the process up? Always 200 while the app runs.
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz() -> Response:
        # Readiness: can this instance serve traffic? Checks dependencies.
        checks: Dict[str, bool] = {"database": db.ping(cfg)}
        cache = routes._get_cache(cfg)
        checks[f"cache:{cache.backend}"] = cache.ping()
        if getattr(limiter, "backend", "memory") == "redis":
            checks["rate_limiter:redis"] = limiter.ping()
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "not_ready", "checks": checks})

    @app.get("/version")
    def version() -> Dict[str, str]:
        return {"name": "ml-failure-analysis-framework", "version": "0.1.0"}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled API error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": str(exc), "path": str(request.url.path)},
            headers={"X-Request-ID": observability.get_request_id()},
        )

    app.include_router(api_router)
    app.include_router(persistence_router)

    # Schema: in local envs auto-create tables for zero-setup demoing; in
    # non-local envs the schema is owned by Alembic migrations (run at deploy).
    if not security.auth_required(env):
        try:
            db.create_all(cfg)
        except Exception as e:  # noqa: BLE001
            logger.warning("DB auto-create skipped: %s", e)

    logger.info(
        "App ready: env=%s auth=%s rate=%s/%ss(%s) cache=%s",
        env, "on" if api_key else "off", limiter.limit, limiter.window,
        getattr(limiter, "backend", "memory"), routes._get_cache(cfg).backend,
    )
    return app

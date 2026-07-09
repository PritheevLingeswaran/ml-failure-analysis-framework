from __future__ import annotations
import logging
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.api.persistence_routes import router as persistence_router
from src.api import security
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

    # --- Single gate: body-size -> rate-limit -> auth (runs for every request) ---
    @app.middleware("http")
    async def gate(request: Request, call_next):
        path = request.url.path
        exempt = request.method == "OPTIONS" or path in _EXEMPT

        # 1) Reject oversized bodies before reading them.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > max_body:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "payload_too_large", "detail": f"Body exceeds {max_body} bytes."},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "Invalid Content-Length."})

        if not exempt:
            # 2) Rate limit per API key / IP.
            if not limiter.allow(security.rate_limit_key(request)):
                return JSONResponse(
                    status_code=429,
                    content={"error": "rate_limited", "detail": f"Rate limit exceeded ({limiter.limit}/{limiter.window}s)."},
                )
            # 3) API-key auth (enforced whenever a key is configured).
            if api_key and request.headers.get("x-api-key") != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "detail": "Missing or invalid X-API-Key header."},
                )

        return await call_next(request)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> Dict[str, str]:
        return {"name": "ml-failure-analysis-framework", "version": "0.1.0"}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled API error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": str(exc), "path": str(request.url.path)},
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

    logger.info("App created for env=%s (auth %s, rate=%s/%ss).", env, "on" if api_key else "off", limiter.limit, limiter.window)
    return app

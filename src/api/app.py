from __future__ import annotations
import logging
import os
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Endpoints that never require an API key (health checks, docs, schema).
_AUTH_EXEMPT = {"/health", "/version", "/docs", "/redoc", "/openapi.json"}


def create_app(cfg: Dict[str, Any]) -> FastAPI:
    app = FastAPI(
        title="ml-failure-analysis-framework",
        version="0.1.0",
        description="Internal model evaluation, slicing, error analysis, and decision-theoretic recommendations.",
    )
    app.state.cfg = cfg

    api_cfg = cfg.get("api", {})

    # CORS — configurable allowed origins. Defaults to the local dev frontend.
    origins = api_cfg.get("cors_origins", ["http://localhost:5173", "http://127.0.0.1:5173"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Optional API-key auth. Enabled only when a key is configured (via
    # api.api_key or the MLFA_API_KEY env var); otherwise the API is open for
    # local/dev use. Health, version and docs are always exempt.
    api_key = os.environ.get("MLFA_API_KEY") or api_cfg.get("api_key")

    @app.middleware("http")
    async def api_key_guard(request: Request, call_next):
        if api_key:
            path = request.url.path
            if request.method != "OPTIONS" and path not in _AUTH_EXEMPT:
                if request.headers.get("x-api-key") != api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "unauthorized", "detail": "Missing or invalid X-API-Key header."},
                    )
        return await call_next(request)

    if api_key:
        logger.info("API key authentication is ENABLED.")

    app.include_router(api_router)

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
            content={
                "error": "internal_server_error",
                "detail": str(exc),
                "path": str(request.url.path),
            },
        )

    return app

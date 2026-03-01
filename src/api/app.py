from __future__ import annotations
import logging
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.utils.config import load_config

logger = logging.getLogger(__name__)

def create_app(cfg: Dict[str, Any]) -> FastAPI:
    app = FastAPI(
        title="ml-failure-analysis-framework",
        version="0.1.0",
        description="Internal model evaluation, slicing, error analysis, and decision-theoretic recommendations.",
    )
    app.state.cfg = cfg
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

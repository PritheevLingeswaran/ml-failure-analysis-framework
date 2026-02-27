from __future__ import annotations
import logging
from typing import Any, Dict
from fastapi import FastAPI

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
    return app

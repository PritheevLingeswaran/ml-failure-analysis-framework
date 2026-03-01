from __future__ import annotations
import copy
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, Request

from src.schemas.api import EvaluateRequest, CompareResponse, SliceMetricsResponse, ErrorsResponse, RecommendResponse
from src.decision_engine.costs import load_costs
from src.evaluation_engine.predictions import build_run_id

from evaluation.evaluate import run_evaluate_in_memory

logger = logging.getLogger(__name__)
router = APIRouter()
_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_CACHE: Dict[str, Dict[str, Any]] = {}
_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

def _api_cfg(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Use a request-local config for API evaluation runs."""
    cfg = copy.deepcopy(base_cfg)
    # Plot generation is expensive and can fail in constrained runtime envs.
    # API responses only need JSON payloads.
    cfg.setdefault("visualization", {})
    cfg["visualization"]["enabled"] = False
    return cfg

def _resolve_use_case(cfg: Dict[str, Any], requested: str) -> str:
    # Keep API-friendly "default" alias stable even when cost config use-case names change.
    if requested in ("", "default"):
        return cfg["decision"]["default_use_case"]

    use_cases = load_costs(cfg).get("use_cases", {})
    if requested not in use_cases:
        available = sorted(use_cases.keys())
        raise HTTPException(
            status_code=422,
            detail=f"Unknown use_case='{requested}'. Available use_cases: {available}",
        )
    return requested


def _cache_key(cfg: Dict[str, Any], use_case: str, split: str = "test") -> str:
    return json.dumps(
        {"run_id": build_run_id(cfg), "use_case": use_case, "split": split},
        sort_keys=True,
    )


def _evaluate_cached(cfg: Dict[str, Any], use_case: str, split: str = "test") -> Dict[str, Any]:
    ttl = int(cfg.get("api", {}).get("cache_ttl_sec", 180))
    key = _cache_key(cfg, use_case, split=split)
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and (now - float(hit["ts"])) <= ttl:
            return hit["result"]

    result = run_evaluate_in_memory(cfg, split=split, use_case=use_case)
    with _LOCK:
        _CACHE[key] = {"ts": now, "result": result}
    return result

@router.post("/evaluate", response_model=CompareResponse)
def evaluate(req: EvaluateRequest, request: Request):
    cfg = _api_cfg(request.app.state.cfg)
    use_case = _resolve_use_case(cfg, req.use_case)
    result = _evaluate_cached(cfg, split="test", use_case=use_case)
    return CompareResponse(summary=result["comparison"], per_model=result["per_model"])


@router.post("/evaluate/async")
def evaluate_async(req: EvaluateRequest, request: Request):
    cfg = _api_cfg(request.app.state.cfg)
    use_case = _resolve_use_case(cfg, req.use_case)
    job_id = str(uuid.uuid4())
    with _LOCK:
        _JOBS[job_id] = {"status": "queued", "created_at": time.time(), "use_case": use_case}

    def _run() -> None:
        with _LOCK:
            _JOBS[job_id]["status"] = "running"
        try:
            result = _evaluate_cached(cfg, split="test", use_case=use_case)
            with _LOCK:
                _JOBS[job_id]["status"] = "completed"
                _JOBS[job_id]["result"] = {
                    "summary": result["comparison"],
                    "per_model": result["per_model"],
                }
        except Exception as e:
            with _LOCK:
                _JOBS[job_id]["status"] = "failed"
                _JOBS[job_id]["error"] = str(e)

    _EXECUTOR.submit(_run)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job_id='{job_id}' not found")
        return job

@router.get("/compare", response_model=CompareResponse)
def compare(request: Request):
    cfg = _api_cfg(request.app.state.cfg)
    result = _evaluate_cached(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    return CompareResponse(summary=result["comparison"], per_model=result["per_model"])

@router.get("/slices", response_model=SliceMetricsResponse)
def slices(request: Request):
    cfg = _api_cfg(request.app.state.cfg)
    result = _evaluate_cached(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    # Flattened slice table
    slices = result["slice_table"]
    return SliceMetricsResponse(slices=slices)

@router.get("/errors", response_model=ErrorsResponse)
def errors(request: Request):
    cfg = _api_cfg(request.app.state.cfg)
    result = _evaluate_cached(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    return ErrorsResponse(
        top_false_positives=result["errors"]["top_false_positives"],
        top_false_negatives=result["errors"]["top_false_negatives"],
        clusters=result["errors"]["clusters"],
    )

@router.get("/recommend", response_model=RecommendResponse)
def recommend(request: Request):
    cfg = _api_cfg(request.app.state.cfg)
    result = _evaluate_cached(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    d = result["decision"]
    return RecommendResponse(
        recommended_model=d["recommended_model"],
        recommended_threshold=d["recommended_threshold"],
        rationale=d["rationale"],
        per_slice_recommendations=d.get("per_slice_recommendations"),
    )


@router.get("/diagnostics")
def diagnostics(request: Request):
    cfg = _api_cfg(request.app.state.cfg)
    result = _evaluate_cached(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    return {
        "run_id": result.get("run_id"),
        "split": result.get("split"),
        "drift": result.get("drift", {}),
        "cost_sensitivity": result.get("cost_sensitivity", {}),
    }

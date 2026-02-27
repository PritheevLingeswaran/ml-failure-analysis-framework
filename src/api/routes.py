from __future__ import annotations
import logging
from typing import Any, Dict
from fastapi import APIRouter, Request

from src.schemas.api import EvaluateRequest, CompareResponse, SliceMetricsResponse, ErrorsResponse, RecommendResponse

from evaluation.evaluate import run_evaluate_in_memory

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/evaluate", response_model=CompareResponse)
def evaluate(req: EvaluateRequest, request: Request):
    cfg = request.app.state.cfg
    result = run_evaluate_in_memory(cfg, split="test", use_case=req.use_case)
    return CompareResponse(summary=result["comparison"], per_model=result["per_model"])

@router.get("/compare", response_model=CompareResponse)
def compare(request: Request):
    cfg = request.app.state.cfg
    result = run_evaluate_in_memory(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    return CompareResponse(summary=result["comparison"], per_model=result["per_model"])

@router.get("/slices", response_model=SliceMetricsResponse)
def slices(request: Request):
    cfg = request.app.state.cfg
    result = run_evaluate_in_memory(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    # Flattened slice table
    slices = result["slice_table"]
    return SliceMetricsResponse(slices=slices)

@router.get("/errors", response_model=ErrorsResponse)
def errors(request: Request):
    cfg = request.app.state.cfg
    result = run_evaluate_in_memory(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    return ErrorsResponse(
        top_false_positives=result["errors"]["top_false_positives"],
        top_false_negatives=result["errors"]["top_false_negatives"],
        clusters=result["errors"]["clusters"],
    )

@router.get("/recommend", response_model=RecommendResponse)
def recommend(request: Request):
    cfg = request.app.state.cfg
    result = run_evaluate_in_memory(cfg, split="test", use_case=cfg["decision"]["default_use_case"])
    d = result["decision"]
    return RecommendResponse(
        recommended_model=d["recommended_model"],
        recommended_threshold=d["recommended_threshold"],
        rationale=d["rationale"],
        per_slice_recommendations=d.get("per_slice_recommendations"),
    )

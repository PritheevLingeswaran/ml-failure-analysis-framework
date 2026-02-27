from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class EvaluateRequest(BaseModel):
    use_case: str = Field(default="default", description="Decision cost use-case")
    per_slice: bool = Field(default=True, description="Compute slice metrics")
    per_slice_threshold_opt: bool = Field(default=True, description="Optimize threshold per slice")

class CompareResponse(BaseModel):
    summary: Dict[str, Any]
    per_model: Dict[str, Any]

class SliceMetricsResponse(BaseModel):
    slices: List[Dict[str, Any]]

class ErrorsResponse(BaseModel):
    top_false_positives: Dict[str, List[Dict[str, Any]]]
    top_false_negatives: Dict[str, List[Dict[str, Any]]]
    clusters: Dict[str, Any]

class RecommendResponse(BaseModel):
    recommended_model: str
    recommended_threshold: float
    rationale: Dict[str, Any]
    per_slice_recommendations: Optional[List[Dict[str, Any]]] = None

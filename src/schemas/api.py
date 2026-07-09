from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class EvaluateRequest(BaseModel):
    # use_case is echoed into config lookups; constrain it to a safe identifier
    # shape and length as defense-in-depth (the value is also checked against the
    # known use-case list in the route).
    use_case: str = Field(
        default="default",
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Decision cost use-case",
    )
    per_slice: bool = Field(default=True, description="Compute slice metrics")
    per_slice_threshold_opt: bool = Field(default=True, description="Optimize threshold per slice")

    model_config = {"extra": "forbid"}  # reject unknown fields

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


class QualityResponse(BaseModel):
    quality: Dict[str, Any]

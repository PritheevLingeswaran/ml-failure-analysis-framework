"""Dataset upload + analysis-run endpoints backed by the database.

These turn the platform from "evaluate the bundled data" into "upload your own
CSV, run a cost-aware analysis on it, and browse the run history."
"""
from __future__ import annotations

import copy
import io
import logging
import os
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.db import base as db
from src.db.models import AnalysisRun, Dataset

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_CONTENT_TYPES = {
    "text/csv", "application/csv", "application/vnd.ms-excel",
    "application/octet-stream", "",
}
_UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))


def _max_upload_bytes() -> int:
    return int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB


def _max_train_rows() -> int:
    return int(os.environ.get("MAX_TRAIN_ROWS", 50_000))


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@router.post("/datasets/upload")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    label_col: str = Form("label"),
    name: Optional[str] = Form(None),
):
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content-type '{file.content_type}'. Upload a CSV.")

    # Read with a hard size cap so a huge upload can't exhaust memory.
    limit = _max_upload_bytes()
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {limit} bytes.")
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV must have at least one feature column and a label column.")
    if label_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Label column '{label_col}' not found. Columns: {list(df.columns)}")
    if df[label_col].nunique(dropna=True) != 2:
        raise HTTPException(status_code=422, detail=f"Label column '{label_col}' must be binary (exactly 2 classes).")

    dataset_id = str(uuid.uuid4())
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = _UPLOAD_DIR / f"{dataset_id}.csv"
    storage_path.write_bytes(raw)

    row = Dataset(
        id=dataset_id,
        name=name or file.filename or dataset_id,
        original_filename=file.filename or "upload.csv",
        storage_path=str(storage_path),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        label_col=label_col,
        columns=[str(c) for c in df.columns],
        size_bytes=len(raw),
    )
    with db.session_scope(request.app.state.cfg) as s:
        s.add(row)
        s.flush()
        out = row.to_dict()
    logger.info("Uploaded dataset %s (%s rows, %s cols)", dataset_id, row.n_rows, row.n_cols)
    return out


@router.get("/datasets")
def list_datasets(request: Request) -> Dict[str, List[Dict[str, Any]]]:
    with db.session_scope(request.app.state.cfg) as s:
        rows = s.query(Dataset).order_by(Dataset.created_at.desc()).limit(200).all()
        return {"datasets": [r.to_dict() for r in rows]}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, request: Request) -> Dict[str, Any]:
    with db.session_scope(request.app.state.cfg) as s:
        row = s.get(Dataset, dataset_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"dataset '{dataset_id}' not found")
        return row.to_dict()


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    dataset_id: Optional[str] = Field(default=None, description="Uploaded dataset to analyze; omit for the bundled data")
    use_case: str = Field(default="default", max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")

    model_config = {"extra": "forbid"}


def _cfg_for_dataset(base_cfg: Dict[str, Any], ds: Dataset) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["paths"]["data_raw"] = ds.storage_path
    cfg.setdefault("data", {}).setdefault("dataset", {})
    cfg["data"]["dataset"]["label_col"] = ds.label_col
    cfg["data"]["dataset"]["id_col"] = cfg["data"]["dataset"].get("id_col", "id")
    cfg["data"]["dataset"]["text_col"] = None
    cfg["data"]["dataset"]["time_col"] = None
    cfg.setdefault("visualization", {})["enabled"] = False
    return cfg


@router.post("/runs")
def create_run(req: RunRequest, request: Request) -> Dict[str, Any]:
    from evaluation.evaluate import run_evaluate_in_memory
    from src.decision_engine.costs import load_costs
    import scripts.train_models as trainer

    base_cfg = request.app.state.cfg
    run_id = str(uuid.uuid4())

    # Resolve dataset + config.
    dataset_id: Optional[str] = req.dataset_id
    if dataset_id:
        with db.session_scope(base_cfg) as s:
            ds = s.get(Dataset, dataset_id)
            if ds is None:
                raise HTTPException(status_code=404, detail=f"dataset '{dataset_id}' not found")
            if ds.n_rows > _max_train_rows():
                raise HTTPException(status_code=422, detail=f"Dataset has {ds.n_rows} rows; max for a run is {_max_train_rows()}.")
            cfg = _cfg_for_dataset(base_cfg, ds)
            needs_training = True
    else:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("visualization", {})["enabled"] = False
        needs_training = False  # bundled data is trained at startup

    # Resolve + validate use_case against the cost config.
    requested = req.use_case
    use_cases = load_costs(cfg).get("use_cases", {})
    if requested in ("", "default"):
        use_case = cfg["decision"]["default_use_case"]
    elif requested in use_cases:
        use_case = requested
    else:
        raise HTTPException(status_code=422, detail=f"Unknown use_case '{requested}'. Available: {sorted(use_cases)}")

    started = perf_counter()
    try:
        if needs_training:
            trainer.run_training(cfg)
        result = run_evaluate_in_memory(cfg, split="test", use_case=use_case)
        decision = result["decision"]
        blr = result["eval_quality"]["business_loss_reduction"]
        summary = {
            "winner": result["comparison"].get("winner"),
            "ranking": result["comparison"].get("ranking"),
            "recommended_model": decision.get("recommended_model"),
            "recommended_threshold": decision.get("recommended_threshold"),
            "business_loss_reduction": blr,
        }
        row = AnalysisRun(
            id=run_id, dataset_id=dataset_id, use_case=use_case, status="completed",
            winner_model=decision.get("recommended_model"),
            recommended_threshold=decision.get("recommended_threshold"),
            reduction_pct=blr.get("reduction_pct"),
            runtime_sec=round(perf_counter() - started, 3),
            summary=summary,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Analysis run %s failed", run_id)
        row = AnalysisRun(
            id=run_id, dataset_id=dataset_id, use_case=use_case, status="failed",
            runtime_sec=round(perf_counter() - started, 3), error=str(e)[:1024],
        )
        with db.session_scope(base_cfg) as s:
            s.add(row)
            s.flush()
            out = row.to_dict()
        raise HTTPException(status_code=500, detail={"run_id": run_id, "error": str(e)})

    with db.session_scope(base_cfg) as s:
        s.add(row)
        s.flush()
        return row.to_dict()


@router.get("/runs")
def list_runs(request: Request) -> Dict[str, List[Dict[str, Any]]]:
    with db.session_scope(request.app.state.cfg) as s:
        rows = s.query(AnalysisRun).order_by(AnalysisRun.created_at.desc()).limit(200).all()
        return {"runs": [r.to_dict() for r in rows]}


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> Dict[str, Any]:
    with db.session_scope(request.app.state.cfg) as s:
        row = s.get(AnalysisRun, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
        return row.to_dict()

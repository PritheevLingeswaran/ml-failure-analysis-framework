from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from src.db.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    name: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(512))
    n_rows: Mapped[int] = mapped_column(Integer)
    n_cols: Mapped[int] = mapped_column(Integer)
    label_col: Mapped[str] = mapped_column(String(128))
    columns: Mapped[List[str]] = mapped_column(JSON)
    size_bytes: Mapped[int] = mapped_column(Integer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "name": self.name,
            "original_filename": self.original_filename,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "label_col": self.label_col,
            "columns": self.columns,
            "size_bytes": self.size_bytes,
        }


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    dataset_id: Mapped[Optional[str]] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    use_case: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    winner_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    recommended_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reduction_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    runtime_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "dataset_id": self.dataset_id,
            "use_case": self.use_case,
            "status": self.status,
            "winner_model": self.winner_model,
            "recommended_threshold": self.recommended_threshold,
            "reduction_pct": self.reduction_pct,
            "runtime_sec": self.runtime_sec,
            "summary": self.summary,
            "error": self.error,
        }


class CostMatrixConfig(Base):
    __tablename__ = "cost_matrix_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    name: Mapped[str] = mapped_column(String(128))
    session_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    costs: Mapped[Dict[str, float]] = mapped_column(JSON)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "name": self.name,
            "session_id": self.session_id,
            "costs": self.costs,
        }

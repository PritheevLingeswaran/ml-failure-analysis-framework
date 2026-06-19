from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict

from src.utils.io import write_json


def write_experiment_record(
    outputs_dir: str | Path,
    run_id: str,
    payload: Dict[str, Any],
) -> Path:
    p = Path(outputs_dir) / "experiments" / f"{run_id}.json"
    record = {
        "recorded_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        **payload,
    }
    write_json(p, record)
    return p


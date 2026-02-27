from __future__ import annotations
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict
from src.utils.config import load_yaml
from src.utils.paths import ensure_dir

def setup_logging(cfg: Dict[str, Any]) -> None:
    """Configure logging.

    Why:
    - Logs are part of the audit trail of evaluation runs.
    - This avoids notebook-style silent failures.
    """
    logging_cfg_path = cfg.get("logging", {}).get("config_path", "configs/logging.yaml")
    if Path(logging_cfg_path).exists():
        logging.config.dictConfig(load_yaml(logging_cfg_path))
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Also log to outputs/logs for traceability
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    logs_dir = outputs_dir / "logs"
    ensure_dir(logs_dir)
    file_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict
import yaml

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge b into a (a is copied)."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load config with base.yaml + override file.

    Why:
    - A production eval platform must be reproducible and auditable.
    - Explicit YAML configs are reviewable artifacts (unlike hidden notebook state).
    """
    config_path = str(config_path)
    override = load_yaml(config_path)
    base_path = override.get("_base", "configs/base.yaml")
    base = load_yaml(base_path)
    cfg = _deep_merge(base, override)

    # Expand env vars in paths
    for k, v in list(cfg.get("paths", {}).items()):
        if isinstance(v, str):
            cfg["paths"][k] = os.path.expandvars(v)

    return cfg

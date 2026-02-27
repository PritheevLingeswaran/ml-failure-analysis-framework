from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

def plot_all(cfg: Dict[str, Any], per_model: Dict[str, Any], comparison: Dict[str, Any], decision: Dict[str, Any]) -> None:
    outdir = Path(cfg["paths"]["outputs_dir"]) / "plots"
    ensure_dir(outdir)

    _plot_cost_curves(cfg, per_model, outdir)
    _plot_confidence_accuracy(cfg, per_model, outdir)
    _plot_slice_heatmap(cfg, per_model, outdir)

def _plot_cost_curves(cfg: Dict[str, Any], per_model: Dict[str, Any], outdir: Path) -> None:
    from src.decision_engine.costs import load_costs, expected_cost_binary
    costs = load_costs(cfg)["use_cases"][cfg["decision"]["default_use_case"]]["binary"]
    grid = _threshold_grid(cfg)

    plt.figure()
    for model_name, payload in per_model.items():
        y_true = payload["overall"]["confusion"]  # not enough; pull from slice? keep it simple by reusing cached best curve computed in scripts
    # Instead, read curves from payload if available; if not, skip
    available = True
    for model_name, payload in per_model.items():
        curve = payload.get("cost_curve")
        if curve is None:
            available = False
            break
    if not available:
        logger.info("Cost curve not found in payload; skipping cost curve plot.")
        return

    plt.figure()
    for model_name, payload in per_model.items():
        curve = payload["cost_curve"]
        plt.plot(curve["thresholds"], curve["expected_costs"], label=model_name)
    plt.xlabel("Threshold")
    plt.ylabel("Expected cost")
    plt.title("Expected cost vs threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cost_vs_threshold.png")
    plt.close()

def _plot_confidence_accuracy(cfg: Dict[str, Any], per_model: Dict[str, Any], outdir: Path) -> None:
    # Use slice bins named "confidence[lo,hi)"
    plt.figure()
    for model_name, payload in per_model.items():
        xs, ys = [], []
        for s in payload["slices"]:
            if s["slice_name"].startswith("confidence["):
                # mid point
                name = s["slice_name"]
                lohi = name[len("confidence["):-1].split(",")
                lo = float(lohi[0]); hi = float(lohi[1])
                mid = (lo + hi) / 2
                xs.append(mid)
                ys.append(s["metrics"]["accuracy"])
        if xs:
            order = np.argsort(xs)
            plt.plot(np.array(xs)[order], np.array(ys)[order], label=model_name)
    plt.xlabel("Confidence bin midpoint")
    plt.ylabel("Accuracy at threshold=0.5")
    plt.title("Confidence vs Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "confidence_vs_accuracy.png")
    plt.close()

def _plot_slice_heatmap(cfg: Dict[str, Any], per_model: Dict[str, Any], outdir: Path) -> None:
    # Heatmap: slices x models with expected cost
    # Limit to top N slices by count for readability
    max_slices = int(cfg.get("visualization", {}).get("max_slices_heatmap", 25))

    # Collect all slice names (exclude overly generic ones)
    slice_names = set()
    for payload in per_model.values():
        for s in payload["slices"]:
            slice_names.add(s["slice_name"])
    slice_names = sorted(slice_names)

    # Compute counts (use first model as reference)
    counts = {}
    first_payload = next(iter(per_model.values()))
    for s in first_payload["slices"]:
        counts[s["slice_name"]] = s["count"]

    slice_names = sorted(slice_names, key=lambda n: counts.get(n, 0), reverse=True)[:max_slices]
    models = list(per_model.keys())

    mat = np.full((len(slice_names), len(models)), np.nan)
    for j, m in enumerate(models):
        m_slices = {s["slice_name"]: s for s in per_model[m]["slices"]}
        for i, sn in enumerate(slice_names):
            s = m_slices.get(sn)
            if s and "decision" in s:
                mat[i, j] = s["decision"]["expected_cost"]

    if len(slice_names) == 0:
        return

    plt.figure(figsize=(max(6, len(models)*1.2), max(6, len(slice_names)*0.3)))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Expected cost (lower is better)")
    plt.xticks(range(len(models)), models, rotation=45, ha="right")
    plt.yticks(range(len(slice_names)), slice_names)
    plt.title("Slice expected cost heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "slice_cost_heatmap.png")
    plt.close()

def _threshold_grid(cfg: Dict[str, Any]) -> np.ndarray:
    g = cfg["evaluation"]["threshold_grid"]
    start, stop, step = float(g["start"]), float(g["stop"]), float(g["step"])
    return np.round(np.arange(start, stop + 1e-12, step), 6)

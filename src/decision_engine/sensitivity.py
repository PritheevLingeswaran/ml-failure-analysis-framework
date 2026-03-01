from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.decision_engine.thresholds import optimize_threshold


def build_cost_scenarios(base_costs: Dict[str, float], fp_mults: List[float], fn_mults: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fm in fp_mults:
        for nm in fn_mults:
            costs = dict(base_costs)
            costs["FP"] = float(base_costs["FP"]) * float(fm)
            costs["FN"] = float(base_costs["FN"]) * float(nm)
            out.append(
                {
                    "name": f"fp_x{fm:g}__fn_x{nm:g}",
                    "fp_multiplier": float(fm),
                    "fn_multiplier": float(nm),
                    "costs": costs,
                }
            )
    return out


def run_cost_sensitivity(
    predictions_by_model: Dict[str, Dict[str, np.ndarray]],
    grid: np.ndarray,
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scenario_rows: List[Dict[str, Any]] = []
    winner_counts: Dict[str, int] = {}

    for s in scenarios:
        rows = []
        for model_name, payload in predictions_by_model.items():
            best = optimize_threshold(payload["y_true"], payload["y_score"], grid, s["costs"])
            rows.append(
                {
                    "model": model_name,
                    "best_threshold": float(best["threshold"]),
                    "expected_cost": float(best["expected_cost"]),
                }
            )
        ranked = sorted(rows, key=lambda r: r["expected_cost"])
        winner = ranked[0]["model"] if ranked else None
        if winner:
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
        scenario_rows.append({"scenario": s, "ranking": ranked, "winner": winner})

    total = max(1, len(scenario_rows))
    robustness = [
        {"model": m, "win_count": c, "win_rate": float(c / total)}
        for m, c in sorted(winner_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {"scenarios": scenario_rows, "winner_robustness": robustness}


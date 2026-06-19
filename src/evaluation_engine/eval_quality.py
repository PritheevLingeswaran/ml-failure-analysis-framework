from __future__ import annotations

from typing import Any, Dict, List

from src.decision_engine.costs import expected_cost_binary, load_costs

DEFAULT_COST_NORMALIZATION_EPS = 1e-9


def compute_eval_quality(
    cfg: Dict[str, Any],
    *,
    run_id: str,
    split: str,
    use_case: str,
    split_count: int,
    total_count: int | None,
    per_model: Dict[str, Dict[str, Any]],
    model_scores: Dict[str, Dict[str, Any]],
    decision: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    recommended_model = decision.get("recommended_model")
    if recommended_model is None or recommended_model not in per_model:
        raise ValueError("Cannot compute evaluation quality without a recommended model payload")

    baseline_threshold = float(cfg.get("evaluation_quality", {}).get("baseline_threshold", 0.5))
    recommended_payload = per_model[recommended_model]

    slice_diag = compute_slice_diagnostics(
        cfg=cfg,
        overall=recommended_payload["overall"],
        slices=recommended_payload["slices"],
    )
    business_loss = compute_business_loss_reduction(
        cfg=cfg,
        use_case=use_case,
        decision=decision,
        per_model=per_model,
        model_scores=model_scores,
        baseline_threshold=baseline_threshold,
    )

    return {
        "run_id": run_id,
        "split": split,
        "dataset_size": {
            "split_count": int(split_count),
            "total_count_if_available": int(total_count) if total_count is not None else None,
        },
        "models_compared": int(len(per_model)),
        "slice_diagnostics": slice_diag,
        "business_loss_reduction": business_loss,
        "runtime": runtime,
    }


def compute_slice_diagnostics(
    cfg: Dict[str, Any],
    *,
    overall: Dict[str, Any],
    slices: List[Dict[str, Any]],
) -> Dict[str, Any]:
    min_count = int(cfg["evaluation"]["instability"]["min_count"])
    drop_threshold = float(cfg.get("evaluation_quality", {}).get("slice_drop_threshold", 0.35))
    cost_increase_threshold = float(
        cfg.get("evaluation_quality", {}).get("slice_cost_increase_threshold", 0.20)
    )
    eps = float(cfg.get("evaluation_quality", {}).get("cost_normalization_eps", DEFAULT_COST_NORMALIZATION_EPS))

    overall_expected_cost = float(overall["expected_cost_at_best_threshold"])
    diagnostics: List[Dict[str, Any]] = []
    metric_keys = ("recall", "precision", "f1")

    for slice_row in slices:
        unstable_flag = bool(slice_row.get("instability", {}).get("unstable", False))
        stable = int(slice_row.get("count", 0)) >= min_count or not unstable_flag
        if not stable:
            continue

        metrics_validity = slice_row.get("metrics_validity") or slice_row.get("metrics", {}).get("metrics_validity", {})
        worst_metric = None
        worst_drop = 0.0
        for metric_name in metric_keys:
            if not bool(metrics_validity.get(f"{metric_name}_valid", False)):
                continue
            overall_metric_raw = overall.get(metric_name)
            slice_metric_raw = slice_row.get("metrics", {}).get(metric_name)
            if overall_metric_raw is None or slice_metric_raw is None:
                continue
            overall_metric = float(overall_metric_raw)
            slice_metric = float(slice_metric_raw)
            drop_pct = relative_drop_pct(overall_metric, slice_metric)
            if drop_pct > worst_drop:
                worst_drop = drop_pct
                worst_metric = metric_name

        slice_expected_cost = float(slice_row.get("decision", {}).get("expected_cost", 0.0) or 0.0)
        cost_delta = compute_cost_increase(overall_expected_cost, slice_expected_cost, eps=eps)
        metric_triggered = worst_metric is not None and worst_drop >= drop_threshold
        cost_triggered = cost_delta["cost_increase_pct"] >= cost_increase_threshold
        qualifies = metric_triggered or cost_triggered
        if not qualifies:
            continue
        triggered_by = "metric_drop" if metric_triggered else "cost_increase"
        metric_used = worst_metric if metric_triggered else None

        diagnostics.append(
            {
                "slice_name": slice_row["slice_name"],
                "count": int(slice_row["count"]),
                "metric": metric_used,
                "metric_used": metric_used,
                "drop_pct": round(worst_drop, 6),
                "expected_cost_increase": round(cost_delta["cost_increase_pct"], 6),
                "cost_increase_abs": round(cost_delta["cost_increase_abs"], 6),
                "cost_increase_pct": round(cost_delta["cost_increase_pct"], 6),
                "triggered_by": triggered_by,
                "unstable_flag": unstable_flag,
                "metrics_validity": metrics_validity,
            }
        )

    diagnostics.sort(
        key=lambda row: (row["cost_increase_pct"], row["drop_pct"], row["count"]),
        reverse=True,
    )
    return {
        "definition": (
            "A stable slice is counted when its recall, precision, or f1 drops by at least the configured "
            "threshold versus the overall recommended-model metric for metrics that are valid on that slice, "
            "or when its optimized expected cost rises by at least the configured normalized threshold versus "
            "the overall optimized expected cost."
        ),
        "thresholds": {
            "min_count": min_count,
            "slice_drop_threshold": drop_threshold,
            "slice_cost_increase_threshold": cost_increase_threshold,
            "cost_normalization_eps": eps,
        },
        "stable_only": True,
        "diagnostics_found": int(len(diagnostics)),
        "top_diagnostics": diagnostics[:5],
    }


def compute_business_loss_reduction(
    cfg: Dict[str, Any],
    *,
    use_case: str,
    decision: Dict[str, Any],
    per_model: Dict[str, Dict[str, Any]],
    model_scores: Dict[str, Dict[str, Any]],
    baseline_threshold: float,
) -> Dict[str, Any]:
    recommended_model = decision["recommended_model"]
    recommended_payload = per_model[recommended_model]
    y_true = model_scores[recommended_model]["y_true"]
    y_score = model_scores[recommended_model]["y_score"]
    use_case_cfg = load_costs(cfg)["use_cases"][use_case]["binary"]

    baseline_expected_cost = expected_cost_binary(y_true, y_score, baseline_threshold, use_case_cfg)
    optimized_expected_cost = float(recommended_payload["best"]["expected_cost"])
    optimized_threshold = float(recommended_payload["best"]["threshold"])

    ranking = decision.get("rationale", {}).get("ranking", [])
    winner_cost = float(ranking[0]["expected_cost"]) if ranking else optimized_expected_cost
    runnerup_cost = float(ranking[1]["expected_cost"]) if len(ranking) > 1 else winner_cost

    return {
        "use_case": use_case,
        "baseline_threshold": baseline_threshold,
        "optimized_threshold": optimized_threshold,
        "baseline_expected_cost": float(baseline_expected_cost),
        "optimized_expected_cost": optimized_expected_cost,
        "reduction_pct": safe_relative_reduction(baseline_expected_cost, optimized_expected_cost),
        "winner_vs_runnerup_cost_delta": float(runnerup_cost - winner_cost),
    }


def safe_relative_reduction(baseline_cost: float, optimized_cost: float) -> float:
    denom = abs(baseline_cost)
    if denom == 0.0:
        return 0.0
    return float((baseline_cost - optimized_cost) / denom)


def relative_drop_pct(overall_value: float, slice_value: float) -> float:
    if overall_value <= 0.0:
        return 0.0
    return float(max(0.0, (overall_value - slice_value) / overall_value))


def compute_cost_increase(overall_cost: float, slice_cost: float, *, eps: float = DEFAULT_COST_NORMALIZATION_EPS) -> Dict[str, float]:
    cost_increase_abs = float(slice_cost - overall_cost)
    denom = max(abs(overall_cost), eps)
    cost_increase_pct = float(max(0.0, cost_increase_abs / denom))
    return {
        "cost_increase_abs": cost_increase_abs,
        "cost_increase_pct": cost_increase_pct,
    }


def render_eval_quality_markdown(eval_quality: Dict[str, Any]) -> str:
    dataset = eval_quality["dataset_size"]
    slice_diag = eval_quality["slice_diagnostics"]
    business = eval_quality["business_loss_reduction"]
    runtime = eval_quality["runtime"]
    top = slice_diag.get("top_diagnostics", [])
    top_summary = ", ".join(
        f"{row['slice_name']} (trigger={row['triggered_by']}, metric={row['metric_used']}, "
        f"drop={row['drop_pct']:.1%}, cost+={row['cost_increase_pct']:.1%})"
        for row in top[:3]
    ) or "none"

    lines = [
        f"# Evaluation Quality Summary",
        f"- run_id: {eval_quality['run_id']}",
        f"- split: {eval_quality['split']}",
        f"- dataset_size.split_count: {dataset['split_count']}",
        f"- dataset_size.total_count_if_available: {dataset['total_count_if_available']}",
        f"- models_compared: {eval_quality['models_compared']}",
        f"- slice_diagnostics_found: {slice_diag['diagnostics_found']}",
        f"- stable_only: {slice_diag['stable_only']}",
        f"- slice_drop_threshold: {slice_diag['thresholds']['slice_drop_threshold']}",
        f"- slice_cost_increase_threshold: {slice_diag['thresholds']['slice_cost_increase_threshold']}",
        f"- baseline_threshold: {business['baseline_threshold']}",
        f"- optimized_threshold: {business['optimized_threshold']}",
        f"- baseline_expected_cost: {business['baseline_expected_cost']:.6f}",
        f"- optimized_expected_cost: {business['optimized_expected_cost']:.6f}",
        f"- reduction_pct: {business['reduction_pct']:.2%}",
        f"- winner_vs_runnerup_cost_delta: {business['winner_vs_runnerup_cost_delta']:.6f}",
        f"- runtime.total_sec: {runtime['total_sec']:.6f}",
        f"- top_diagnostics: {top_summary}",
    ]
    return "\n".join(lines) + "\n"

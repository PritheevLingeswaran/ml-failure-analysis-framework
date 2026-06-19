# Decision-theoretic evaluation framework

## The core idea
You don't deploy a model to maximize accuracy.
You deploy a model to make **decisions** with **consequences**.

Two models can have the same accuracy but radically different business impact
because FP and FN costs differ.

## Cost matrix (binary)
We define per-use-case costs in `configs/decision_costs.yaml`:

- TP: benefit (often negative cost)
- TN: neutral or minor benefit
- FP: cost of false alarm / unnecessary action
- FN: cost of missed detection / incident

Expected cost at threshold t:
- predict positive if p >= t
- compute mean cost across outcomes

## Threshold optimization
For each model:
- Evaluate expected cost across a threshold grid
- Choose threshold that minimizes expected cost

This often yields:
- A *different* threshold than "0.5"
- A *different* winning model than accuracy-based selection

## Slice-specific decisions
We repeat the same expected cost computation per slice.
Result:
- One model can be best globally but unacceptable in a critical segment.

## Output: defensible recommendation
The recommendation includes:
- Winning model and threshold
- Ranking with expected cost values
- Per-slice recommendations where relevant
- Instability flags to prevent overconfident decisions on tiny slices

## Evaluation quality audit layer
Every evaluation run also emits an `eval_quality` artifact and a timings artifact.
These values are defensible because they are computed from the same persisted predictions, slice tables, threshold sweeps, and cost matrices that drove the recommendation.

- `dataset_size`: the evaluated split row count plus the full dataset row count when available
- `models_compared`: the number of model payloads actually evaluated in the run, including any enabled ensemble artifact
- `slice_diagnostics`: only stable slices count, where stable means `count >= evaluation.instability.min_count` or the slice was not flagged unstable
- A stable slice is a diagnostic when recall, precision, or f1 drops by at least `evaluation_quality.slice_drop_threshold` versus the overall recommended-model metric, but only for metrics that are valid on that slice
- Undefined slice metrics are emitted as `null` with `metrics_validity` metadata and are excluded from metric-drop diagnostics
- A stable slice is also a diagnostic when optimized slice expected cost rises by at least `evaluation_quality.slice_cost_increase_threshold` on the normalized measure `cost_increase_pct = (slice_cost - overall_cost) / max(abs(overall_cost), eps)`
- `cost_increase_abs` is still emitted for readability, but thresholding uses normalized `cost_increase_pct` so the signal is comparable across runs with different cost scales or negative baseline cost
- `business_loss_reduction`: compares expected cost for the recommended model at the configured baseline threshold versus the optimized threshold already selected by the pipeline
- `runtime`: includes end-to-end wall-clock time and raw stage events for `load_data`, `make_splits`, `load_predictions`, `slicing`, `decision_optimization`, `error_analysis`, `plotting`, and `write_outputs`

Sample snippet:

```json
{
  "run_id": "8d1d2b3c4e5f",
  "split": "test",
  "dataset_size": {"split_count": 200, "total_count_if_available": 1000},
  "models_compared": 3,
  "slice_diagnostics": {
    "diagnostics_found": 2,
    "stable_only": true
  },
  "business_loss_reduction": {
    "baseline_threshold": 0.5,
    "optimized_threshold": 0.62,
    "reduction_pct": 0.18
  },
  "runtime": {
    "total_sec": 4.73,
    "stages": {"load_data": 0.02, "slicing": 1.11}
  }
}
```

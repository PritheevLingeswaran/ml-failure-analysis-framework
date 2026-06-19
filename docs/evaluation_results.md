# Sample evaluation outputs

After running:
- `python scripts/run_all.py --config configs/dev.yaml`

You will get:
- `outputs/metrics/comparison__<run_id>__test.json`
- `outputs/metrics/decision__<run_id>__test.json`
- `outputs/metrics/eval_quality__<run_id>__test.json`
- `outputs/metrics/timings__<run_id>__test.json`
- `outputs/reports/errors__<run_id>__test.json`
- `outputs/reports/eval_quality__<run_id>__test.md`
- `outputs/plots/*.png`

Interpretation guide:
- Start with decision ranking (expected cost). That is the objective.
- Check whether the winner changes under `low_fp` or `low_fn` use-cases.
- Inspect slice heatmap to identify segments where the global winner is not best.
- Use error clusters to identify recurring failure modes.

## Evaluation quality definitions
- `dataset_size.split_count` is the number of rows in the evaluated split, not the raw file size.
- `models_compared` is the count of model payloads that were actually evaluated in-memory for the run.
- `slice_diagnostics` only counts stable slices. Stability follows `evaluation.instability.min_count` and existing slice instability flags so tiny slices stay hypothesis-only.
- Metric degradation is measured as a relative drop from the overall recommended-model metric: `(overall - slice) / overall`.
- Cost degradation is measured as relative increase from the overall optimized expected cost for the recommended model.
- `business_loss_reduction` compares the recommended model's expected cost at `evaluation_quality.baseline_threshold` against the best expected cost found on the configured threshold grid.
- `winner_vs_runnerup_cost_delta` quantifies the margin between the recommended model and the second-ranked model on the same expected-cost objective.

Sample snippet:

```json
{
  "dataset_size": {
    "split_count": 200,
    "total_count_if_available": 1000
  },
  "models_compared": 3,
  "slice_diagnostics": {
    "diagnostics_found": 2,
    "top_diagnostics": [
      {
        "slice_name": "hard_examples",
        "metric": "recall",
        "drop_pct": 0.41,
        "expected_cost_increase": 0.27,
        "unstable_flag": false
      }
    ]
  },
  "business_loss_reduction": {
    "baseline_threshold": 0.5,
    "optimized_threshold": 0.62,
    "baseline_expected_cost": 0.91,
    "optimized_expected_cost": 0.74,
    "reduction_pct": 0.19
  }
}
```

# Sample evaluation outputs

After running:
- `python scripts/run_all.py --config configs/dev.yaml`

You will get:
- `outputs/metrics/comparison__<run_id>__test.json`
- `outputs/metrics/decision__<run_id>__test.json`
- `outputs/reports/errors__<run_id>__test.json`
- `outputs/plots/*.png`

Interpretation guide:
- Start with decision ranking (expected cost). That is the objective.
- Check whether the winner changes under `low_fp` or `low_fn` use-cases.
- Inspect slice heatmap to identify segments where the global winner is not best.
- Use error clusters to identify recurring failure modes.

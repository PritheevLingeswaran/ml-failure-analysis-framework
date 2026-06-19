# Failure cases (examples)

This repo ships with synthetic data. The point is to demonstrate failure analysis patterns.

Typical patterns you'll see after running `run_all`:
- Long text inputs may correlate with lower precision (more false positives)
- High-confidence bins may still contain systematic errors (overconfidence)
- High-value transactions (`amount >= 1000`) may show elevated false negatives (missed risk)

What to do with a failure case:
1) Confirm it's not an instability artifact (check slice count)
2) Inspect top FP/FN examples
3) Check clusters for repeated patterns
4) Decide if mitigation is:
   - threshold change
   - model switch for that slice
   - feature improvement
   - data/labeling fix

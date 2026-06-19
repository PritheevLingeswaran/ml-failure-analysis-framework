# Data layout

- `raw/datasets/` : raw input datasets (CSV for this reference implementation)
- `processed/` : derived artifacts (splits, features, labels, predictions)

This repo includes a script that generates a **synthetic sample dataset**:
`python scripts/prepare_data.py --config configs/dev.yaml`

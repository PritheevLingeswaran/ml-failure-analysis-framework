# ml-failure-analysis-framework

An internal, production-minded **model evaluation & decision platform** for ML organizations.

This system is built around one idea: **evaluation is about decisions, not scores.**
- There is no globally “best” model
- Aggregate metrics hide failures
- Error costs are unequal and business-context dependent
- Slices dominate real-world incidents
- Recommendations must be defensible and auditable

## What this repo provides

- Deterministic train/val/test splits
- 2–3 baseline models (Logistic Regression, Random Forest, optional XGBoost)
- Versioned predictions + model artifacts
- Standard metrics + calibration metrics
- Slice-based metrics and instability flags
- Error analysis: top FP/FN, per-slice confusion, clustering, representative cases
- Decision-theoretic evaluation: expected cost, threshold optimization, slice cost analysis
- Visualizations: cost curves, slice heatmaps, confidence vs accuracy, error distributions
- FastAPI service for evaluation/comparison/slices/errors/recommendations
- Config-driven, testable, auditable outputs

## Quickstart (local)

### 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare sample data
```bash
python scripts/prepare_data.py --config configs/dev.yaml
```

### 3) Train models + generate predictions
```bash
python scripts/train_models.py --config configs/dev.yaml
```

### 4) Run evaluation (metrics, slices, decisions, plots, reports)
```bash
python scripts/run_eval.py --config configs/dev.yaml
```

### 5) Start API
```bash
python scripts/run_api.py --config configs/dev.yaml
# then: http://localhost:8000/docs
```

## Where to look

- Docs:
  - `docs/architecture.md`
  - `docs/slicing_strategy.md`
  - `docs/decision_framework.md`
  - `docs/decisions.md`
  - `docs/tradeoffs.md`
  - `docs/failure_cases.md`
  - `docs/evaluation_results.md`

- Outputs:
  - `outputs/metrics/`
  - `outputs/plots/`
  - `outputs/reports/`
  - `outputs/logs/`

## Non-goals (by design)

This is not an AutoML or research playground. The goal is **repeatable, auditable evaluation** with slice-level transparency.

## License
MIT (see `LICENSE`).

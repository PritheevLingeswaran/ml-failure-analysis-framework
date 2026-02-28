# ml-failure-analysis-framework

An internal, production-minded **model evaluation & cost-aware decision platform** for ML organizations.

This system is built around one idea: **evaluation is about decisions, not leaderboard scores.**

- There is no globally “best” model  
- Aggregate metrics hide slice-level failures  
- Error costs are unequal and business-context dependent  
- Thresholds define decisions, not AUC  
- Slices dominate real-world production incidents  
- Recommendations must be defensible, transparent, and auditable  

---

## What this repo provides

- Deterministic train / validation / test splits  
- 2–3 baseline models (Logistic Regression, Random Forest, optional XGBoost)  
- Versioned predictions and model artifacts  
- Standard metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)  
- Calibration metrics (ECE, Brier score)  
- Slice-based metrics with instability detection  
- Error analysis:
  - Top false positives / false negatives  
  - Per-slice confusion matrices  
  - Failure clustering and representative cases  
- Decision-theoretic evaluation:
  - Configurable cost matrix (FP, FN, TP, TN)  
  - Expected loss computation  
  - Threshold optimization  
  - Slice-level cost analysis  
- Visualization artifacts:
  - Cost vs threshold curves  
  - Slice heatmaps  
  - Confidence vs accuracy plots  
  - Error distribution charts  
- FastAPI service exposing:
  - `/evaluate`
  - `/compare`
  - `/slices`
  - `/errors`
  - `/recommend`
- Config-driven, testable, and auditable outputs  

---

## Quickstart (Local)

### 1) Setup

```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

### 2) Prepare sample data

```bash
python scripts/prepare_data.py --config configs/dev.yaml
```

---

### 3) Train models + generate predictions

```bash
python scripts/train_models.py --config configs/dev.yaml
```

---

### 4) Run evaluation (metrics, slices, decisions, plots, reports)

```bash
python scripts/run_eval.py --config configs/dev.yaml
```

Artifacts will be generated under:

```
outputs/
├── metrics/
├── plots/
├── reports/
├── logs/
```

---

### 5) Start API

```bash
python scripts/run_api.py --config configs/dev.yaml
```

Then open:

```
http://localhost:8000/docs
```

Use the API to compare models, inspect slice-level failures, analyze error patterns, and generate cost-optimized recommendations.

---

## Where to look

### Documentation

- `docs/architecture.md`
- `docs/slicing_strategy.md`
- `docs/decision_framework.md`
- `docs/decisions.md`
- `docs/tradeoffs.md`
- `docs/failure_cases.md`
- `docs/evaluation_results.md`

### Generated Outputs

- `outputs/metrics/`
- `outputs/plots/`
- `outputs/reports/`
- `outputs/logs/`

---

## Non-goals (by design)

This is not an AutoML system or a research notebook.  
The objective is **repeatable, transparent, cost-aware evaluation** with slice-level visibility for real production decisions.

---

## License

MIT (see `LICENSE`).

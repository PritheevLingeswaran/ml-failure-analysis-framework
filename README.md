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
  - Threshold uncertainty (bootstrap CI)
  - Cost sensitivity scenarios (winner robustness)
  - Slice-level cost analysis  
- Visualization artifacts:
  - Cost vs threshold curves  
  - Slice heatmaps  
  - Confidence vs accuracy plots  
  - Error distribution charts  
- FastAPI service exposing:
  - `/evaluate`
  - `/evaluate/async`
  - `/jobs/{job_id}`
  - `/compare`
  - `/slices`
  - `/errors`
  - `/recommend`
  - `/diagnostics`
  - `/health`
  - `/version`
- Config-driven, testable, and auditable outputs  
 - Advanced diagnostics:
   - Probability calibration (`none`/`platt`/`isotonic`)
   - Time-aware split option (`data.split.strategy: time`)
   - Drift report (PSI/TV distance)
   - Risky slice auto-discovery
   - Fairness disparity slices for protected columns

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

## Data: synthetic, real, or your own

The pipeline is data-source agnostic. `scripts/prepare_data.py` writes a CSV to
`paths.data_raw`; everything downstream just consumes that CSV.

**Real, bundled dataset** (no synthetic numbers — real measurements):

```bash
python scripts/run_all.py --config configs/real.yaml   # Breast Cancer Wisconsin (569 real cases)
python scripts/run_api.py --config configs/real.yaml
```

`configs/real.yaml` uses `data.source: sklearn_breast_cancer` and the
`diagnostic_screening` cost matrix (missing a malignant case is catastrophic).
On this data the models score ROC-AUC ≈ 0.98 with ECE ≈ 0.01–0.03.

**Bring your own CSV** — no code changes needed:

1. Put a CSV at `paths.data_raw` with, at minimum, a binary `label` column (0/1)
   and an `id` column. Optionally a free-text column and a timestamp column.
2. Create a config (copy `configs/real.yaml`) and set:
   - `paths.data_raw` → your file
   - `data.dataset.label_col` / `id_col` / `text_col` / `time_col` (use `null` if absent)
   - `decision.default_use_case` → a cost matrix in `configs/decision_costs.yaml`
     that reflects *your* real cost of false positives vs false negatives.
3. Skip `prepare_data.py` (your file already exists) and run
   `train_models.py` → `run_eval.py`, or point the API at your config.

Columns you don't have (text, region, amount, time) degrade gracefully — the
slices that need them are skipped and logged, not errored.

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

## CI troubleshooting

Run the same core checks locally before pushing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m compileall -q src scripts evaluation tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest tests -q
```

Typical CI failure causes:

- Wrong interpreter: CI is pinned to Python 3.11, so reproduce with 3.11 locally when debugging version-specific issues.
- Missing or stale dependencies: reinstall from `requirements.txt` and run `python -m pip freeze` to confirm the environment matches CI.
- Pytest plugin interference: CI disables auto-loaded third-party plugins with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to keep test discovery predictable.
- Filesystem assumptions: tests should write only to temporary directories such as `tmp_path`, not repo-level `outputs/` or machine-specific paths.

---

## License

MIT (see `LICENSE`).

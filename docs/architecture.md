# Architecture

## System goal
Provide a repeatable, auditable evaluation workflow that explains:
- **Where models fail**
- **Why they fail**
- **What those failures cost**
- **Which model + threshold is best for a specific decision context**

## High-level flow

1) **Config** (`configs/*.yaml`)
- Dataset definition, split seed
- Model list and parameters
- Slice definitions
- Decision cost matrices
- Logging settings

2) **Data layer** (`src/datasets/`)
- Load raw dataset
- Deterministic splits: train/val/test
- Preserve metadata for slicing

3) **Model layer** (`src/models/`)
- Unified training + inference interface
- Save model artifacts and versioned predictions

4) **Evaluation engine** (`src/evaluation_engine/`)
- Standard metrics (accuracy/F1/AUC)
- Calibration (ECE/Brier)
- Confidence-aware summaries

5) **Slicing engine** (`src/slicing/`)
- Rule-based slices from `slices.yaml`
- Automatic slices (text length, confidence, label frequency, easy/hard)
- Per-slice metrics + per-slice cost-opt threshold
- Instability flags for low-N slices

6) **Error analysis** (`src/error_analysis/`)
- Top FP/FN (high-confidence wrong predictions)
- Clustering of failure modes (TF-IDF + KMeans baseline)

7) **Decision engine** (`src/decision_engine/`)
- Cost matrix → expected loss
- Threshold optimization (overall + per slice)
- Recommendation based on **min expected cost**, not accuracy

8) **Visualization** (`src/visualization/`)
- Cost vs threshold curves
- Confidence vs accuracy
- Slice cost heatmap

9) **API** (`src/api/`)
- `/evaluate`, `/compare`, `/slices`, `/errors`, `/recommend`

## Auditability guarantees
- Deterministic run_id based on config hash
- Persisted artifacts:
  - `data/processed/predictions/<run_id>/...`
  - `outputs/metrics/*.json`
  - `outputs/reports/*.json`
  - `outputs/plots/*.png`
  - `outputs/logs/run.log`

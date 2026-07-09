#!/bin/sh
set -e

# Config the API will serve (compose/k8s can override via CONFIG_PATH).
CONFIG_PATH="${CONFIG_PATH:-configs/prod.yaml}"
export CONFIG_PATH

# Apply database migrations (idempotent). Uses DATABASE_URL if set, else the
# default SQLite file. Skip with SKIP_MIGRATIONS=1.
if [ "${SKIP_MIGRATIONS:-0}" != "1" ]; then
  echo "[entrypoint] applying database migrations (alembic upgrade head)..."
  python -m alembic upgrade head
fi

# Bootstrap the data pipeline only when the predictions this config needs are
# absent. If you mount real data + trained models, this is skipped and the
# container serves them immediately. Set SKIP_BOOTSTRAP=1 to always skip.
if [ "${SKIP_BOOTSTRAP:-0}" != "1" ]; then
  if python - <<PY
import os, sys
from src.utils.config import load_config
from src.evaluation_engine.predictions import build_run_id
cfg = load_config(os.environ["CONFIG_PATH"])
rid = build_run_id(cfg)
need = f"data/processed/predictions/{rid}/logreg__test.csv"
sys.exit(0 if os.path.exists(need) else 1)
PY
  then
    echo "[entrypoint] model artifacts present; skipping bootstrap."
  else
    echo "[entrypoint] artifacts missing -> preparing data and training models for ${CONFIG_PATH}..."
    python scripts/prepare_data.py --config "$CONFIG_PATH"
    python scripts/train_models.py --config "$CONFIG_PATH"
    echo "[entrypoint] bootstrap complete."
  fi
fi

exec "$@"

# Deployment

Production deployment guide for the ML Failure Analysis API. Artifacts:

| Artifact | Purpose |
|---|---|
| `Dockerfile` | Multi-stage, non-root (uid 10001), pinned deps, healthcheck |
| `docker-compose.yml` | Local prod-like stack: api + Redis + Postgres |
| `docker-compose.prod.yml` | Prod override: prebuilt image, workers, limits, restart |
| `k8s/` | Kustomize manifests (Deployment, Service, HPA, Ingress, migrate Job) |
| `alembic/` | Database schema migrations |

## Required configuration

Everything is env-driven (env var wins over YAML). Secrets must come from a secret
store — never bake them into the image or commit them.

| Variable | Required | Example | Notes |
|---|---|---|---|
| `APP_ENV` | yes | `prod` | Any non-local value makes auth **mandatory** |
| `MLFA_API_KEY` | yes (prod) | `<random>` | App refuses to start without it in prod |
| `DATABASE_URL` | prod | `postgresql+psycopg2://user:pass@host:5432/mlfa` | SQLite used if unset (dev only) |
| `REDIS_URL` | prod | `redis://host:6379/0` | Enables shared cache + shared rate limiter |
| `CORS_ORIGINS` | no | `https://app.example.com` | Comma-separated allow-list; empty = none |
| `RATE_LIMIT_DEFAULT` | no | `120/minute` | Per API key / IP |
| `LOG_FORMAT` | no | `json` | JSON logs for aggregation |
| `WORKERS` | no | `2` | uvicorn workers per pod/container |
| `SKIP_MIGRATIONS` | no | `1` | Set when a migration Job runs them instead |
| `SKIP_BOOTSTRAP` | no | `1` | Set in prod; don't generate demo data |
| `MAX_BODY_BYTES` / `MAX_UPLOAD_BYTES` / `MAX_TRAIN_ROWS` | no | | Request/upload/run guards |

## Local prod-like stack (Docker Compose)

```bash
export MLFA_API_KEY=dev-secret
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
curl -s localhost:8000/readyz            # {"status":"ready", ...}
curl -s -H "X-API-Key: dev-secret" localhost:8000/compare | jq .summary.winner
```

## Kubernetes

Postgres and Redis are expected to be **external/managed** (RDS + ElastiCache, or
their operators) — the manifests reference them via `DATABASE_URL` / `REDIS_URL`
in the Secret, and do not deploy stateful DB/cache pods.

```bash
# 1) Create the secret out-of-band (NOT in git):
kubectl create namespace mlfa
kubectl -n mlfa create secret generic mlfa-secrets \
  --from-literal=MLFA_API_KEY="$(openssl rand -hex 24)" \
  --from-literal=DATABASE_URL='postgresql+psycopg2://user:pass@your-postgres:5432/mlfa' \
  --from-literal=REDIS_URL='redis://your-redis:6379/0'

# 2) Set the image tag and apply:
(cd k8s && kustomize edit set image mlfa=ghcr.io/OWNER/REPO:v1.2.3)
kubectl apply -k k8s

# 3) Migrations run as a one-shot Job (mlfa-migrate); app pods have SKIP_MIGRATIONS=1.
kubectl -n mlfa wait --for=condition=complete job/mlfa-migrate --timeout=120s
```

Probes: `livenessProbe` → `/healthz` (restart a hung pod), `readinessProbe` →
`/readyz` (checks DB + Redis; a dependency blip removes the pod from the Service
without killing it). `/metrics` is annotated for Prometheus scraping.

## Scaling notes

- **Workers vs replicas:** each pod runs `WORKERS` uvicorn workers (CPU-bound
  evaluation → ~1 worker per core; `resources.limits.cpu` caps it). Scale *out*
  with replicas (HPA targets 70% CPU, 3→10). Because the cache and rate limiter
  are **Redis-backed**, N workers/replicas share one cache (each evaluation is
  computed once, not N times) and one global rate-limit counter.
- **Single-flight** is per-process; across replicas the shared cache means each
  replica computes a cold key at most once. A fully distributed lock is not
  implemented (documented gap).
- **DB connections:** total ≈ replicas × workers × pool size. Size the Postgres
  `max_connections` (or use PgBouncer) accordingly.

## Backup & restore (Postgres)

```bash
# Backup (schedule via CronJob / managed snapshots)
pg_dump "$DATABASE_URL" -Fc -f mlfa-$(date +%F).dump
# Restore into an empty database
pg_restore --clean --if-exists -d "$DATABASE_URL" mlfa-YYYY-MM-DD.dump
```
Uploaded dataset files (`data/uploads/`) live on the data volume / object store —
back those up alongside the DB so run history and its files stay consistent.

## Rollback

1. **App rollback** (safe, fast): redeploy the previous image tag.
   `kubectl -n mlfa set image deploy/mlfa-api api=ghcr.io/OWNER/REPO:<prev>`
   or `kubectl -n mlfa rollout undo deploy/mlfa-api`.
2. **Schema rollback** (careful): only if the new release added a migration that
   the old code can't tolerate. `alembic downgrade -1` — but a downgrade that
   drops columns is destructive; prefer forward-fixes and backup first.
3. Rolling strategy is `maxUnavailable: 0`, so a bad rollout never drops capacity
   below the current replica count while it converges.

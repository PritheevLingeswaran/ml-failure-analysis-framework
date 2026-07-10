# Changelog

## Unreleased — production hardening

### Deployment (Phase 0)
- Self-bootstrapping Docker image: entrypoint prepares data + trains models when
  artifacts are absent (skipped if you mount real data/models). Verified end to end.

### Security (Phase 1)
- Auth mandatory in non-local environments (app refuses to start without a key).
- Rate limiting per API key / IP; strict input validation; request body-size cap;
  CORS locked to an explicit allow-list. `pip-audit`: no known vulnerabilities.

### Shared state & persistence (Phase 2)
- Pluggable evaluation cache: in-memory locally, **Redis** when `REDIS_URL` is set.
- **Postgres** persistence via SQLAlchemy (SQLite by default), **Alembic** migrations.
- `POST /datasets/upload`, `GET /datasets[/{id}]`, `POST /runs`, `GET /runs[/{id}]`
  — upload a CSV, run a cost-aware analysis on it, browse run history.

### Observability (Phase 3)
- Redis-backed rate limiter (shared across workers; fails open to in-memory).
- Structured JSON logging with a per-request ID propagated via contextvar through
  the handler, cache, and rate limiter (grep one `request_id` to trace a request).
- Prometheus `/metrics` (latency, request/error counts, cache hit/miss, rate-limit
  rejections); `/healthz` (liveness) and `/readyz` (readiness: DB + Redis checks).

### Deployment readiness (Phase 4)
- Multi-stage, **non-root** (uid 10001) Dockerfile with a healthcheck.
- `requirements.txt` **pinned to exact versions** (reproducible builds); dev/test
  tools split into `requirements-dev.txt` so the runtime image stays lean.
- `docker-compose.prod.yml` override (prebuilt image, multiple uvicorn workers,
  resource limits, restart policy) + `src/api/asgi.py` factory for `--workers`.
- Kubernetes manifests under `k8s/` (Kustomize, no Helm): Deployment with
  liveness/readiness probes + non-root securityContext, Service, HPA, Ingress,
  and a one-shot migration Job. `DEPLOYMENT.md` covers env, scaling, backup, rollback.

### Load testing & CI/CD (Phase 5)
- k6 load test (`load/`) that exercises the real training path (`/datasets/upload`
  + `/runs`) under the 2-worker + Redis + Postgres stack; baseline in `load/README.md`.
- Removed unnecessary artifact-file writes from the API request path (in-memory
  JSON only) — cuts hot-path I/O and fixes a non-root/read-only-fs write error.
- CI: `pip-audit` gate on every push; build + push the image to **GHCR** on merge
  to `main` (SHA + `latest` tags) using the built-in `GITHUB_TOKEN`.

### Deliberately scoped out / future optimizations
- **`CostMatrixConfig` CRUD**: table is modeled and migrated, but no HTTP
  endpoints are exposed yet — per-user/session cost-matrix management is future work.
- Cross-worker single-flight uses the shared Redis cache (each worker computes at
  most once); a fully distributed lock is not implemented.
- **`/runs` is synchronous and retrains per call**; a production build would make
  it async (job queue) and cache trained models per dataset. (The same-dataset
  *race condition* is fixed — concurrent runs are now serialized by a distributed
  lock, Redis-backed cross-worker, per-process fallback. See `src/api/locks.py`.)
- **Image size (~940 MB) left unoptimized on purpose**: the size is dominated by
  the scientific-stack wheels (numpy/pandas/scipy/scikit-learn/matplotlib), not
  build tooling — the multi-stage build already strips compilers/caches. Further
  slimming (dropping matplotlib from the runtime, or a slim scientific base image)
  is a deliberate future task, not a quick win.

## 0.1.0
- Initial implementation of evaluation + slicing + decision-theoretic framework
- FastAPI endpoints for evaluation and recommendations

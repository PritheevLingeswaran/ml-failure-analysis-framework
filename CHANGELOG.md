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

### Deliberately scoped out (documented gaps)
- **`CostMatrixConfig` CRUD**: the table is modeled and migrated, but no HTTP
  endpoints are exposed yet — per-user/session cost-matrix management is future work.
- Cross-worker single-flight uses the shared Redis cache (each worker computes at
  most once); a fully distributed lock is not implemented.

## 0.1.0
- Initial implementation of evaluation + slicing + decision-theoretic framework
- FastAPI endpoints for evaluation and recommendations

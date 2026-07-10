# Load testing

`k6_loadtest.js` exercises the **real work paths** — `POST /datasets/upload` +
`POST /runs` (which trains models, evaluates, and persists to Postgres) alongside
cached `GET /compare` reads — not cheap endpoints like `/healthz`.

## Run it

```bash
# 1) Bring up the prod-like stack (2 workers + Redis + Postgres).
#    Raise the rate limit so it doesn't cap the capacity you're measuring.
export MLFA_API_KEY=dev-secret RATE_LIMIT_DEFAULT=1000000/minute
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 2) Run k6 (containerized — no install needed).
docker run --rm -e BASE=http://host.docker.internal:8000 -e API_KEY=dev-secret \
  -v "${PWD}/load:/scripts" grafana/k6 run /scripts/k6_loadtest.js
```

## Baseline (this machine, local compose — NOT a cloud benchmark)

Config: `docker-compose.prod.yml`, **2 uvicorn workers**, Redis cache + limiter,
Postgres persistence; 8 VUs reading + 1 VU training, 25 s.

| Path | Metric | Result |
|---|---|---|
| `GET /compare` (Redis cache hit) | throughput | **~70 req/s** |
| | p95 latency | **~212 ms** (median ~75 ms) |
| `POST /runs` (upload→train→eval→persist) | latency p95 | **~15 s** under concurrent read load (~6–8 s isolated) |
| | throughput | ~0.06 runs/s per worker (CPU-bound) |
| Overall | error rate | **0%** (0 / ~2,170 requests) |

### How to read these

- **Reads scale with the cache, not recomputation** — once `/compare` is warm,
  every worker/replica serves it from Redis, so throughput is bound by the web
  layer, not the ~8 s evaluation.
- **`/runs` is CPU-bound and gated by worker count** — it genuinely trains models.
  Throughput ≈ workers / train-time; scale it with replicas (HPA), not by
  hammering a fixed pod. Under concurrent read load a single run slows (~6 s → ~15 s)
  because reads share the same 2 workers.

### Known limitations surfaced by this test

- Concurrent `/runs` on the **same** dataset share a run-id and race on the
  shared predictions directory — run different datasets concurrently, or serialize
  per-dataset. A production build would make `/runs` async (job queue) and cache
  trained models per dataset instead of retraining each call.
- Numbers are from one developer machine; treat them as *relative* signal
  (reads ≫ runs; 0% errors), not an SLA. Real capacity planning needs the target
  infra and a longer soak.

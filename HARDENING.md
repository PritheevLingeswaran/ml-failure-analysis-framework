# Hardening this platform: from "portfolio demo" to "deployable service"

This is the story of taking an ML failure-analysis platform — a FastAPI backend,
a React dashboard, and a decision-theoretic evaluation engine — from something
that *ran on my laptop and looked finished* to something that could actually be
deployed. It's written as a narrative because the interesting part isn't the
feature list; it's **what broke, why, and how each break got caught.**

A running theme: almost every real bug was found by *trying to run the thing for
real* — building the actual Docker image, standing up the actual Postgres, firing
actual concurrent load — not by reading the code. That's the meta-lesson.

---

## The starting point

The app worked in the sense that `python scripts/run_api.py` served requests and
the dashboard rendered. But "works on my machine" was hiding a lot: unpinned
dependencies, an untested Dockerfile, an in-memory cache that silently breaks with
more than one worker, optional auth, no persistence, no metrics, no load numbers.
It was a *demo*, and demos lie by omission.

I worked in six phases, each committed and merged to `main` separately so the
history reads as a reviewable progression.

---

## Phase 0 — First contact with reality

**Goal:** before touching anything, confirm the branch was merged and *actually
build and run the Docker image.*

**What broke:** the image built fine and `/health` returned 200 — but `/compare`
threw a 500. Root cause: `.dockerignore` excluded `data/processed`, so the
container shipped **no trained models**, and the first real request hit a
`FileNotFoundError` on the predictions file. The Dockerfile had never been run
end to end; it only *looked* done.

**Fix:** a self-bootstrapping entrypoint that trains the models on first start if
artifacts are missing (and is skipped when you mount real data). Now `docker run`
works out of the box.

**Lesson to tell:** "The Dockerfile compiled but was never exercised. The moment
I ran it, the missing-artifacts problem was obvious. Build ≠ runs."

---

## Phase 1 — Locking the doors

**Goal:** auth, rate limiting, input validation, CORS.

The API was **open** — any caller, no key, could read everything, and CORS was
permissive. I made auth *mandatory in non-local environments*: the app now
**refuses to start** in prod without an API key (fail-fast beats failing open),
which I verified in the container (no key → exit 1; with key → 401 without header,
200 with). CORS became an explicit allow-list; request bodies got a size cap;
the `use_case` input got a strict pattern.

**The surprise bug:** I reached for `slowapi` for rate limiting, wired it per the
docs, and my test showed it *silently not enforcing*. I bisected it in a minimal
repro and found that `slowapi`'s `default_limits` **don't apply to routes mounted
on an `APIRouter`** — which is every one of my endpoints. A rate limiter that
looks configured but never triggers is worse than none. I dropped it and wrote a
small fixed-window limiter I could actually test.

**Lesson to tell:** "The rate limiter passed code review and did nothing. Only a
test that actually sent >limit requests caught it. Verify the security control
*fires*, don't trust that it's wired."

---

## Phase 2 — Shared state (why the in-memory cache was a lie)

**Goal:** make it correct with more than one worker.

The evaluation cache and rate limiter both lived in **process memory**. That's
fine with one worker and a lie with several: each worker has its own cold cache,
so N workers recompute the same ~8s evaluation N times, and a "120/min" rate limit
becomes "120/min *per worker*." The fix is shared state: **Redis** for the cache
(pluggable — in-memory locally, Redis when `REDIS_URL` is set), and **Postgres**
for durable data via SQLAlchemy, with **Alembic migrations** (not `create_all`,
because `create_all` can't evolve or roll back a live schema).

I also built the feature that makes it demoable: `POST /datasets/upload` +
`POST /runs` + history endpoints — upload a CSV, run a cost-aware analysis on it,
browse past runs.

**What I verified honestly:** at first I could *only* test this against SQLite +
unit tests, because Docker Desktop's daemon was down when I tried the full compose
stack. I said so explicitly rather than claiming the Postgres path worked. It was
Phase 3 — when the daemon came back — that actually ran the Alembic migration
against real Postgres and persisted a real run. I went back and marked Phase 2
*verified* only then.

**Lesson to tell:** "In-memory shared state is the classic single-worker trap.
Redis isn't there for speed here — it's there for *correctness* across workers."

---

## Phase 3 — Seeing inside

**Goal:** observability, and finish the Redis migration.

Two things worth explaining well:

- **Request tracing done right.** I put the request ID in a **contextvar**, not a
  function argument, so *any* code in the request's context — the handler, the
  cache layer, the rate limiter — can read it, and a logging filter stamps it onto
  every record. The proof: I fired one request with `X-Request-ID: trace-xyz-999`
  and grepped the JSON logs — that ID appeared on lines from **four different
  loggers** (the cache, the dataset loader, the splitter, the evaluation engine).
  That's what makes "trace one request end to end" *true* instead of aspirational.

- **`/healthz` vs `/readyz`.** Liveness answers "is the process alive?" and drives
  whether Kubernetes *restarts* a hung pod. Readiness answers "can it serve *right
  now*?" — it checks Redis + Postgres — and drives whether the load balancer
  *routes* to it. Splitting them means a DB blip pulls a pod from rotation without
  killing it, so it recovers instead of crash-looping. Collapsing them is a classic
  cascading-restart cause.

I also moved the rate limiter to Redis (atomic `INCR`/`EXPIRE`), fixing the
per-worker-limit lie, and fails *open* to in-memory if Redis is down — a limiter
should never be the thing that takes the API down. Prometheus `/metrics` exposes
latency, cache hit/miss, and rate-limit rejections.

---

## Phase 4 — Packaging for the world

**Goal:** multi-stage non-root image, pinned deps, prod compose, k8s manifests.

**Bug 1 — dependency drift.** The build pulled pandas 3.0 / numpy 2.4 — newer than
I'd tested — because `requirements.txt` used `>=`. I pinned everything to exact
versions (the image you test is the image you ship) and split dev tools into a
separate file so the runtime image doesn't carry pytest.

**Bug 2 — non-root can't write.** I switched the image to a non-root user (uid
10001) for defense in depth. It promptly crash-looped: `sqlite3.OperationalError:
unable to open database file`, then in compose `PermissionError` on the data dir.
Root cause: the `data/` directory arrived in the image as mode **555** — owned by
the app user but with no write bit — so `chown` wasn't enough; I had to `chmod`
it writable. Non-root is a great default that surfaces every "assumed I could
write here" assumption at once.

**Bug 3 — the stale-image trap.** The multi-worker compose crash-looped with
`Could not import module "src.api.asgi"`. The module existed — but I'd *built the
image before creating the file*, so the running container had a stale layer. A
rebuild fixed it. Easy to lose an hour to; worth naming.

**Bug 4 — an invalid CI expression.** The editor flagged `TMPDIR: ${{ runner.temp
}}` in the workflow's *job-level* `env`, where the `runner` context isn't valid
(it only exists inside steps). Pre-existing, and exactly the kind of thing that
bites the moment you're watching CI. Removed it.

k8s: plain manifests + Kustomize, **not Helm** — deliberately. Helm earns its keep
packaging a chart for others with many knobs; for one app (Deployment + Service +
HPA + Ingress + a migration Job) its Go-templating is indirection you read
*around*. I validated the manifests with `kubectl kustomize` (renders 7 resources,
image substituted) but was explicit that **server-side/cluster validation needs a
cluster I don't have here.**

---

## Phase 5 — Proving it under load

**Goal:** load test the *real* work, publish the image, scan deps.

I deliberately load-tested the **training path** (`/datasets/upload` + `/runs`),
not `/healthz` — a req/s number on a health check is a vanity metric. The k6 script
uploads a dataset, then hammers `/runs` (CPU-bound: trains + evaluates + persists)
alongside cached `/compare` reads, against the 2-worker + Redis + Postgres stack.

**The bug the load test found:** under load, everything 500'd with `PermissionError`
writing `outputs/metrics/*.json`. The API was persisting **artifact files on every
request** — needless hot-path I/O that also collided with the non-root container's
mounted volume. The API returns JSON in-memory; it never needed those files. I
added a flag so the request path skips artifact writes (CLI runs still write them).
This is a good story because the *performance* smell and the *permission* bug were
the same root cause, and only load exposed it.

**The numbers I can quote** (local machine, not a cloud benchmark, and I say so):
cached reads ~**70 req/s at p95 212 ms**; `/runs` ~**15 s p95** under concurrent
load (~6–8 s isolated), throughput gated by worker count; **0% errors over ~2,170
requests.** The interesting part is the *shape*: reads scale with the Redis cache,
runs are CPU-bound and scale only with more workers/replicas.

CI now runs a `pip-audit` gate on every push and publishes the image to **GHCR**
on merge to `main` using the built-in `GITHUB_TOKEN` — no external registry
secret. (Honest caveat: the actual push runs inside Actions with a packages-scoped
token; I wrote and validated the workflow but can't execute that step locally.)

---

## Post-review — the one real bug left in the gaps list

The gaps list had one item that wasn't a *scoping decision* but an actual bug:
two concurrent `/runs` on the **same** dataset resolve to the same run-id and race
on the shared predictions directory. I fixed it with a **distributed lock** keyed
per dataset — Redis-backed (correct across workers, auto-expiring lease so a crash
can't wedge it), with a per-process fallback. Verified three ways: a unit test of
the primitive, an endpoint test showing two concurrent same-dataset runs serialize
(both succeed, no race) instead of colliding, and a smoke test against real Redis
(same-key contention → `LockBusy`, different keys parallel).

---

## The bug catalogue (the war stories)

| # | Bug | Root cause | How it was caught |
|---|---|---|---|
| 1 | **ECE inflated to 0.6–0.9** | Binned by raw probability and compared to accuracy instead of the standard confidence-based ECE (Guo et al.) | Numbers looked implausible; traced the formula |
| 2 | **`event_time` data leakage** | A string timestamp got one-hot encoded into ~6000 junk feature columns; also broke logreg convergence & drift | The 6024-wide feature matrix was the tell |
| 3 | **Request stampede** | Concurrent cold requests each recomputed the ~8s evaluation | Frontend hung; added single-flight + shared cache |
| 4 | **Rate limiter didn't fire** | `slowapi` `default_limits` don't enforce on `APIRouter` routes | A test that sent >limit requests still got 200s |
| 5 | **Non-root can't write** | `data/` dir shipped as mode 555; `chown` without `chmod` | Container crash-looped on first DB write |
| 6 | **Artifact hot-path I/O** | API persisted metric JSON on every request; collided with non-root volume | Surfaced by the load test (500s under load) |
| 7 | **Invalid CI expression** | `runner.temp` used in job-level `env` where the context isn't valid | Editor/schema warning while editing CI |
| 8 | **`/runs` same-dataset race** | Concurrent runs share a run-id + predictions dir | Reasoned + reproduced under concurrent load |

**On "cross-tenant leak" specifically** — to be precise and not overclaim in an
interview: this system is **single-tenant**, so there was no literal cross-tenant
*data* leak. The two things that map to that concern are (a) the **open API** —
before Phase 1 any unauthenticated caller could read all results, an isolation gap
that mandatory auth closed — and (b) the **`event_time` data leakage** (#2), which
is ML *feature* leakage, a different meaning of the word. Claim those two; don't
claim a cross-tenant data breach that didn't happen.

---

## Verified vs assumed vs can't-verify (be honest about this in interviews)

**Genuinely verified** against real infrastructure on my machine:
- Docker image builds, runs non-root, healthy; auth fail-fast; all endpoints.
- Full `api + Redis + Postgres` compose stack: Alembic migration against real
  Postgres, upload→train→persist a run, request-ID traced across loggers, `/readyz`
  reporting real dependency health, 2-worker mode, the run lock against real Redis.
- 51 automated tests; `pip-audit` clean on pinned deps; k6 load numbers.

**Validated but not fully executed:**
- k8s manifests render via `kubectl kustomize` — but no cluster here for a real apply.
- The GHCR publish workflow is written + YAML-validated — but the push runs in CI
  with a token I don't have locally.

**Not claimed:**
- CI "green on GitHub Actions" from my side (no Actions visibility locally).
- Any cloud-scale performance number — the load figures are one laptop.

---

## Honest gaps that remain (design decisions, not oversights)

- **`/runs` is synchronous and retrains per call.** The right production shape is
  an async job queue + caching trained models per dataset. (The race is fixed; the
  synchronicity is a scoping choice.)
- **`CostMatrixConfig`** is modeled and migrated but has no CRUD endpoints yet.
- **Cross-worker single-flight** relies on the shared cache (each worker computes a
  cold key at most once), not a distributed lock on the compute itself.
- **Image ~940 MB** — left as-is on purpose: dominated by scientific-stack wheels
  (numpy/pandas/scipy/scikit-learn/matplotlib), not build tooling, which multi-stage
  already strips. Slimming it is real work, not a quick win.
- **Still demo data by nature.** The framework is production-shaped; the *numbers*
  demonstrate the framework, not a real model's quality.

---

## The two-minute version (how to open the story)

"I took an ML evaluation platform from a working demo to a deployable service in
six phases. The through-line is that every real bug was found by *running it for
real*, not reading code: the Docker image that built but 500'd on missing models;
a rate limiter that passed review but never fired; a non-root user that couldn't
write its own data dir; and — my favorite — a load test that revealed the API was
doing pointless disk I/O on every request, which was *also* the thing breaking
under the non-root container. I fixed a genuine race condition in the training
endpoint with a Redis lock, added Postgres persistence with real migrations,
Redis-backed shared cache and rate limiting, request-ID tracing you can grep end
to end, Prometheus metrics, split liveness/readiness probes, a multi-stage non-root
image, and k8s manifests. And I can tell you exactly what I verified against real
infrastructure versus what I could only validate on my laptop — including that I
haven't watched the GHCR push go green yet."

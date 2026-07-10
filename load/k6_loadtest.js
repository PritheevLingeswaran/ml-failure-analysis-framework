// Load test for the REAL work paths, not cheap endpoints.
//
// Scenarios (run concurrently against the compose.prod stack: 2 workers + Redis
// + Postgres):
//   - trainRuns:  POST /runs  -> trains models + evaluates + persists (CPU-bound)
//   - cachedReads: GET /compare -> served from the Redis cache (cheap)
//
// setup() uploads one dataset and warms the cache. Per-path latency is reported
// via custom trends so you can quote each number separately.
//
// Run (from repo root, with compose.prod up):
//   docker run --rm -e BASE=http://host.docker.internal:8000 -e API_KEY=dev-secret \
//     -v "${PWD}/load:/scripts" grafana/k6 run /scripts/k6_loadtest.js
import http from "k6/http";
import { check } from "k6";
import { Trend, Counter } from "k6/metrics";

const BASE = __ENV.BASE || "http://host.docker.internal:8000";
const KEY = __ENV.API_KEY || "dev-secret";
const H = { "X-API-Key": KEY };

const runTrend = new Trend("run_duration_ms", true);
const readTrend = new Trend("read_duration_ms", true);
const runsOk = new Counter("runs_completed");
const readsOk = new Counter("reads_completed");

export const options = {
  scenarios: {
    // Training is CPU-bound; 1 VU = sequential runs (no same-dataset race) so
    // the number reflects true per-run latency. Reads run concurrently on the
    // other worker, served from the Redis cache.
    trainRuns: { executor: "constant-vus", exec: "trainRun", vus: 1, duration: "25s" },
    cachedReads: { executor: "constant-vus", exec: "cachedRead", vus: 8, duration: "25s" },
  },
  // Informational thresholds (test still reports numbers if crossed).
  thresholds: {
    "http_req_failed": ["rate<0.10"],
    "read_duration_ms": ["p(95)<3000"],
    "run_duration_ms": ["p(95)<20000"],
  },
};

function makeCsv(rows) {
  let out = "f0,f1,f2,f3,f4,label\n";
  for (let i = 0; i < rows; i++) {
    const feats = Array.from({ length: 5 }, () => (Math.random() * 4 - 2).toFixed(3)).join(",");
    out += `${feats},${i % 2}\n`; // alternate label -> both classes present
  }
  return out;
}

export function setup() {
  const csv = makeCsv(200);
  const res = http.post(
    `${BASE}/datasets/upload`,
    { file: http.file(csv, "load.csv", "text/csv"), label_col: "label" },
    { headers: H },
  );
  check(res, { "upload 200": (r) => r.status === 200 });
  const datasetId = res.json("id");
  // Warm the /compare cache once so cachedReads measures cache hits, not the
  // first cold computation.
  http.get(`${BASE}/compare`, { headers: H });
  return { datasetId };
}

export function trainRun(data) {
  const res = http.post(
    `${BASE}/runs`,
    JSON.stringify({ dataset_id: data.datasetId, use_case: "default" }),
    { headers: { ...H, "Content-Type": "application/json" }, timeout: "60s" },
  );
  runTrend.add(res.timings.duration);
  if (check(res, { "run 200": (r) => r.status === 200 })) runsOk.add(1);
}

export function cachedRead() {
  const res = http.get(`${BASE}/compare`, { headers: H, timeout: "30s" });
  readTrend.add(res.timings.duration);
  if (check(res, { "compare 200": (r) => r.status === 200 })) readsOk.add(1);
}

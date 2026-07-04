# Failure Analysis — Frontend

A polished, read-only web console for the `ml-failure-analysis-framework` evaluation
API. Light-editorial design, hand-built SVG charts, no charting-library boilerplate.

**Stack:** Vite · React 18 · TypeScript · Tailwind CSS v4. Zero backend changes —
the dev server proxies same-origin `/api/*` calls to the FastAPI backend on `:8000`,
so the backend's lack of CORS never comes up.

## Run it

```bash
# 1) Start the backend (from the repo root), if it isn't already running:
python scripts/run_api.py --config configs/dev.yaml     # serves http://localhost:8000

# 2) Start the frontend:
cd frontend
npm install        # first time only
npm run dev        # http://localhost:5173
```

Open **http://localhost:5173**. If the API isn't reachable, the sidebar status dot
turns red and each view shows a retry-able error state.

## Views → endpoints

| View            | Endpoint(s)                          | What it shows |
|-----------------|--------------------------------------|---------------|
| Overview        | `/recommend`, `/compare`, `/quality` | Cost-optimal model + threshold, ranking, cost-vs-threshold curve, diverging slices |
| Model Comparison| `/compare`                           | Scorecard, confusion matrices (0.5 vs best threshold), calibration, threshold CI |
| Slice Analysis  | `/slices`                            | Slice × model heatmap, per-slice table, instability & single-class flags |
| Error Analysis  | `/errors`                            | Top false positives / negatives, mined failure clusters |
| Diagnostics     | `/diagnostics`                       | Winner robustness across cost scenarios, feature drift (PSI/TV) |
| Audit & Quality | `/quality`                           | Business loss reduction, slice diagnostics, runtime breakdown |

Backend health is polled via `/version`.

## Production

`npm run build` emits a static bundle to `dist/`. Since it's fully static, you can
serve it from any host — including mounting it on the FastAPI app itself so the UI
and API share one origin (no proxy, no CORS). The API base path is `/api`
(`src/api/client.ts`); change it there if you serve the API elsewhere.

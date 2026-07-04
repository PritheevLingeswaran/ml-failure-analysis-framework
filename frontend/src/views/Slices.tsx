import { useMemo, useState } from "react";
import { api, useQuery } from "../api/client";
import { Badge, Card, ErrorState, LoadingBlock, SectionHeader } from "../components/ui";
import { SliceHeatmap } from "../components/charts";
import { cost, modelLabel, num, pct, titleCase } from "../lib/format";
import type { SliceRow } from "../api/types";

type MetricKey = "recall" | "precision" | "f1" | "accuracy" | "expected_cost";
const METRICS: { key: MetricKey; label: string; kind: "metric" | "cost" }[] = [
  { key: "recall", label: "Recall", kind: "metric" },
  { key: "precision", label: "Precision", kind: "metric" },
  { key: "f1", label: "F1", kind: "metric" },
  { key: "accuracy", label: "Accuracy", kind: "metric" },
  { key: "expected_cost", label: "Expected cost", kind: "cost" },
];

const valOf = (row: SliceRow, key: MetricKey): number | null =>
  key === "expected_cost" ? row.decision.expected_cost : row.metrics[key];

export default function Slices() {
  const sl = useQuery(api.slices, "slices");
  const [metric, setMetric] = useState<MetricKey>("recall");

  if (sl.error) return <ErrorState error={sl.error} onRetry={sl.reload} />;
  if (sl.loading || !sl.data) return <LoadingBlock />;

  const rows = sl.data.slices;
  const models = Array.from(new Set(rows.map((r) => r.model ?? r.model_name)));
  const sliceNames = Array.from(new Set(rows.map((r) => r.slice_name)));
  const active = METRICS.find((m) => m.key === metric)!;

  const heatRows = sliceNames.map((slice) => {
    const byModel: Record<string, number | null> = {};
    for (const m of models) {
      const row = rows.find((r) => r.slice_name === slice && (r.model ?? r.model_name) === m);
      byModel[m] = row ? valOf(row, metric) : null;
    }
    return { slice: titleCase(slice), byModel };
  });

  const fmt = active.kind === "cost" ? (v: number | null) => cost(v, 2) : (v: number | null) => pct(v, 0);

  const unstable = rows.filter((r) => r.instability.unstable);
  const singleClass = sliceNames.filter((s) =>
    rows.some((r) => r.slice_name === s && r.metrics_validity.reasons.includes("single_class_slice")),
  );

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Slice analysis · test split"
        title="Where do aggregate metrics lie?"
        desc="A model with strong overall numbers can still fail a business-critical segment. Each cell is one slice × model; darker means stronger. Watch for cells that break rank with a model's headline score."
      />

      <Card>
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--color-ink-3)]">
            Metric
          </div>
          <div className="flex flex-wrap gap-1.5">
            {METRICS.map((m) => (
              <button
                key={m.key}
                onClick={() => setMetric(m.key)}
                className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                  metric === m.key
                    ? "bg-[var(--color-accent)] text-white"
                    : "border border-[var(--color-line)] bg-[var(--color-surface)] text-[var(--color-ink-2)] hover:border-[var(--color-accent)]"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <SliceHeatmap rows={heatRows} models={models} metricLabel={active.label} value={fmt} />
        <p className="mt-4 text-xs leading-relaxed text-[var(--color-ink-3)]">
          {active.kind === "cost"
            ? "Expected cost per slice at each model's slice-optimized threshold — negative means net-positive value under the cost matrix."
            : "Blank cells are undefined for that slice (e.g. single-class segments where precision/recall don't apply)."}
        </p>
      </Card>

      {/* Detail table */}
      <SliceTable rows={rows} models={models} />

      <div className="grid gap-8 sm:grid-cols-2">
        <Card>
          <SectionHeader eyebrow="Stability" title="Instability flags" />
          {unstable.length === 0 ? (
            <p className="text-sm text-[var(--color-ink-2)]">
              No slice tripped the instability detector — per-slice recommendations are backed by enough samples to trust.
            </p>
          ) : (
            <ul className="space-y-2 text-sm">
              {unstable.map((r, i) => (
                <li key={i} className="flex justify-between gap-2">
                  <span className="text-[var(--color-ink)]">{titleCase(r.slice_name)}</span>
                  <Badge tone="warn">{r.instability.reason}</Badge>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card>
          <SectionHeader eyebrow="Metric validity" title="Single-class slices" />
          <p className="mb-3 text-sm leading-relaxed text-[var(--color-ink-2)]">
            These segments contain only one label, so precision, recall, F1 and AUC are mathematically undefined and
            correctly reported as blank rather than fabricated.
          </p>
          <div className="flex flex-wrap gap-1.5">
            {singleClass.length ? (
              singleClass.map((s) => (
                <Badge key={s} tone="neutral">
                  {titleCase(s)}
                </Badge>
              ))
            ) : (
              <span className="text-sm text-[var(--color-ink-3)]">None.</span>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

function SliceTable({ rows, models }: { rows: SliceRow[]; models: string[] }) {
  const [model, setModel] = useState(models[0]);
  const filtered = useMemo(
    () => rows.filter((r) => (r.model ?? r.model_name) === model),
    [rows, model],
  );

  return (
    <Card pad={false}>
      <div className="flex flex-wrap items-center justify-between gap-3 px-6 pt-5">
        <h3 className="font-display text-xl">Per-slice detail</h3>
        <div className="flex gap-1.5">
          {models.map((m) => (
            <button
              key={m}
              onClick={() => setModel(m)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                model === m
                  ? "bg-[var(--color-ink)] text-white"
                  : "border border-[var(--color-line)] text-[var(--color-ink-2)] hover:border-[var(--color-ink)]"
              }`}
            >
              {modelLabel(m)}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead>
            <tr className="border-y border-[var(--color-line)] text-left text-[11px] uppercase tracking-wider text-[var(--color-ink-3)]">
              <th className="py-2.5 pl-6 pr-3 font-semibold">Slice</th>
              <th className="px-3 py-2.5 text-right font-semibold">n</th>
              <th className="px-3 py-2.5 text-right font-semibold">Acc</th>
              <th className="px-3 py-2.5 text-right font-semibold">Prec</th>
              <th className="px-3 py-2.5 text-right font-semibold">Recall</th>
              <th className="px-3 py-2.5 text-right font-semibold">F1</th>
              <th className="px-3 py-2.5 text-right font-semibold">Best thr</th>
              <th className="px-3 py-2.5 pr-6 text-right font-semibold">Exp. cost</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.slice_name} className="border-b border-[var(--color-line-2)] last:border-0 hover:bg-[var(--color-paper)]">
                <td className="py-3 pl-6 pr-3">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-[var(--color-ink)]">{titleCase(r.slice_name)}</span>
                    {r.instability.unstable && <Badge tone="warn">unstable</Badge>}
                  </div>
                  <div className="text-[11px] text-[var(--color-ink-3)]">{r.description}</div>
                </td>
                <td className="nums px-3 py-3 text-right text-[var(--color-ink-2)]">{r.count}</td>
                <td className="nums px-3 py-3 text-right text-[var(--color-ink-2)]">{pct(r.metrics.accuracy, 0)}</td>
                <td className="nums px-3 py-3 text-right text-[var(--color-ink-2)]">{pct(r.metrics.precision, 0)}</td>
                <td className="nums px-3 py-3 text-right text-[var(--color-ink-2)]">{pct(r.metrics.recall, 0)}</td>
                <td className="nums px-3 py-3 text-right text-[var(--color-ink-2)]">{num(r.metrics.f1, 2)}</td>
                <td className="nums px-3 py-3 text-right text-[var(--color-ink-2)]">{num(r.decision.best_threshold, 2)}</td>
                <td className="nums px-3 py-3 pr-6 text-right font-medium text-[var(--color-ink)]">
                  {cost(r.decision.expected_cost, 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

import { api, useQuery } from "../api/client";
import { Badge, Card, ErrorState, LoadingBlock, SectionHeader, StatTile } from "../components/ui";
import { cost, num, pct, reductionLabel, titleCase } from "../lib/format";

export default function Quality() {
  const q = useQuery(api.quality, "quality");
  if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />;
  if (q.loading || !q.data) return <LoadingBlock />;

  const Q = q.data.quality;
  const blr = Q.business_loss_reduction;
  const diag = Q.slice_diagnostics;
  const stages = Object.entries(Q.runtime.stages).sort((a, b) => b[1] - a[1]);
  const maxStage = Math.max(...stages.map(([, v]) => v), 0.001);

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow={`Run ${Q.run_id} · ${Q.split} split`}
        title="Audit trail & evaluation quality"
        desc="Every recommendation on this dashboard is reproducible and defensible. This is the provenance behind it — what was measured, how much loss the decision avoids, and where it took time."
      />

      {/* Business impact */}
      <Card className="relative overflow-hidden">
        <div className="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-[var(--color-good-soft)] opacity-70 blur-2xl" />
        <div className="relative">
          <SectionHeader
            eyebrow={`Use case · ${titleCase(blr.use_case)}`}
            title="Business loss reduction"
          />
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
            <StatTile
              label="Baseline cost"
              value={cost(blr.baseline_expected_cost, 3)}
              sub={`@ threshold ${num(blr.baseline_threshold, 2)}`}
            />
            <StatTile
              label="Optimized cost"
              value={cost(blr.optimized_expected_cost, 3)}
              tone="good"
              sub={`@ threshold ${num(blr.optimized_threshold, 2)}`}
            />
            <StatTile
              label="Reduction"
              value={reductionLabel(blr.reduction_pct)}
              tone="good"
              sub={blr.optimized_expected_cost < 0 ? "loss eliminated · net-positive" : "expected loss avoided"}
            />
            <StatTile
              label="Margin vs runner-up"
              value={num(blr.winner_vs_runnerup_cost_delta, 3)}
              tone="accent"
              sub="cost delta to 2nd place"
            />
          </div>
          <div className="mt-6">
            <div className="mb-1.5 flex justify-between text-xs text-[var(--color-ink-3)]">
              <span>optimized</span>
              <span>baseline</span>
            </div>
            <div className="relative h-3 w-full overflow-hidden rounded-full bg-[var(--color-bad-soft)]">
              <div
                className="h-full rounded-full bg-[var(--color-good)] transition-all duration-1000"
                style={{ width: `${Math.max((1 - blr.reduction_pct) * 100, 1)}%` }}
              />
            </div>
          </div>
        </div>
      </Card>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Slice diagnostics */}
        <Card>
          <SectionHeader eyebrow="Coverage" title="Slice diagnostics" />
          <div className="mb-4 grid grid-cols-3 gap-4">
            <StatTile label="Models" value={Q.models_compared} />
            <StatTile label="Diagnostics" value={diag.diagnostics_found} tone={diag.diagnostics_found ? "warn" : "good"} />
            <StatTile label="Test size" value={Q.dataset_size.split_count.toLocaleString()} sub={`of ${Q.dataset_size.total_count_if_available.toLocaleString()}`} />
          </div>
          <p className="mb-4 text-xs leading-relaxed text-[var(--color-ink-3)]">{diag.definition}</p>
          <ul className="space-y-2">
            {diag.top_diagnostics.map((t) => (
              <li
                key={t.slice_name}
                className="flex items-center justify-between gap-3 rounded-lg border border-[var(--color-line)] px-3.5 py-2.5"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-[var(--color-ink)]">{titleCase(t.slice_name)}</div>
                  <div className="text-[11px] text-[var(--color-ink-3)]">
                    n={t.count} · triggered by {t.triggered_by.replace(/_/g, " ")}
                  </div>
                </div>
                <Badge tone="warn">
                  {t.triggered_by === "metric_drop" && t.metric
                    ? `${t.metric} ↓ ${pct(t.drop_pct, 0)}`
                    : `cost ↑ ${pct(t.cost_increase_pct / 100, 0)}`}
                </Badge>
              </li>
            ))}
          </ul>
          <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1 border-t border-[var(--color-line)] pt-3 text-[11px] text-[var(--color-ink-3)]">
            {Object.entries(diag.thresholds).map(([k, v]) => (
              <span key={k}>
                {k.replace(/_/g, " ")} <span className="nums text-[var(--color-ink-2)]">{v}</span>
              </span>
            ))}
          </div>
        </Card>

        {/* Runtime */}
        <Card>
          <SectionHeader eyebrow="Performance" title="Runtime breakdown" />
          <div className="mb-5">
            <StatTile label="Total evaluation time" value={num(Q.runtime.total_sec, 2)} unit="sec" tone="accent" />
          </div>
          <div className="space-y-2.5">
            {stages.map(([stage, secs]) => (
              <div key={stage} className="flex items-center gap-3">
                <div className="w-36 shrink-0 truncate text-sm text-[var(--color-ink-2)]">
                  {stage.replace(/_/g, " ")}
                </div>
                <div className="h-4 flex-1 overflow-hidden rounded bg-[var(--color-line-2)]">
                  <div
                    className="h-full rounded bg-[var(--color-accent)] transition-all duration-700"
                    style={{ width: `${Math.max((secs / maxStage) * 100, 1.5)}%`, opacity: 0.55 + (secs / maxStage) * 0.45 }}
                  />
                </div>
                <div className="nums w-14 shrink-0 text-right text-xs text-[var(--color-ink-2)]">{num(secs, 2)}s</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

import { api, useQuery } from "../api/client";
import { Badge, Card, Dot, ErrorState, LoadingBlock, SectionHeader, StatTile } from "../components/ui";
import { CostCurveChart } from "../components/charts";
import { cost, modelColor, modelLabel, num, reductionLabel, titleCase } from "../lib/format";

export default function Overview() {
  const rec = useQuery(api.recommend, "recommend");
  const cmp = useQuery(api.compare, "compare");
  const qual = useQuery(api.quality, "quality");

  if (rec.error) return <ErrorState error={rec.error} onRetry={rec.reload} />;
  if (rec.loading || cmp.loading || qual.loading || !rec.data || !cmp.data || !qual.data)
    return <LoadingBlock />;

  const r = rec.data;
  const ranking = r.rationale.ranking;
  const winner = ranking[0];
  const runnerUp = ranking[1];
  const blr = qual.data.quality.business_loss_reduction;
  const maxCost = Math.max(...ranking.map((m) => m.expected_cost));
  const minCost = Math.min(0, ...ranking.map((m) => m.expected_cost));
  const span = maxCost - minCost || 1;

  const curves = Object.entries(cmp.data.per_model).map(([model, pm]) => ({ model, curve: pm.cost_curve }));
  const markers = ranking.map((m) => ({ model: m.model, threshold: m.best_threshold }));

  const riskySlices = (r.per_slice_recommendations ?? [])
    .filter((s) => s.recommended_model !== r.recommended_model || s.expected_cost > winner.expected_cost + 0.15)
    .slice(0, 4);

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Test split · cost-optimal decision"
        title="Which model should we ship, and at what threshold?"
        desc={`Ranked by expected business cost under the “${titleCase(r.rationale.use_case)}” cost matrix — not by raw accuracy. The winner minimizes expected loss, which is what production actually pays for.`}
      />

      {/* Hero recommendation */}
      <Card className="relative overflow-hidden">
        <div className="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-[var(--color-accent-soft)] opacity-60 blur-2xl" />
        <div className="relative grid gap-8 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <Badge tone="accent">Recommended to ship</Badge>
            <div className="mt-4 flex items-baseline gap-3">
              <Dot color={modelColor(r.recommended_model)} />
              <h3 className="font-display text-4xl text-[var(--color-ink)]">{modelLabel(r.recommended_model)}</h3>
            </div>
            <p className="mt-3 max-w-md text-sm leading-relaxed text-[var(--color-ink-2)]">{r.rationale.note}</p>

            <div className="mt-7 grid grid-cols-2 gap-6 sm:grid-cols-3">
              <StatTile
                label="Decision threshold"
                value={num(r.recommended_threshold, 2)}
                tone="accent"
                sub="cost-optimized cut-point"
              />
              <StatTile
                label="Expected cost"
                value={cost(winner.expected_cost, 3)}
                tone="good"
                sub={`vs runner-up +${num(blr.winner_vs_runnerup_cost_delta, 3)}`}
              />
              <StatTile
                label="Loss reduction"
                value={reductionLabel(blr.reduction_pct)}
                tone="good"
                sub={
                  blr.optimized_expected_cost < 0
                    ? "optimized is net-positive"
                    : `vs baseline @ ${num(blr.baseline_threshold, 2)}`
                }
              />
            </div>
          </div>

          {/* Ranking bars */}
          <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-paper)] p-5">
            <div className="mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--color-ink-3)]">
              Expected cost by model — lower is better
            </div>
            <div className="space-y-4">
              {ranking.map((m, i) => {
                const zero = ((0 - minCost) / span) * 100;
                const w = (Math.abs(m.expected_cost) / span) * 100;
                const left = m.expected_cost >= 0 ? zero : zero - w;
                const isWinner = m.model === r.recommended_model;
                return (
                  <div key={m.model}>
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span className="flex items-center gap-1.5 font-medium text-[var(--color-ink)]">
                        <Dot color={modelColor(m.model, i)} />
                        {modelLabel(m.model)}
                        {isWinner && <Badge tone="good" className="ml-1">winner</Badge>}
                      </span>
                      <span className="nums text-[var(--color-ink-2)]">{cost(m.expected_cost, 3)}</span>
                    </div>
                    <div className="relative h-2.5 w-full rounded-full bg-[var(--color-line-2)]">
                      <div
                        className="absolute inset-y-0 rounded-full transition-all duration-700"
                        style={{
                          left: `${left}%`,
                          width: `${Math.max(w, 1.5)}%`,
                          background: modelColor(m.model, i),
                          opacity: isWinner ? 1 : 0.5,
                        }}
                      />
                      {minCost < 0 && (
                        <div className="absolute inset-y-0 w-px bg-[var(--color-ink-3)]" style={{ left: `${zero}%` }} />
                      )}
                    </div>
                    <div className="mt-1 flex gap-3 text-[11px] text-[var(--color-ink-3)]">
                      <span>thr {num(m.best_threshold, 2)}</span>
                      <span>F1 {num(m.f1_at_best, 2)}</span>
                      <span>AUC {num(m.roc_auc, 2)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </Card>

      {/* Cost curve */}
      <div className="grid gap-8 lg:grid-cols-[1.6fr_1fr]">
        <Card>
          <SectionHeader
            eyebrow="Decision surface"
            title="Cost vs. threshold"
            desc="Every model's expected cost across all thresholds. Rings mark each model's cost-optimal cut-point — thresholds define decisions, AUC does not."
          />
          <CostCurveChart curves={curves} markers={markers} />
        </Card>

        <Card>
          <SectionHeader eyebrow="Watchlist" title="Slices that diverge" />
          {riskySlices.length === 0 ? (
            <p className="text-sm text-[var(--color-ink-2)]">
              No slice materially disagrees with the global recommendation.
            </p>
          ) : (
            <ul className="space-y-3">
              {riskySlices.map((s) => (
                <li
                  key={s.slice_name}
                  className="flex items-start justify-between gap-3 rounded-lg border border-[var(--color-line)] px-3.5 py-3"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-[var(--color-ink)]">
                      {titleCase(s.slice_name)}
                    </div>
                    <div className="mt-0.5 text-xs text-[var(--color-ink-3)]">
                      prefers <span className="font-medium text-[var(--color-ink-2)]">{modelLabel(s.recommended_model)}</span>{" "}
                      @ {num(s.recommended_threshold, 2)} · n={s.count}
                    </div>
                  </div>
                  <span className="nums shrink-0 text-sm text-[var(--color-ink-2)]">{cost(s.expected_cost, 2)}</span>
                </li>
              ))}
            </ul>
          )}
          {runnerUp && (
            <p className="mt-4 border-t border-[var(--color-line)] pt-4 text-xs leading-relaxed text-[var(--color-ink-3)]">
              Runner-up <span className="font-medium text-[var(--color-ink-2)]">{modelLabel(runnerUp.model)}</span> costs{" "}
              <span className="nums">{cost(runnerUp.expected_cost, 3)}</span> — a{" "}
              <span className="nums">{num(runnerUp.expected_cost - winner.expected_cost, 3)}</span> gap. Shift the cost
              matrix and this ordering can flip.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}

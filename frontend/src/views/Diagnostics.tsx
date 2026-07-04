import { api, useQuery } from "../api/client";
import { Badge, Card, Dot, ErrorState, LoadingBlock, SectionHeader, StatTile } from "../components/ui";
import { Donut } from "../components/charts";
import { modelColor, modelLabel, num, pct } from "../lib/format";

export default function Diagnostics() {
  const d = useQuery(api.diagnostics, "diagnostics");
  if (d.error) return <ErrorState error={d.error} onRetry={d.reload} />;
  if (d.loading || !d.data) return <LoadingBlock />;

  const { drift, cost_sensitivity } = d.data;
  const features = [...drift.all_features].sort((a, b) => b.score - a.score);
  const maxScore = Math.max(...features.map((f) => f.score), 0.001);

  // cost-sensitivity grid axes
  const fps = Array.from(new Set(cost_sensitivity.scenarios.map((s) => s.scenario.fp_multiplier))).sort((a, b) => a - b);
  const fns = Array.from(new Set(cost_sensitivity.scenarios.map((s) => s.scenario.fn_multiplier))).sort((a, b) => a - b);
  const winnerAt = (fp: number, fn: number) =>
    cost_sensitivity.scenarios.find((s) => s.scenario.fp_multiplier === fp && s.scenario.fn_multiplier === fn)?.winner;

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Diagnostics · test split"
        title="Is the recommendation robust — and is the data still in-distribution?"
        desc="Two questions that decide whether a decision survives contact with production: does the winner hold as error costs shift, and have the input features drifted from training?"
      />

      {/* Cost sensitivity */}
      <Card>
        <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr]">
          <div>
            <SectionHeader eyebrow="Robustness" title="Winner across cost scenarios" />
            <div className="overflow-x-auto">
              <table className="border-separate border-spacing-1">
                <thead>
                  <tr>
                    <th className="p-1 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
                      FN&nbsp;× ╲ FP&nbsp;×
                    </th>
                    {fps.map((fp) => (
                      <th key={fp} className="nums p-1 text-center text-[11px] text-[var(--color-ink-2)]">
                        {fp}×
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {fns.map((fn) => (
                    <tr key={fn}>
                      <td className="nums p-1 text-right text-[11px] text-[var(--color-ink-2)]">{fn}×</td>
                      {fps.map((fp) => {
                        const w = winnerAt(fp, fn);
                        return (
                          <td key={fp} className="p-0">
                            <div
                              className="flex h-10 w-16 items-center justify-center rounded-md text-[11px] font-semibold text-white"
                              style={{ background: w ? modelColor(w) : "var(--color-line-2)" }}
                              title={w ? `FP×${fp}, FN×${fn} → ${modelLabel(w)}` : ""}
                            >
                              {w ?? "—"}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-4 max-w-xl text-xs leading-relaxed text-[var(--color-ink-3)]">
              Each cell re-optimizes the decision under a scaled cost matrix (false-positive cost × FP, false-negative
              cost × FN). A single color across the grid means the recommendation is insensitive to how you price errors.
            </p>
          </div>

          <div className="flex flex-col gap-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
              Win rate across {cost_sensitivity.scenarios.length} scenarios
            </div>
            {cost_sensitivity.winner_robustness.map((w) => (
              <div key={w.model} className="flex items-center gap-4 rounded-lg border border-[var(--color-line)] p-4">
                <div className="relative">
                  <Donut value={w.win_rate} color={modelColor(w.model)} />
                  <span className="nums absolute inset-0 flex items-center justify-center text-sm font-semibold text-[var(--color-ink)]">
                    {pct(w.win_rate, 0)}
                  </span>
                </div>
                <div>
                  <div className="flex items-center gap-1.5 font-medium text-[var(--color-ink)]">
                    <Dot color={modelColor(w.model)} />
                    {modelLabel(w.model)}
                  </div>
                  <div className="mt-0.5 text-xs text-[var(--color-ink-3)]">
                    wins {w.win_count} / {cost_sensitivity.scenarios.length}
                  </div>
                  {w.win_rate === 1 && <Badge tone="good" className="mt-2">unconditional winner</Badge>}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Drift */}
      <Card>
        <SectionHeader
          eyebrow="Distribution shift"
          title="Feature drift (train → test)"
          desc="PSI for numeric features, total-variation distance for categoricals. Drift doesn't invalidate the model, but a drifted feature is where silent degradation starts."
        />
        <div className="mb-6 grid grid-cols-2 gap-6 sm:grid-cols-3">
          <StatTile label="Features checked" value={drift.num_features_checked} />
          <StatTile
            label="Drifted"
            value={drift.num_drifted}
            tone={drift.num_drifted > 0 ? "warn" : "good"}
            sub={drift.num_drifted > 0 ? "above threshold" : "all stable"}
          />
          <StatTile
            label="Top signal"
            value={drift.top_drifted[0]?.feature ?? "—"}
            sub={drift.top_drifted[0] ? `${drift.top_drifted[0].metric} ${num(drift.top_drifted[0].score, 2)}` : undefined}
            tone={drift.top_drifted.length ? "warn" : "neutral"}
          />
        </div>
        <div className="space-y-1.5">
          {features.map((f) => (
            <div key={f.feature} className="flex items-center gap-3">
              <div className="w-28 shrink-0 truncate text-sm text-[var(--color-ink)]">{f.feature}</div>
              <div className="h-4 flex-1 overflow-hidden rounded bg-[var(--color-line-2)]">
                <div
                  className="h-full rounded transition-all duration-700"
                  style={{
                    width: `${Math.max((f.score / maxScore) * 100, 1.5)}%`,
                    background: f.drifted ? "var(--color-warn)" : "var(--color-accent)",
                    opacity: f.drifted ? 1 : 0.4,
                  }}
                />
              </div>
              <div className="nums w-16 shrink-0 text-right text-xs text-[var(--color-ink-2)]">{num(f.score, 3)}</div>
              <div className="w-14 shrink-0 text-[10px] uppercase tracking-wider text-[var(--color-ink-3)]">
                {f.metric === "psi" ? "PSI" : "TV"}
              </div>
              {f.drifted && <Badge tone="warn">drift</Badge>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

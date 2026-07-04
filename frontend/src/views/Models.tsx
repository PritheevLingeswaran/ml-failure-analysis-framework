import { useState } from "react";
import { api, useQuery } from "../api/client";
import { Badge, Card, Dot, ErrorState, LoadingBlock, SectionHeader } from "../components/ui";
import { ConfusionMatrix, MetricBar } from "../components/charts";
import { cost, modelColor, modelLabel, num, pct } from "../lib/format";

const METRICS: { key: "roc_auc" | "pr_auc" | "f1_at_best"; label: string }[] = [
  { key: "roc_auc", label: "ROC-AUC" },
  { key: "pr_auc", label: "PR-AUC" },
  { key: "f1_at_best", label: "F1 @ best" },
];

export default function Models() {
  const cmp = useQuery(api.compare, "compare");
  const [selected, setSelected] = useState<string | null>(null);

  if (cmp.error) return <ErrorState error={cmp.error} onRetry={cmp.reload} />;
  if (cmp.loading || !cmp.data) return <LoadingBlock />;

  const { ranking, winner } = cmp.data.summary;
  const active = selected ?? winner;
  const pm = cmp.data.per_model[active];

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Comparison · test split"
        title="Model scorecard"
        desc="Aggregate metrics rank models differently depending on what you optimize. Expected cost is the tie-breaker that reflects real error asymmetry; calibration (Brier, ECE) governs whether the probabilities can be trusted at a threshold."
      />

      {/* Comparison table */}
      <Card pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-left text-[11px] uppercase tracking-wider text-[var(--color-ink-3)]">
                <th className="py-3 pl-6 pr-3 font-semibold">Model</th>
                <th className="px-3 py-3 text-right font-semibold">Exp. cost</th>
                <th className="px-3 py-3 text-right font-semibold">Best thr</th>
                <th className="px-3 py-3 text-right font-semibold">ROC-AUC</th>
                <th className="px-3 py-3 text-right font-semibold">PR-AUC</th>
                <th className="px-3 py-3 text-right font-semibold">F1</th>
                <th className="px-3 py-3 text-right font-semibold">Brier</th>
                <th className="px-3 py-3 pr-6 text-right font-semibold">ECE</th>
              </tr>
            </thead>
            <tbody>
              {ranking.map((m, i) => {
                const isActive = m.model === active;
                return (
                  <tr
                    key={m.model}
                    onClick={() => setSelected(m.model)}
                    className={`cursor-pointer border-b border-[var(--color-line-2)] transition last:border-0 hover:bg-[var(--color-paper)] ${
                      isActive ? "bg-[var(--color-accent-soft)]/40" : ""
                    }`}
                  >
                    <td className="py-3.5 pl-6 pr-3">
                      <span className="flex items-center gap-2 font-medium text-[var(--color-ink)]">
                        <Dot color={modelColor(m.model, i)} />
                        {modelLabel(m.model)}
                        {m.model === winner && <Badge tone="good">winner</Badge>}
                      </span>
                    </td>
                    <td className="nums px-3 py-3.5 text-right font-semibold text-[var(--color-ink)]">
                      {cost(m.expected_cost, 3)}
                    </td>
                    <td className="nums px-3 py-3.5 text-right text-[var(--color-ink-2)]">{num(m.best_threshold, 2)}</td>
                    <td className="nums px-3 py-3.5 text-right text-[var(--color-ink-2)]">{num(m.roc_auc, 3)}</td>
                    <td className="nums px-3 py-3.5 text-right text-[var(--color-ink-2)]">{num(m.pr_auc, 3)}</td>
                    <td className="nums px-3 py-3.5 text-right text-[var(--color-ink-2)]">{num(m.f1_at_best, 3)}</td>
                    <td className="nums px-3 py-3.5 text-right text-[var(--color-ink-2)]">{num(m.brier, 3)}</td>
                    <td className="nums px-3 py-3.5 pr-6 text-right text-[var(--color-ink-2)]">{num(m.ece, 3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="px-6 py-3 text-xs text-[var(--color-ink-3)]">
          Select a row to inspect its confusion matrix and calibration below.
        </div>
      </Card>

      {/* Detail for selected model */}
      {pm && (
        <div className="grid gap-8 lg:grid-cols-3">
          <Card>
            <div className="mb-4 flex items-center gap-2">
              <Dot color={modelColor(active)} />
              <h3 className="font-display text-xl">{modelLabel(active)}</h3>
            </div>
            <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
              Confusion @ threshold 0.5
            </div>
            <div className="mt-3">
              <ConfusionMatrix c={pm.overall.confusion} />
            </div>
            <div className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
              <Row label="Accuracy" v={pct(pm.overall.accuracy)} />
              <Row label="Precision" v={pct(pm.overall.precision)} />
              <Row label="Recall" v={pct(pm.overall.recall)} />
              <Row label="F1" v={num(pm.overall.f1)} />
            </div>
          </Card>

          <Card>
            <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
              Confusion @ best threshold {num(pm.best.threshold, 2)}
            </div>
            <div className="mt-3">
              <ConfusionMatrix c={pm.best.metrics.confusion} />
            </div>
            <div className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
              <Row label="Precision" v={pct(pm.best.metrics.precision)} />
              <Row label="Recall" v={pct(pm.best.metrics.recall)} />
              <Row label="Exp. cost" v={cost(pm.best.expected_cost, 3)} />
              <Row label="Accuracy" v={pct(pm.best.metrics.accuracy)} />
            </div>
            <p className="mt-4 text-xs leading-relaxed text-[var(--color-ink-3)]">
              Lowering the threshold to {num(pm.best.threshold, 2)} trades precision for recall — correct when a missed
              positive costs far more than a false alarm.
            </p>
          </Card>

          <Card>
            <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
              Discrimination & calibration
            </div>
            <div className="mt-4 space-y-4">
              {METRICS.map((mt) => {
                const row = ranking.find((r) => r.model === active)!;
                const v = row[mt.key];
                return (
                  <div key={mt.key}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span className="text-[var(--color-ink-2)]">{mt.label}</span>
                      <span className="nums text-[var(--color-ink)]">{num(v, 3)}</span>
                    </div>
                    <MetricBar value={v} color={modelColor(active)} />
                  </div>
                );
              })}
            </div>
            <div className="mt-6 space-y-2.5 border-t border-[var(--color-line)] pt-4 text-sm">
              <Row label="Calibration" v={pm.calibration.method} />
              <Row
                label="Threshold 90% CI"
                v={`${num(pm.threshold_uncertainty.ci[0], 2)} – ${num(pm.threshold_uncertainty.ci[1], 2)}`}
              />
              <Row label="Mean bootstrap thr" v={num(pm.threshold_uncertainty.mean_threshold, 3)} />
              <Row label="Test examples" v={pm.overall.count.toLocaleString()} />
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

function Row({ label, v }: { label: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-[var(--color-ink-3)]">{label}</span>
      <span className="nums text-[var(--color-ink)]">{v}</span>
    </div>
  );
}

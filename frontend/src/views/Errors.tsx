import { useState } from "react";
import { api, useQuery } from "../api/client";
import { Badge, Card, EmptyNote, ErrorState, LoadingBlock, SectionHeader } from "../components/ui";
import { modelLabel, num } from "../lib/format";
import type { ErrorExample } from "../api/types";

export default function Errors() {
  const err = useQuery(api.errors, "errors");
  const [model, setModel] = useState<string | null>(null);

  if (err.error) return <ErrorState error={err.error} onRetry={err.reload} />;
  if (err.loading || !err.data) return <LoadingBlock />;

  const models = Object.keys(err.data.top_false_positives);
  const active = model ?? models[0];
  const fps = err.data.top_false_positives[active] ?? [];
  const fns = err.data.top_false_negatives[active] ?? [];
  const clusters = err.data.clusters[active];

  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Error analysis · test split"
        title="What is the model getting wrong, and why?"
        desc="The highest-confidence mistakes and the clusters they fall into. Confident errors are the expensive ones — they signal systematic blind spots, not random noise."
        right={
          <div className="flex gap-1.5">
            {models.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                  active === m
                    ? "bg-[var(--color-ink)] text-white"
                    : "border border-[var(--color-line)] text-[var(--color-ink-2)] hover:border-[var(--color-ink)]"
                }`}
              >
                {modelLabel(m)}
              </button>
            ))}
          </div>
        }
      />

      <div className="grid gap-8 lg:grid-cols-2">
        <ErrorColumn
          title="Top false positives"
          tone="bad"
          hint="Predicted positive, actually negative — false alarms."
          items={fps}
        />
        <ErrorColumn
          title="Top false negatives"
          tone="warn"
          hint="Predicted negative, actually positive — missed cases, usually the costlier error."
          items={fns}
        />
      </div>

      <Card>
        <SectionHeader
          eyebrow="Failure clustering"
          title="Representative failure modes"
          desc={
            clusters?.available
              ? `${clusters.n_clusters} clusters mined from this model's errors, ranked by size. Shared vocabulary reveals the systematic patterns behind the mistakes.`
              : undefined
          }
        />
        {!clusters?.available ? (
          <EmptyNote>Clustering was not available for this model.</EmptyNote>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {clusters.summaries.map((c) => (
              <div key={c.cluster} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-paper)] p-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
                    Cluster {c.cluster}
                  </span>
                  <div className="flex gap-1.5">
                    {c.fp > 0 && <Badge tone="bad">{c.fp} FP</Badge>}
                    {c.fn > 0 && <Badge tone="warn">{c.fn} FN</Badge>}
                  </div>
                </div>
                <div className="mb-3 flex flex-wrap gap-1">
                  {c.top_terms
                    .filter((t) => !/^x{6,}$/.test(t.trim()))
                    .slice(0, 5)
                    .map((t, i) => (
                      <span
                        key={i}
                        className="rounded bg-[var(--color-surface)] px-1.5 py-0.5 text-[11px] text-[var(--color-ink-2)] ring-1 ring-[var(--color-line)]"
                      >
                        {t.length > 24 ? t.slice(0, 24) + "…" : t}
                      </span>
                    ))}
                </div>
                <div className="space-y-1.5">
                  {c.examples.slice(0, 2).map((ex, i) => (
                    <p key={i} className="truncate text-xs text-[var(--color-ink-3)]" title={ex.text}>
                      <span className="nums mr-1.5 text-[var(--color-ink-2)]">{num(ex.y_score, 2)}</span>
                      {cleanText(ex.text)}
                    </p>
                  ))}
                </div>
                <div className="mt-3 text-[11px] text-[var(--color-ink-3)]">{c.count} errors in cluster</div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function ErrorColumn({
  title,
  hint,
  tone,
  items,
}: {
  title: string;
  hint: string;
  tone: "bad" | "warn";
  items: ErrorExample[];
}) {
  return (
    <Card>
      <div className="mb-1 flex items-center gap-2">
        <h3 className="font-display text-xl">{title}</h3>
        <Badge tone={tone}>{items.length}</Badge>
      </div>
      <p className="mb-4 text-xs text-[var(--color-ink-3)]">{hint}</p>
      {items.length === 0 ? (
        <EmptyNote>No examples.</EmptyNote>
      ) : (
        <ul className="divide-y divide-[var(--color-line-2)]">
          {items.slice(0, 8).map((ex) => (
            <li key={ex.id} className="flex items-start gap-3 py-2.5">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-[var(--color-ink)]" title={ex.text}>
                  {cleanText(ex.text)}
                </p>
                <div className="mt-0.5 text-[11px] text-[var(--color-ink-3)]">{ex.id}</div>
              </div>
              <div className="shrink-0 text-right">
                <div className="nums text-sm text-[var(--color-ink)]">{num(ex.confidence, 2)}</div>
                <div className="text-[10px] uppercase tracking-wider text-[var(--color-ink-3)]">conf</div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/** Collapse the synthetic-dataset "xxxx…" padding so real tokens are readable. */
function cleanText(t: string): string {
  const cleaned = t.replace(/x{6,}/gi, "▒").trim();
  return cleaned.length ? cleaned : t;
}

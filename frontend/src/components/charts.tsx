import { useId, useMemo, useState } from "react";
import { modelColor, modelLabel, num } from "../lib/format";
import type { CostCurve } from "../api/types";

const INK3 = "#74747e";
const LINE = "#e7e4dd";

/* ── Sparkline ─────────────────────────────────────────────── */
export function Sparkline({
  values,
  color = "#4f46e5",
  width = 132,
  height = 34,
}: {
  values: number[];
  color?: string;
  width?: number;
  height?: number;
}) {
  if (!values.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = width / (values.length - 1 || 1);
  const pts = values.map((v, i) => [i * step, height - ((v - min) / span) * (height - 4) - 2]);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const [mx, my] = pts[pts.findIndex((_, i) => values[i] === min)];
  return (
    <svg width={width} height={height} className="overflow-visible">
      <path d={d} fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={mx} cy={my} r={2.5} fill={color} />
    </svg>
  );
}

/* ── Cost vs threshold, multi-model ────────────────────────── */
export function CostCurveChart({
  curves,
  markers,
}: {
  curves: { model: string; curve: CostCurve }[];
  markers: { model: string; threshold: number }[];
}) {
  const uid = useId();
  const [hover, setHover] = useState<number | null>(null);
  const W = 720;
  const H = 300;
  const P = { t: 16, r: 16, b: 34, l: 44 };
  const iw = W - P.l - P.r;
  const ih = H - P.t - P.b;

  const allT = curves.flatMap((c) => c.curve.thresholds);
  const allC = curves.flatMap((c) => c.curve.expected_costs);
  const tMin = Math.min(...allT);
  const tMax = Math.max(...allT);
  const cMin = Math.min(0, ...allC);
  const cMax = Math.max(...allC);
  const x = (t: number) => P.l + ((t - tMin) / (tMax - tMin || 1)) * iw;
  const y = (c: number) => P.t + ih - ((c - cMin) / (cMax - cMin || 1)) * ih;

  const yTicks = 4;
  const commonT = curves[0]?.curve.thresholds ?? [];
  const hoverIdx = hover;

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxWidth: "100%" }}>
        {/* gridlines + y labels */}
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const val = cMin + ((cMax - cMin) * i) / yTicks;
          const yy = y(val);
          return (
            <g key={i}>
              <line x1={P.l} x2={W - P.r} y1={yy} y2={yy} stroke={LINE} strokeWidth={1} />
              <text x={P.l - 8} y={yy + 3} textAnchor="end" fontSize={10} fill={INK3} className="nums">
                {val.toFixed(1)}
              </text>
            </g>
          );
        })}
        {/* x labels */}
        {[tMin, (tMin + tMax) / 2, tMax].map((t, i) => (
          <text
            key={i}
            x={x(t)}
            y={H - 12}
            textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
            fontSize={10}
            fill={INK3}
            className="nums"
          >
            {t.toFixed(2)}
          </text>
        ))}
        <text x={P.l} y={H - 12} fontSize={10} fill={INK3} opacity={0} />
        {/* zero line */}
        {cMin < 0 && (
          <line x1={P.l} x2={W - P.r} y1={y(0)} y2={y(0)} stroke="#c9c4ba" strokeDasharray="3 3" strokeWidth={1} />
        )}
        {/* curves */}
        {curves.map(({ model, curve }, ci) => {
          const color = modelColor(model, ci);
          const d = curve.thresholds
            .map((t, i) => `${i ? "L" : "M"}${x(t).toFixed(1)} ${y(curve.expected_costs[i]).toFixed(1)}`)
            .join(" ");
          const len = 2000;
          return (
            <path
              key={model}
              d={d}
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="draw"
              style={{ ["--len" as string]: len, strokeDasharray: len, animationDelay: `${ci * 120}ms` }}
            />
          );
        })}
        {/* best-threshold markers */}
        {markers.map(({ model, threshold }, ci) => {
          const c = curves.find((x) => x.model === model);
          if (!c) return null;
          const idx = c.curve.thresholds.reduce(
            (best, t, i) => (Math.abs(t - threshold) < Math.abs(c.curve.thresholds[best] - threshold) ? i : best),
            0,
          );
          const cx = x(c.curve.thresholds[idx]);
          const cy = y(c.curve.expected_costs[idx]);
          return (
            <g key={`${uid}-m-${model}`}>
              <line x1={cx} x2={cx} y1={cy} y2={P.t + ih} stroke={modelColor(model, ci)} strokeWidth={1} opacity={0.25} />
              <circle cx={cx} cy={cy} r={4} fill="#fff" stroke={modelColor(model, ci)} strokeWidth={2} />
            </g>
          );
        })}
        {/* hover interaction layer */}
        {commonT.map((t, i) => (
          <rect
            key={i}
            x={x(t) - (iw / commonT.length) / 2}
            y={P.t}
            width={iw / commonT.length}
            height={ih}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {hoverIdx != null && commonT[hoverIdx] != null && (
          <g pointerEvents="none">
            <line
              x1={x(commonT[hoverIdx])}
              x2={x(commonT[hoverIdx])}
              y1={P.t}
              y2={P.t + ih}
              stroke="#1a1a1e"
              strokeWidth={1}
              opacity={0.15}
            />
            {curves.map((c, ci) => {
              const cc = c.curve.expected_costs[hoverIdx];
              if (cc == null) return null;
              return <circle key={ci} cx={x(commonT[hoverIdx])} cy={y(cc)} r={3} fill={modelColor(c.model, ci)} />;
            })}
          </g>
        )}
      </svg>
      {/* legend + hover readout */}
      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1 text-xs">
        {curves.map((c, ci) => (
          <span key={c.model} className="inline-flex items-center gap-1.5 text-[var(--color-ink-2)]">
            <span className="inline-block h-[3px] w-4 rounded-full" style={{ background: modelColor(c.model, ci) }} />
            {modelLabel(c.model)}
            {hoverIdx != null && (
              <span className="nums ml-1 text-[var(--color-ink-3)]">{num(c.curve.expected_costs[hoverIdx], 2)}</span>
            )}
          </span>
        ))}
        <span className="ml-auto text-[var(--color-ink-3)]">
          {hoverIdx != null ? `threshold ${num(commonT[hoverIdx], 2)}` : "hover for detail"}
        </span>
      </div>
    </div>
  );
}

/* ── Confusion matrix ──────────────────────────────────────── */
export function ConfusionMatrix({
  c,
  compact = false,
}: {
  c: { tn: number; fp: number; fn: number; tp: number };
  compact?: boolean;
}) {
  const total = c.tn + c.fp + c.fn + c.tp || 1;
  const cells = [
    { k: "TN", v: c.tn, correct: true },
    { k: "FP", v: c.fp, correct: false },
    { k: "FN", v: c.fn, correct: false },
    { k: "TP", v: c.tp, correct: true },
  ];
  return (
    <div className={compact ? "w-full max-w-[210px]" : "w-full max-w-[260px]"}>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-[10px] text-[var(--color-ink-3)]">
        <div />
        <div className="pb-1 text-center font-semibold uppercase tracking-wider">Pred 0</div>
        <div className="pb-1 text-center font-semibold uppercase tracking-wider">Pred 1</div>
      </div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1">
        <div className="flex items-center pr-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
          <span className="[writing-mode:vertical-rl] rotate-180">Actual</span>
        </div>
        <div className="grid grid-cols-2 gap-1">
          {cells.map((cell) => {
            const frac = cell.v / total;
            const bg = cell.correct
              ? `rgba(15,118,110,${0.06 + frac * 0.5})`
              : `rgba(185,28,28,${0.06 + frac * 0.7})`;
            const fg = cell.correct ? "var(--color-good)" : "var(--color-bad)";
            return (
              <div
                key={cell.k}
                className="flex aspect-[1.6] flex-col items-center justify-center rounded-md"
                style={{ background: bg }}
              >
                <span className="nums text-lg leading-none" style={{ color: fg }}>
                  {cell.v}
                </span>
                <span className="mt-0.5 text-[9px] font-semibold uppercase tracking-wider" style={{ color: fg }}>
                  {cell.k}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ── Horizontal proportion bar ─────────────────────────────── */
export function MetricBar({ value, color = "#4f46e5" }: { value: number | null; color?: string }) {
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-line-2)]">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${v * 100}%`, background: value == null ? "#d9d5cc" : color }}
      />
    </div>
  );
}

/* ── Slice heatmap (slices × models) ───────────────────────── */
export function SliceHeatmap({
  rows,
  models,
  metricLabel,
  value,
}: {
  rows: { slice: string; byModel: Record<string, number | null> }[];
  models: string[];
  metricLabel: string;
  value: (v: number | null) => string;
}) {
  const flat = rows.flatMap((r) => models.map((m) => r.byModel[m])).filter((v): v is number => v != null);
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 1;
  const scale = (v: number) => (max === min ? 0.5 : (v - min) / (max - min));
  // sequential indigo scale, higher = stronger
  const bg = (v: number | null) =>
    v == null ? "var(--color-line-2)" : `rgba(79,70,229,${0.08 + scale(v) * 0.72})`;
  const fg = (v: number | null) => (v == null ? "var(--color-ink-3)" : scale(v) > 0.55 ? "#fff" : "var(--color-accent-ink)");

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1">
        <thead>
          <tr>
            <th className="sticky left-0 bg-[var(--color-surface)] pb-2 text-left text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink-3)]">
              {metricLabel}
            </th>
            {models.map((m) => (
              <th
                key={m}
                className="min-w-[92px] pb-2 text-center text-[11px] font-semibold text-[var(--color-ink-2)]"
              >
                {modelLabel(m)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.slice}>
              <td className="sticky left-0 z-10 max-w-[220px] truncate bg-[var(--color-surface)] py-1 pr-3 text-sm text-[var(--color-ink)]">
                {r.slice}
              </td>
              {models.map((m) => {
                const v = r.byModel[m];
                return (
                  <td key={m} className="p-0">
                    <div
                      className="nums flex h-9 items-center justify-center rounded-md text-xs font-medium"
                      style={{ background: bg(v), color: fg(v) }}
                      title={`${r.slice} · ${modelLabel(m)} · ${value(v)}`}
                    >
                      {value(v)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Donut for a single 0..1 rate ──────────────────────────── */
export function Donut({ value, color = "#4f46e5", size = 72 }: { value: number; color?: string; size?: number }) {
  const r = (size - 8) / 2;
  const circ = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={LINE} strokeWidth={6} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={6}
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={circ * (1 - value)}
        style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(0.16,1,0.3,1)" }}
      />
    </svg>
  );
}

export function useMaxCost(values: number[]) {
  return useMemo(() => Math.max(...values.map((v) => Math.abs(v)), 0.001), [values]);
}

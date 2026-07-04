import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
  pad = true,
  style,
}: {
  children: ReactNode;
  className?: string;
  pad?: boolean;
  style?: React.CSSProperties;
}) {
  return (
    <section
      style={style}
      className={`rounded-xl border border-[var(--color-line)] bg-[var(--color-surface)] shadow-[0_1px_2px_rgba(20,20,30,0.03),0_8px_24px_-16px_rgba(20,20,30,0.10)] ${
        pad ? "p-5 sm:p-6" : ""
      } ${className}`}
    >
      {children}
    </section>
  );
}

export function SectionHeader({
  eyebrow,
  title,
  desc,
  right,
}: {
  eyebrow?: string;
  title: string;
  desc?: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-5 flex items-end justify-between gap-4">
      <div>
        {eyebrow && (
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--color-ink-3)]">
            {eyebrow}
          </div>
        )}
        <h2 className="font-display text-2xl text-[var(--color-ink)] sm:text-[26px]">{title}</h2>
        {desc && <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-[var(--color-ink-2)]">{desc}</p>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}

type Tone = "neutral" | "accent" | "good" | "warn" | "bad";
const TONES: Record<Tone, string> = {
  neutral: "bg-[var(--color-line-2)] text-[var(--color-ink-2)] border-[var(--color-line)]",
  accent: "bg-[var(--color-accent-soft)] text-[var(--color-accent-ink)] border-transparent",
  good: "bg-[var(--color-good-soft)] text-[var(--color-good)] border-transparent",
  warn: "bg-[var(--color-warn-soft)] text-[var(--color-warn)] border-transparent",
  bad: "bg-[var(--color-bad-soft)] text-[var(--color-bad)] border-transparent",
};

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Dot({ color }: { color: string }) {
  return <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />;
}

export function StatTile({
  label,
  value,
  unit,
  sub,
  tone = "neutral",
  delay = 0,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  tone?: Tone;
  delay?: number;
}) {
  const accent =
    tone === "good"
      ? "var(--color-good)"
      : tone === "warn"
        ? "var(--color-warn)"
        : tone === "bad"
          ? "var(--color-bad)"
          : tone === "accent"
            ? "var(--color-accent)"
            : "var(--color-ink)";
  return (
    <div className="rise" style={{ animationDelay: `${delay}ms` }}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--color-ink-3)]">
        {label}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span className="nums text-[28px] leading-none" style={{ color: accent }}>
          {value}
        </span>
        {unit && <span className="text-sm text-[var(--color-ink-3)]">{unit}</span>}
      </div>
      {sub && <div className="mt-1.5 text-xs text-[var(--color-ink-2)]">{sub}</div>}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-md bg-[var(--color-line-2)] ${className}`}
      style={{ animationDuration: "1.4s" }}
    />
  );
}

export function LoadingBlock({ label = "Loading evaluation…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 px-1 py-16 text-sm text-[var(--color-ink-3)]">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[var(--color-line)] border-t-[var(--color-accent)]" />
      {label}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <Card className="border-[var(--color-bad-soft)]">
      <div className="flex flex-col gap-3">
        <Badge tone="bad">Request failed</Badge>
        <p className="text-sm text-[var(--color-ink-2)]">
          Could not reach the evaluation API. Make sure the backend is running on{" "}
          <code className="nums text-xs">http://localhost:8000</code>.
        </p>
        <code className="nums rounded-md bg-[var(--color-bad-soft)] px-3 py-2 text-xs text-[var(--color-bad)]">
          {error}
        </code>
        {onRetry && (
          <button
            onClick={onRetry}
            className="self-start rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3.5 py-1.5 text-sm font-medium text-[var(--color-ink)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
          >
            Retry
          </button>
        )}
      </div>
    </Card>
  );
}

export function EmptyNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-[var(--color-line)] px-4 py-8 text-center text-sm text-[var(--color-ink-3)]">
      {children}
    </div>
  );
}

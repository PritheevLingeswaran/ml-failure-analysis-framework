export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3): string =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const cost = (v: number | null | undefined, digits = 3): string =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const int = (v: number | null | undefined): string =>
  v == null || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString();

export const signed = (v: number, digits = 3): string =>
  `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;

/** Human label for a model id. */
export const modelLabel = (m: string): string =>
  ({
    logreg: "Logistic Regression",
    rf: "Random Forest",
    ensemble_avg: "Ensemble (avg)",
  })[m] ?? m;

/** Stable per-model accent color (indigo-anchored, colorblind-distinct). */
export const MODEL_COLORS: Record<string, string> = {
  rf: "#4f46e5", // indigo — usually the winner
  logreg: "#0f766e", // teal
  ensemble_avg: "#b45309", // amber
};
export const modelColor = (m: string, i = 0): string =>
  MODEL_COLORS[m] ?? ["#4f46e5", "#0f766e", "#b45309", "#9333ea", "#0369a1"][i % 5];

export const titleCase = (s: string): string =>
  s.replace(/[_\[]/g, " ").replace(/\s+/g, " ").trim();

/**
 * Loss-reduction can exceed 100% when the optimized decision has *negative*
 * expected cost (true-positive rewards outweigh error costs). Show it honestly
 * without implying a bug: cap the headline at ">100%" and let the caller add a
 * "net-positive" note.
 */
export const reductionLabel = (reductionPct: number): string =>
  reductionPct > 1.001 ? ">100%" : pct(reductionPct, 1);

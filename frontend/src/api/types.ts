// Types mirror the actual FastAPI responses observed from the running backend
// (src/api/routes.py + src/schemas/api.py). Metrics can be null on degenerate
// slices (single-class), so nullable numbers are used where the backend emits them.

export interface Confusion {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface MetricsValidity {
  precision_valid: boolean;
  recall_valid: boolean;
  f1_valid: boolean;
  reasons: string[];
}

export interface Metrics {
  threshold: number;
  accuracy: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  roc_auc: number | null;
  pr_auc: number | null;
  brier: number | null;
  ece: number | null;
  confusion: Confusion;
  avg_confidence: number;
  metrics_validity: MetricsValidity;
}

export interface RankingRow {
  model: string;
  best_threshold: number;
  expected_cost: number;
  f1_at_best: number;
  roc_auc: number;
  pr_auc: number;
  brier: number;
  ece: number;
  count: number;
}

export interface CostCurve {
  thresholds: number[];
  expected_costs: number[];
}

export interface OverallModel extends Metrics {
  model_name: string;
  split: string;
  count: number;
  best_threshold: number;
  expected_cost_at_best_threshold: number;
  threshold_ci: [number, number];
}

export interface SliceRow {
  slice_name: string;
  description: string;
  model_name: string;
  model?: string;
  split: string;
  count: number;
  metrics: Metrics;
  metrics_validity: MetricsValidity;
  decision: {
    best_threshold: number;
    expected_cost: number;
    costs: Record<string, number>;
  };
  instability: {
    unstable: boolean;
    reason: string;
    sample_count: number;
  };
}

export interface PerModel {
  overall: OverallModel;
  slices: SliceRow[];
  best: {
    threshold: number;
    expected_cost: number;
    metrics: Metrics;
  };
  calibration: { method: string; optimized_on_calibrated_scores: boolean };
  threshold_uncertainty: { mean_threshold: number; ci: [number, number] };
  cost_curve: CostCurve;
}

export interface CompareResponse {
  summary: { ranking: RankingRow[]; winner: string };
  per_model: Record<string, PerModel>;
}

export interface SlicesResponse {
  slices: SliceRow[];
}

export interface ErrorExample {
  id: string;
  y_score: number;
  confidence: number;
  text: string;
}

export interface ClusterSummary {
  cluster: number;
  count: number;
  fp: number;
  fn: number;
  top_terms: string[];
  examples: { text: string; y_score: number; error_type: string }[];
}

export interface ModelClusters {
  available: boolean;
  n_clusters: number;
  summaries: ClusterSummary[];
}

export interface ErrorsResponse {
  top_false_positives: Record<string, ErrorExample[]>;
  top_false_negatives: Record<string, ErrorExample[]>;
  clusters: Record<string, ModelClusters>;
}

export interface SliceRecommendation {
  slice_name: string;
  recommended_model: string;
  recommended_threshold: number;
  expected_cost: number;
  count: number;
  unstable: boolean;
}

export interface RecommendResponse {
  recommended_model: string;
  recommended_threshold: number;
  rationale: {
    objective: string;
    use_case: string;
    ranking: RankingRow[];
    note: string;
  };
  per_slice_recommendations: SliceRecommendation[] | null;
}

export interface DriftFeature {
  feature: string;
  type: string;
  metric: string;
  score: number;
  drifted: boolean;
}

export interface CostScenario {
  scenario: {
    name: string;
    fp_multiplier: number;
    fn_multiplier: number;
    costs: Record<string, number>;
  };
  ranking: { model: string; best_threshold: number; expected_cost: number }[];
  winner: string;
}

export interface DiagnosticsResponse {
  run_id: string;
  split: string;
  drift: {
    num_features_checked: number;
    num_drifted: number;
    top_drifted: DriftFeature[];
    all_features: DriftFeature[];
  };
  cost_sensitivity: {
    scenarios: CostScenario[];
    winner_robustness: { model: string; win_count: number; win_rate: number }[];
  };
}

export interface QualityResponse {
  quality: {
    run_id: string;
    split: string;
    dataset_size: { split_count: number; total_count_if_available: number };
    models_compared: number;
    slice_diagnostics: {
      definition: string;
      thresholds: Record<string, number>;
      diagnostics_found: number;
      top_diagnostics: {
        slice_name: string;
        count: number;
        metric: string | null;
        drop_pct: number;
        cost_increase_pct: number;
        triggered_by: string;
        unstable_flag: boolean;
      }[];
    };
    business_loss_reduction: {
      use_case: string;
      baseline_threshold: number;
      optimized_threshold: number;
      baseline_expected_cost: number;
      optimized_expected_cost: number;
      reduction_pct: number;
      winner_vs_runnerup_cost_delta: number;
    };
    runtime: { total_sec: number; stages: Record<string, number> };
  };
}

export interface VersionResponse {
  name: string;
  version: string;
}

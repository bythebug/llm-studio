const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? res.statusText);
  }
  return res.json();
}

// ── Base models ─────────────────────────────────────────────────────────────

export interface BaseModel {
  id: string;
  hf_id: string;
  max_tokens: number;
  family: string;
}

export const getBaseModels = () => req<{ models: BaseModel[] }>("/base-models");

// ── Jobs ────────────────────────────────────────────────────────────────────

export const createJob = (user_id: number, model_name: string) =>
  req("/jobs", { method: "POST", body: JSON.stringify({ user_id, model_name }) });

export const deleteJob = async (id: number) => {
  const res = await fetch(`${BASE}/jobs/${id}`, { method: "DELETE" });
  if (!res.ok) { const d = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(d.detail); }
};

export const loadSampleJobs = () =>
  req<{ jobs: { job_id: number; model_name: string; description: string; rows: number }[] }>("/sample-jobs", { method: "POST" });

export const startTraining = (id: number) =>
  req(`/jobs/${id}/start_training`, { method: "POST" });

export const getJobStatus = (id: number) => req<JobStatus>(`/jobs/${id}/status`);

export const getJobLogs = (id: number) => req<{ logs: LogEntry[] }>(`/jobs/${id}/logs`);

// ── Data ────────────────────────────────────────────────────────────────────

export const uploadData = async (id: number, file: File) => {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/jobs/${id}/upload_data`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
  return res.json();
};

export const getDataPreview = (id: number, rows = 5) =>
  req<DataPreview>(`/jobs/${id}/data_preview?rows=${rows}`);

export const getDataStats = (id: number) => req<DataStats>(`/jobs/${id}/data_stats`);

// ── Evaluation ──────────────────────────────────────────────────────────────

export const getMetrics = (id: number) => req<MetricsResponse>(`/jobs/${id}/metrics`);

export const getModelComparison = (id: number) =>
  req<ComparisonResponse>(`/jobs/${id}/model_comparison`);

// ── Inference ───────────────────────────────────────────────────────────────

export const predict = (id: number, input: string, version?: number) =>
  req<PredictResponse>(`/jobs/${id}/predict`, {
    method: "POST",
    body: JSON.stringify({ input, version: version ?? null, max_new_tokens: 128 }),
  });

export const predictBatch = (id: number, inputs: string[], version?: number) =>
  req<BatchPredictResponse>(`/jobs/${id}/predict_batch`, {
    method: "POST",
    body: JSON.stringify({ inputs, version: version ?? null, max_new_tokens: 128 }),
  });

export const getModels = (id: number) => req<ModelsResponse>(`/jobs/${id}/models`);

export const getMonitoring = (id: number) =>
  req<MonitoringResponse>(`/jobs/${id}/monitoring`);

// ── Compute ──────────────────────────────────────────────────────────────────

export const listCompute = () => req<{ instances: ComputeInstance[] }>("/compute");

export const addCompute = (body: AddComputeRequest) =>
  req<ComputeInstance>("/compute", { method: "POST", body: JSON.stringify(body) });

export const testCompute = (id: number) =>
  req<{ success: boolean; message: string; status: string }>(`/compute/${id}/test`, { method: "POST" });

export const deleteCompute = async (id: number) => {
  const res = await fetch(`${BASE}/compute/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 204) throw new Error(res.statusText);
};

export const startTrainingRemote = (jobId: number, computeId: number | null) =>
  req(`/jobs/${jobId}/start_training_remote`, {
    method: "POST",
    body: JSON.stringify({ compute_id: computeId }),
  });

// ── Experiments ─────────────────────────────────────────────────────────────

export const getExperiments = () => req<{ experiments: Experiment[] }>("/experiments");

export const getExperimentRuns = (id: string) =>
  req<{ runs: Run[] }>(`/experiments/${id}`);

// ── Types ────────────────────────────────────────────────────────────────────

export interface JobStatus {
  job_id: number;
  model_name: string;
  status: "queued" | "training" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  model_versions: ModelVersionSummary[];
  loss_curves: LossCurves;
}

export interface ModelVersionSummary {
  version: number;
  loss: number | null;
  accuracy: number | null;
  path: string;
}

export interface LossCurves {
  epochs?: number[];
  train_loss?: number[];
  val_loss?: number[];
  train_perplexity?: number[];
  val_perplexity?: number[];
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

export interface DataPreview {
  job_id: number;
  total_rows: number;
  preview: { input: string; output: string }[];
}

export interface DataStats {
  job_id: number;
  num_rows: number;
  avg_input_length: number;
  avg_output_length: number;
  vocab_size: number;
  split: { train: number; val: number; test: number };
}

export interface MetricsResponse {
  job_id: number;
  versions: VersionMetrics[];
}

export interface VersionMetrics {
  version_num: number;
  training_loss: number | null;
  evaluation: EvalResult | string;
}

export interface EvalResult {
  task_type: string;
  metrics: Record<string, number>;
  interpretation: Record<string, string>;
}

export interface ComparisonResponse {
  best_version: number;
  primary_metric: string;
  recommendation: string;
  ranking: RankedVersion[];
}

export interface RankedVersion {
  rank: number;
  version_num: number;
  metrics: Record<string, number>;
}

export interface PredictResponse {
  output: string;
  confidence: number;
  input_token_count: number;
  output_token_count: number;
  latency_ms: number;
  version_num: number;
}

export interface ModelsResponse {
  versions: ModelVersion[];
}

export interface ModelVersion {
  version_num: number;
  loss: number | null;
  accuracy: number | null;
  cached: boolean;
  created_at: string;
}

export interface BatchPredictResponse {
  job_id: number;
  version_num: number;
  total_latency_ms: number;
  avg_latency_ms: number;
  predictions: {
    input: string;
    output: string;
    confidence: number;
    input_token_count: number;
    output_token_count: number;
  }[];
}

export interface ComparisonResponse {
  job_id: number;
  best_version: number;
  primary_metric: string;
  recommendation: string;
  ranking: {
    rank: number;
    version_num: number;
    metrics: Record<string, number>;
  }[];
  significance: {
    versions: string;
    metric: string;
    significant: boolean;
    better_version: number | null;
    p_value: number | null;
  }[];
}

export interface MonitoringResponse {
  job_id: number;
  latency_stats: {
    count: number;
    mean_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    max_ms: number;
  };
  drift_alerts: { timestamp: string; drift_score: number; message: string }[];
  baseline_ready: boolean;
}

export interface ComputeInstance {
  id: number;
  name: string;
  host: string;
  port: number;
  username: string;
  key_path: string | null;
  last_status: "unknown" | "connected" | "error";
  last_checked: string | null;
  created_at: string;
}

export interface AddComputeRequest {
  name: string;
  host: string;
  port: number;
  username: string;
  key_path?: string;
}

export interface Experiment {
  experiment_id: string;
  name: string;
  lifecycle_stage: string;
}

export interface Run {
  run_id: string;
  run_name: string;
  job_id: string;
  status: string;
  params: Record<string, string>;
  metrics: Record<string, number>;
}

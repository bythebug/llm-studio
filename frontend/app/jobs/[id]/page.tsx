"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, RefreshCw, Play, AlertCircle, Trophy, Activity, Server, Download } from "lucide-react";
import {
  getJobStatus, startTrainingRemote, getJobLogs, uploadData, getDataPreview, getDataStats,
  getMetrics, predict, predictBatch, getModels, getModelComparison, getMonitoring,
  listCompute,
  type JobStatus, type LogEntry, type DataPreview, type DataStats,
  type MetricsResponse, type PredictResponse, type BatchPredictResponse,
  type ModelsResponse, type ComparisonResponse, type MonitoringResponse,
  type ComputeInstance,
} from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";
import { LossCurveChart } from "@/components/LossCurveChart";

type Tab = "overview" | "data" | "training" | "evaluation" | "inference" | "monitoring";

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-white border border-gray-200 shadow-sm ${className}`}>{children}</div>;
}

function StatCard({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <Card className="p-5">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">{label}</p>
      <p className="text-gray-900 font-semibold mt-1 text-sm">{value}</p>
    </Card>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 bg-red-50 border border-red-200 p-4">
      <AlertCircle size={15} className="text-red-500 mt-0.5 shrink-0" />
      <p className="text-red-700 text-sm">{message}</p>
    </div>
  );
}

export default function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const jobId = parseInt(id);

  const [job, setJob] = useState<JobStatus | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [error, setError] = useState("");
  const [tabError, setTabError] = useState("");

  // Tab data
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [stats, setStats] = useState<DataStats | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringResponse | null>(null);

  // Inference state
  const [inferMode, setInferMode] = useState<"single" | "batch">("single");
  const [input, setInput] = useState("");
  const [batchInput, setBatchInput] = useState("");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictResponse | null>(null);
  const [predicting, setPredicting] = useState(false);

  // Compute state
  const [computeInstances, setComputeInstances] = useState<ComputeInstance[]>([]);
  const [selectedComputeId, setSelectedComputeId] = useState<number | null>(null);

  // Action state
  const [uploading, setUploading] = useState(false);
  const [starting, setStarting] = useState(false);

  const refreshJob = useCallback(async () => {
    try { setJob(await getJobStatus(jobId)); setError(""); }
    catch (e: unknown) { setError((e as Error).message); }
  }, [jobId]);

  useEffect(() => { refreshJob(); }, [refreshJob]);

  useEffect(() => {
    listCompute().then(r => setComputeInstances(r.instances)).catch(() => {});
  }, []);

  useEffect(() => {
    if (job?.status !== "training") return;
    const t = setInterval(refreshJob, 4000);
    return () => clearInterval(t);
  }, [job?.status, refreshJob]);

  const loadTab = async (t: Tab) => {
    setTab(t); setTabError("");
    try {
      if (t === "training") setLogs((await getJobLogs(jobId)).logs);
      if (t === "data") {
        const [p, s] = await Promise.all([getDataPreview(jobId), getDataStats(jobId).catch(() => null)]);
        setPreview(p); setStats(s);
      }
      if (t === "evaluation") {
        const [m, c] = await Promise.allSettled([getMetrics(jobId), getModelComparison(jobId)]);
        if (m.status === "fulfilled") setMetrics(m.value);
        if (c.status === "fulfilled") setComparison(c.value);
      }
      if (t === "inference") setModels(await getModels(jobId).catch(() => null));
      if (t === "monitoring") setMonitoring(await getMonitoring(jobId));
    } catch (e: unknown) { setTabError((e as Error).message); }
  };

  const handleStartTraining = async () => {
    setStarting(true);
    try { await startTrainingRemote(jobId, selectedComputeId); await refreshJob(); loadTab("training"); }
    catch (e: unknown) { setTabError((e as Error).message); }
    finally { setStarting(false); }
  };

  const handleUpload = async (file: File) => {
    setUploading(true); setTabError("");
    try {
      await uploadData(jobId, file);
      const [p, s] = await Promise.all([getDataPreview(jobId), getDataStats(jobId).catch(() => null)]);
      setPreview(p); setStats(s);
    } catch (e: unknown) { setTabError((e as Error).message); }
    finally { setUploading(false); }
  };

  const handlePredict = async () => {
    setPredicting(true); setTabError(""); setPrediction(null); setBatchResult(null);
    try {
      if (inferMode === "single") {
        if (!input.trim()) return;
        setPrediction(await predict(jobId, input));
      } else {
        const lines = batchInput.split("\n").map(l => l.trim()).filter(Boolean);
        if (!lines.length) return;
        setBatchResult(await predictBatch(jobId, lines));
      }
    } catch (e: unknown) { setTabError((e as Error).message); }
    finally { setPredicting(false); }
  };

  if (error) return <ErrorBanner message={error} />;
  if (!job) return <div className="text-gray-400 text-sm p-4">Loading…</div>;

  const tabs: { key: Tab; label: string }[] = [
    { key: "overview",   label: "Overview"   },
    { key: "data",       label: "Data"       },
    { key: "training",   label: "Training"   },
    { key: "evaluation", label: "Evaluation" },
    { key: "inference",  label: "Inference"  },
    { key: "monitoring", label: "Monitoring" },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <Link href="/" className="text-gray-400 hover:text-gray-600 text-xs flex items-center gap-1 transition-colors">
            <ChevronLeft size={12} /> Jobs
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">
            Job #{job.job_id}
            <span className="text-gray-400 font-normal ml-2 text-lg">— {job.model_name}</span>
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={job.status} />
          <button onClick={refreshJob} className="text-gray-400 hover:text-gray-600 transition-colors p-1.5 hover:bg-gray-100">
            <RefreshCw size={14} />
          </button>
          {(job.status === "queued" || job.status === "failed") && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 border border-gray-200 text-xs text-gray-500 pl-2.5 pr-1 py-1.5">
                <Server size={11} className="shrink-0" />
                <select
                  value={selectedComputeId ?? ""}
                  onChange={e => setSelectedComputeId(e.target.value === "" ? null : parseInt(e.target.value))}
                  className="bg-transparent text-xs text-gray-700 focus:outline-none cursor-pointer pr-1"
                >
                  <option value="">Local</option>
                  {computeInstances.map(inst => (
                    <option key={inst.id} value={inst.id}>{inst.name}</option>
                  ))}
                </select>
              </div>
              <button onClick={handleStartTraining} disabled={starting}
                className="flex items-center gap-1.5 bg-blue-600 text-white px-4 py-2 text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50">
                <Play size={12} />
                {starting ? "Starting…" : "Start Training"}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {tabs.map(t => (
          <button key={t.key} onClick={() => loadTab(t.key)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-px ${
              tab === t.key ? "border-blue-600 text-blue-600" : "border-transparent text-gray-500 hover:text-gray-700"
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      {tabError && <ErrorBanner message={tabError} />}

      {/* ── Overview ── */}
      {tab === "overview" && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <StatCard label="Status"  value={<StatusBadge status={job.status} />} />
            <StatCard label="Model"   value={job.model_name} />
            <StatCard label="Versions" value={job.model_versions?.length ?? 0} />
            <StatCard label="Created" value={new Date(job.created_at).toLocaleString()} />
            <StatCard label="Updated" value={new Date(job.updated_at).toLocaleString()} />
            <StatCard label="Job ID"  value={<span className="font-mono">#{job.job_id}</span>} />
          </div>
          {job.model_versions?.length > 0 && (
            <Card className="p-6">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-4">Model Versions</p>
              <div className="divide-y divide-gray-100">
                {job.model_versions.map(v => (
                  <div key={v.version} className="flex items-center justify-between py-3 text-sm">
                    <span className="font-medium text-gray-900">v{v.version}</span>
                    <div className="flex items-center gap-4">
                      <span className="text-gray-500">{v.loss != null ? `val loss: ${v.loss.toFixed(4)}` : "—"}</span>
                      <a
                        href={`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/jobs/${job.job_id}/models/${v.version}/download`}
                        download
                        className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium transition-colors"
                      >
                        <Download size={11} />
                        Download
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── Data ── */}
      {tab === "data" && (
        <div className="space-y-5">
          <Card className="p-6">
            <p className="text-sm font-semibold text-gray-700 mb-4">Upload Training Data</p>
            <label className={`flex flex-col items-center justify-center border-2 border-dashed p-10 cursor-pointer transition-colors ${uploading ? "border-gray-200 opacity-50 cursor-not-allowed" : "border-gray-300 hover:border-blue-400 hover:bg-blue-50"}`}>
              <span className="text-2xl mb-2">📂</span>
              <span className="text-gray-600 text-sm font-medium">{uploading ? "Uploading…" : "Drop CSV or JSON here, or click to browse"}</span>
              <span className="text-gray-400 text-xs mt-1">CSV: input_text, expected_output columns</span>
              <input type="file" accept=".csv,.json" className="hidden" disabled={uploading}
                onChange={e => { if (e.target.files?.[0]) handleUpload(e.target.files[0]); }} />
            </label>
          </Card>

          {stats && (
            <div className="grid grid-cols-3 gap-4">
              {[
                { label: "Total Rows",        value: stats.num_rows },
                { label: "Vocab Size",        value: stats.vocab_size?.toLocaleString() ?? "—" },
                { label: "Avg Input Length",  value: `${stats.avg_input_length} chars` },
                { label: "Train",             value: stats.split?.train },
                { label: "Validation",        value: stats.split?.val },
                { label: "Test",              value: stats.split?.test },
              ].map(s => <StatCard key={s.label} label={s.label} value={s.value} />)}
            </div>
          )}

          {preview?.preview?.length ? (
            <Card className="overflow-hidden">
              <div className="px-6 py-3 border-b border-gray-100 bg-gray-50">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Preview — {preview.total_rows} rows total</span>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50 text-xs text-gray-500 border-b border-gray-100">
                    <th className="text-left px-6 py-3 font-medium w-1/2">Input</th>
                    <th className="text-left px-6 py-3 font-medium w-1/2">Expected Output</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {preview.preview.map((row, i) => (
                    <tr key={i}>
                      <td className="px-6 py-3 text-gray-700 truncate max-w-xs">{row.input}</td>
                      <td className="px-6 py-3 text-gray-500 truncate max-w-xs">{row.output}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ) : !tabError ? <p className="text-gray-400 text-sm text-center py-6">No data uploaded yet.</p> : null}
        </div>
      )}

      {/* ── Training ── */}
      {tab === "training" && (
        <div className="space-y-5">
          <Card className="p-6">
            <p className="text-sm font-semibold text-gray-700 mb-4">Loss Curves</p>
            <LossCurveChart curves={job.loss_curves ?? {}} />
          </Card>
          <Card className="overflow-hidden">
            <div className="px-6 py-3 border-b border-gray-100 bg-gray-50">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Training Logs</span>
            </div>
            {logs.length === 0 ? <p className="text-gray-400 text-sm p-6">No logs yet.</p> : (
              <div className="divide-y divide-gray-100 max-h-72 overflow-y-auto">
                {logs.map((log, i) => (
                  <div key={i} className="px-6 py-2.5 flex gap-4 text-xs">
                    <span className="text-gray-400 shrink-0 font-mono">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    <span className={log.level === "error" ? "text-red-600" : "text-gray-700"}>{log.message}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ── Evaluation ── */}
      {tab === "evaluation" && (
        <div className="space-y-5">

          {/* Per-version metrics */}
          {!metrics || metrics.versions.length === 0 ? (
            <p className="text-gray-400 text-sm text-center py-8">No evaluation results yet.</p>
          ) : metrics.versions.map(v => (
            <Card key={v.version_num} className="p-6">
              <p className="text-sm font-semibold text-gray-700 mb-4">Version {v.version_num}</p>
              {typeof v.evaluation === "string" ? (
                <p className="text-gray-500 text-sm">{v.evaluation}</p>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    {Object.entries(v.evaluation.metrics ?? {}).map(([k, val]) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 p-4">
                        <p className="text-gray-500 text-xs uppercase font-medium">{k}</p>
                        <p className="text-gray-900 font-bold text-lg mt-1">{typeof val === "number" ? val.toFixed(3) : val}</p>
                      </div>
                    ))}
                  </div>
                  {v.evaluation.interpretation && (
                    <div className="bg-blue-50 p-4 space-y-1">
                      {Object.entries(v.evaluation.interpretation).map(([k, text]) => (
                        <p key={k} className="text-blue-700 text-xs">{String(text)}</p>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </Card>
          ))}

          {/* Model comparison */}
          {comparison && (
            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Trophy size={15} className="text-amber-500" />
                <p className="text-sm font-semibold text-gray-700">Model Comparison</p>
              </div>

              {/* Recommendation */}
              <div className="bg-amber-50 border border-amber-200 p-4 mb-5">
                <p className="text-amber-800 text-sm">{comparison.recommendation}</p>
              </div>

              {/* Ranking table */}
              <table className="w-full text-sm mb-5">
                <thead>
                  <tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wider">
                    <th className="text-left px-4 py-2.5 font-medium">Rank</th>
                    <th className="text-left px-4 py-2.5 font-medium">Version</th>
                    {comparison.ranking[0] && Object.keys(comparison.ranking[0].metrics).map(k => (
                      <th key={k} className="text-left px-4 py-2.5 font-medium">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {comparison.ranking.map(r => (
                    <tr key={r.version_num} className={r.version_num === comparison.best_version ? "bg-green-50" : ""}>
                      <td className="px-4 py-3 text-gray-500">#{r.rank}</td>
                      <td className="px-4 py-3 font-medium text-gray-900">
                        v{r.version_num}
                        {r.version_num === comparison.best_version && (
                          <span className="ml-2 text-xs text-green-600 font-medium">best</span>
                        )}
                      </td>
                      {Object.values(r.metrics).map((val, i) => (
                        <td key={i} className="px-4 py-3 text-gray-700">{typeof val === "number" ? val.toFixed(3) : val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Significance */}
              {comparison.significance.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Statistical Significance</p>
                  <div className="space-y-2">
                    {comparison.significance.map((s, i) => (
                      <div key={i} className="flex items-center justify-between text-xs text-gray-600 bg-gray-50 px-4 py-2.5">
                        <span>{s.versions} — {s.metric}</span>
                        <span className={s.significant ? "text-green-600 font-medium" : "text-gray-400"}>
                          {s.significant ? `Significant — v${s.better_version} wins` : "Not significant"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          )}
        </div>
      )}

      {/* ── Inference ── */}
      {tab === "inference" && (
        <div className="space-y-5">
          {models && models.versions.length > 0 && (
            <Card className="p-6">
              <p className="text-sm font-semibold text-gray-700 mb-3">Available Versions</p>
              <div className="flex gap-3 flex-wrap">
                {models.versions.map(v => (
                  <div key={v.version_num} className="border border-gray-200 px-4 py-3 text-xs">
                    <p className="font-semibold text-gray-900">v{v.version_num}</p>
                    <p className="text-gray-500 mt-0.5">{v.loss != null ? `loss ${v.loss.toFixed(3)}` : "—"}</p>
                    {v.cached && <p className="text-green-600 font-medium mt-0.5">● cached</p>}
                  </div>
                ))}
              </div>
            </Card>
          )}

          <Card className="p-6 space-y-4">
            {/* Mode toggle */}
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold text-gray-700">Inference Playground</p>
              <div className="flex border border-gray-200 text-xs">
                {(["single", "batch"] as const).map(m => (
                  <button key={m} onClick={() => { setInferMode(m); setPrediction(null); setBatchResult(null); }}
                    className={`px-3 py-1.5 font-medium transition-colors ${inferMode === m ? "bg-blue-600 text-white" : "text-gray-500 hover:text-gray-700"}`}>
                    {m === "single" ? "Single" : "Batch"}
                  </button>
                ))}
              </div>
            </div>

            {inferMode === "single" ? (
              <textarea value={input} onChange={e => setInput(e.target.value)}
                placeholder="Enter your input text…" rows={4}
                className="w-full border border-gray-300 px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none" />
            ) : (
              <div>
                <textarea value={batchInput} onChange={e => setBatchInput(e.target.value)}
                  placeholder={"One input per line:\nTranslate: Hello\nTranslate: Goodbye\nTranslate: Thank you"} rows={6}
                  className="w-full border border-gray-300 px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none font-mono" />
                <p className="text-xs text-gray-400 mt-1">{batchInput.split("\n").filter(l => l.trim()).length} inputs</p>
              </div>
            )}

            <button onClick={handlePredict} disabled={predicting || (inferMode === "single" ? !input.trim() : !batchInput.trim())}
              className="bg-blue-600 text-white px-5 py-2.5 text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-40">
              {predicting ? "Running…" : inferMode === "single" ? "Run Prediction" : "Run Batch"}
            </button>
          </Card>

          {/* Single result */}
          {prediction && (
            <Card className="p-6 space-y-4">
              <p className="text-sm font-semibold text-gray-700">Output</p>
              <div className="bg-gray-50 border border-gray-200 p-4 text-sm text-gray-900 whitespace-pre-wrap min-h-12">
                {prediction.output || <span className="text-gray-400">(empty output)</span>}
              </div>
              <div className="grid grid-cols-4 gap-3">
                {[
                  { label: "Confidence",    value: `${(prediction.confidence * 100).toFixed(1)}%` },
                  { label: "Latency",       value: `${prediction.latency_ms}ms` },
                  { label: "Input tokens",  value: prediction.input_token_count },
                  { label: "Output tokens", value: prediction.output_token_count },
                ].map(m => (
                  <div key={m.label} className="bg-gray-50 border border-gray-200 p-3">
                    <p className="text-gray-500 text-xs">{m.label}</p>
                    <p className="text-gray-900 font-semibold mt-0.5 text-sm">{m.value}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Batch results */}
          {batchResult && (
            <Card className="overflow-hidden">
              <div className="px-6 py-3 border-b border-gray-100 bg-gray-50 flex items-center justify-between">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Batch Results — {batchResult.predictions.length} predictions
                </span>
                <span className="text-xs text-gray-400">
                  Total {batchResult.total_latency_ms}ms · avg {batchResult.avg_latency_ms}ms
                </span>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wider border-b border-gray-100">
                    <th className="text-left px-5 py-2.5 font-medium w-1/3">Input</th>
                    <th className="text-left px-5 py-2.5 font-medium w-1/3">Output</th>
                    <th className="text-left px-5 py-2.5 font-medium">Confidence</th>
                    <th className="text-left px-5 py-2.5 font-medium">Tokens in/out</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {batchResult.predictions.map((p, i) => (
                    <tr key={i} className="hover:bg-gray-50 transition-colors">
                      <td className="px-5 py-3 text-gray-600 truncate max-w-xs">{p.input}</td>
                      <td className="px-5 py-3 text-gray-900 truncate max-w-xs">{p.output || <span className="text-gray-400">—</span>}</td>
                      <td className="px-5 py-3 text-gray-600">{(p.confidence * 100).toFixed(1)}%</td>
                      <td className="px-5 py-3 text-gray-500 font-mono text-xs">{p.input_token_count} / {p.output_token_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}

      {/* ── Monitoring ── */}
      {tab === "monitoring" && (
        <div className="space-y-5">
          {!monitoring ? (
            <p className="text-gray-400 text-sm text-center py-8">No monitoring data yet — run some predictions first.</p>
          ) : (
            <>
              {/* Latency stats */}
              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Activity size={15} className="text-blue-500" />
                  <p className="text-sm font-semibold text-gray-700">Inference Latency</p>
                  <span className="text-xs text-gray-400 ml-auto">{monitoring.latency_stats.count ?? 0} requests</span>
                </div>
                {!monitoring.latency_stats.count ? (
                  <p className="text-gray-400 text-sm">No requests recorded yet.</p>
                ) : (
                  <div className="grid grid-cols-5 gap-3">
                    {[
                      { label: "Mean",  value: `${monitoring.latency_stats.mean_ms}ms` },
                      { label: "p50",   value: `${monitoring.latency_stats.p50_ms}ms` },
                      { label: "p95",   value: `${monitoring.latency_stats.p95_ms}ms`, highlight: true },
                      { label: "p99",   value: `${monitoring.latency_stats.p99_ms}ms`, highlight: true },
                      { label: "Max",   value: `${monitoring.latency_stats.max_ms}ms` },
                    ].map(s => (
                      <div key={s.label} className={`border p-4 ${s.highlight ? "border-orange-200 bg-orange-50" : "border-gray-200 bg-gray-50"}`}>
                        <p className="text-xs text-gray-500 font-medium">{s.label}</p>
                        <p className={`font-bold text-lg mt-1 ${s.highlight ? "text-orange-600" : "text-gray-900"}`}>{s.value}</p>
                      </div>
                    ))}
                  </div>
                )}
              </Card>

              {/* Drift detection */}
              <Card className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <p className="text-sm font-semibold text-gray-700">Confidence Drift</p>
                  <span className={`text-xs font-medium px-2 py-0.5 ${monitoring.baseline_ready ? "bg-green-50 text-green-700" : "bg-gray-100 text-gray-500"}`}>
                    {monitoring.baseline_ready ? "Baseline ready" : "Building baseline…"}
                  </span>
                </div>
                {!monitoring.baseline_ready && (
                  <p className="text-gray-400 text-sm">Need 50 predictions to establish a baseline. Run more inferences.</p>
                )}
                {monitoring.baseline_ready && monitoring.drift_alerts.length === 0 && (
                  <div className="flex items-center gap-2 text-green-600 text-sm">
                    <span className="text-lg">✓</span> No drift detected — model confidence is stable.
                  </div>
                )}
                {monitoring.drift_alerts.length > 0 && (
                  <div className="space-y-2">
                    {monitoring.drift_alerts.map((a, i) => (
                      <div key={i} className="bg-red-50 border border-red-200 p-3 text-xs">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-red-700 font-medium">Drift score: {a.drift_score.toFixed(2)}</span>
                          <span className="text-red-400">{new Date(a.timestamp).toLocaleString()}</span>
                        </div>
                        <p className="text-red-600">{a.message}</p>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </>
          )}
        </div>
      )}
    </div>
  );
}

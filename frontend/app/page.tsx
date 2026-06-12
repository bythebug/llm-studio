"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Plus, RefreshCw, AlertCircle, Trash2, FlaskConical } from "lucide-react";
import { createJob, deleteJob, loadSampleJobs, getBaseModels, getJobStatus, type BaseModel, type JobStatus } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

const FALLBACK_MODELS: BaseModel[] = [
  { id: "gpt2",        hf_id: "gpt2",                            max_tokens: 1024, family: "causal-lm" },
  { id: "gpt2-medium", hf_id: "gpt2-medium",                     max_tokens: 1024, family: "causal-lm" },
  { id: "llama-3-8b",  hf_id: "meta-llama/Meta-Llama-3-8B",      max_tokens: 8192, family: "causal-lm" },
  { id: "mistral-7b",  hf_id: "mistralai/Mistral-7B-v0.1",       max_tokens: 4096, family: "causal-lm" },
  { id: "t5-small",    hf_id: "t5-small",                         max_tokens: 512,  family: "seq2seq"   },
];
const KNOWN_JOB_IDS_KEY = "llm_studio_job_ids";

function loadJobIds(): number[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(KNOWN_JOB_IDS_KEY) ?? "[]"); }
  catch { return []; }
}

function saveJobId(id: number) {
  const ids = loadJobIds();
  if (!ids.includes(id)) localStorage.setItem(KNOWN_JOB_IDS_KEY, JSON.stringify([...ids, id]));
}

export default function Dashboard() {
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [baseModels, setBaseModels] = useState<BaseModel[]>(FALLBACK_MODELS);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState("");
  const [creating, setCreating] = useState(false);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [model, setModel] = useState("gpt2");
  const [userId, setUserId] = useState("1");

  const fetchJobs = async () => {
    setApiError("");
    const ids = loadJobIds();
    if (!ids.length) { setLoading(false); return; }
    try {
      const results = await Promise.allSettled(ids.map(getJobStatus));
      setJobs(results.flatMap(r => r.status === "fulfilled" ? [r.value] : []));
    } catch {
      setApiError("Cannot reach the API. Make sure the backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    getBaseModels()
      .then(r => { if (r.models.length) setBaseModels(r.models); })
      .catch(() => { /* fallback list stays */ });
  }, []);

  const handleCreate = async () => {
    setCreating(true);
    setApiError("");
    try {
      const res = await createJob(parseInt(userId) || 1, model) as { job_id: number };
      saveJobId(res.job_id);
      await fetchJobs();
    } catch (e: unknown) {
      setApiError((e as Error).message);
    } finally {
      setCreating(false);
    }
  };

  const handleLoadSamples = async () => {
    setLoadingSamples(true);
    setApiError("");
    try {
      const res = await loadSampleJobs() as { jobs: { job_id: number }[] };
      res.jobs.forEach(j => saveJobId(j.job_id));
      await fetchJobs();
    } catch (e: unknown) { setApiError((e as Error).message); }
    finally { setLoadingSamples(false); }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteJob(id);
      const ids = loadJobIds().filter(i => i !== id);
      localStorage.setItem(KNOWN_JOB_IDS_KEY, JSON.stringify(ids));
      setJobs(j => j.filter(j => j.job_id !== id));
    } catch (e: unknown) { setApiError((e as Error).message); }
  };

  const stats = {
    total: jobs.length,
    training: jobs.filter(j => j.status === "training").length,
    completed: jobs.filter(j => j.status === "completed").length,
    failed: jobs.filter(j => j.status === "failed").length,
  };

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Jobs</h1>
          <p className="text-gray-500 text-sm mt-0.5">Manage LLM fine-tuning jobs</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleLoadSamples} disabled={loadingSamples}
            className="flex items-center gap-1.5 border border-gray-200 text-gray-500 hover:text-gray-800 hover:bg-gray-50 transition-colors px-3 py-1.5 text-xs font-medium disabled:opacity-40">
            <FlaskConical size={13} />
            {loadingSamples ? "Loading…" : "Load Samples"}
          </button>
          <button onClick={fetchJobs} className="text-gray-400 hover:text-gray-600 transition-colors p-2 hover:bg-gray-100">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {/* API error banner */}
      {apiError && (
        <div className="flex items-start gap-3 bg-red-50 border border-red-200  p-4">
          <AlertCircle size={16} className="text-red-500 mt-0.5 shrink-0" />
          <p className="text-red-700 text-sm">{apiError}</p>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "Total", value: stats.total, color: "text-gray-900" },
          { label: "Training", value: stats.training, color: "text-blue-600" },
          { label: "Completed", value: stats.completed, color: "text-green-600" },
          { label: "Failed", value: stats.failed, color: "text-red-600" },
        ].map(s => (
          <div key={s.label} className="bg-white border border-gray-200  p-5 shadow-sm">
            <p className="text-gray-500 text-xs font-medium uppercase tracking-wider">{s.label}</p>
            <p className={`text-3xl font-bold mt-1 ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* New job */}
      <div className="bg-white border border-gray-200  p-6 shadow-sm">
        <h2 className="text-sm font-semibold text-gray-700 mb-4">Create New Job</h2>
        <div className="flex gap-3 items-end flex-wrap">
          <div>
            <label className="text-xs text-gray-500 font-medium block mb-1.5">User ID</label>
            <input
              type="number"
              value={userId}
              onChange={e => setUserId(e.target.value)}
              className="border border-gray-300  px-3 py-2 text-sm text-gray-900 w-24 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 font-medium block mb-1.5">Base Model</label>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              className="border border-gray-300  px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {baseModels.map(m => (
                <option key={m.id} value={m.id} title={`${m.hf_id} · ${m.max_tokens} tokens · ${m.family}`}>
                  {m.id}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={handleCreate}
            disabled={creating}
            className="flex items-center gap-2 bg-blue-600 text-white  px-4 py-2 text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            <Plus size={14} />
            {creating ? "Creating…" : "Create Job"}
          </button>
        </div>
      </div>

      {/* Jobs table */}
      <div className="bg-white border border-gray-200  shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100">
          <h2 className="text-sm font-semibold text-gray-700">All Jobs</h2>
        </div>
        {loading ? (
          <div className="p-8 text-center text-gray-400 text-sm">Loading…</div>
        ) : jobs.length === 0 ? (
          <div className="p-12 text-center">
            <p className="text-gray-400 text-sm">No jobs yet. Create one above.</p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 text-gray-500 text-xs font-medium uppercase tracking-wider">
                <th className="text-left px-6 py-3">Job</th>
                <th className="text-left px-6 py-3">Model</th>
                <th className="text-left px-6 py-3">Status</th>
                <th className="text-left px-6 py-3">Versions</th>
                <th className="text-left px-6 py-3">Last Updated</th>
                <th className="px-6 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {jobs.map(job => (
                <tr key={job.job_id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-4 text-gray-500 font-mono text-xs">#{job.job_id}</td>
                  <td className="px-6 py-4 font-medium text-gray-900">{job.model_name}</td>
                  <td className="px-6 py-4"><StatusBadge status={job.status} /></td>
                  <td className="px-6 py-4 text-gray-500">{job.model_versions?.length ?? 0}</td>
                  <td className="px-6 py-4 text-gray-400 text-xs">{new Date(job.updated_at).toLocaleString()}</td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-3">
                      <Link href={`/jobs/${job.job_id}`} className="text-blue-600 hover:text-blue-800 text-xs font-medium">
                        Open →
                      </Link>
                      <button onClick={() => handleDelete(job.job_id)}
                        className="text-gray-300 hover:text-red-500 transition-colors">
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { getExperiments, getExperimentRuns, type Experiment, type Run } from "@/lib/api";
import { ChevronDown, ChevronRight, AlertCircle } from "lucide-react";

export default function Experiments() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runs, setRuns] = useState<Record<string, Run[]>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getExperiments()
      .then(r => setExperiments(r.experiments))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const toggle = async (id: string) => {
    if (expanded === id) { setExpanded(null); return; }
    setExpanded(id);
    if (!runs[id]) {
      try {
        const r = await getExperimentRuns(id);
        setRuns(prev => ({ ...prev, [id]: r.runs }));
      } catch {
        setRuns(prev => ({ ...prev, [id]: [] }));
      }
    }
  };

  if (loading) return <div className="text-gray-400 text-sm p-4">Loading…</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
        <p className="text-gray-500 text-sm mt-0.5">MLflow training runs</p>
      </div>

      {error && (
        <div className="flex items-start gap-3 bg-red-50 border border-red-200  p-4">
          <AlertCircle size={15} className="text-red-500 mt-0.5 shrink-0" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {experiments.length === 0 && !error ? (
        <div className="bg-white border border-gray-200  p-12 text-center shadow-sm">
          <p className="text-gray-400 text-sm">No experiments yet. Start a training job to create one.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {experiments.map(exp => (
            <div key={exp.experiment_id} className="bg-white border border-gray-200  shadow-sm overflow-hidden">
              <button
                onClick={() => toggle(exp.experiment_id)}
                className="w-full flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {expanded === exp.experiment_id
                    ? <ChevronDown size={14} className="text-gray-400" />
                    : <ChevronRight size={14} className="text-gray-400" />}
                  <span className="font-semibold text-gray-900 text-sm">{exp.name}</span>
                  <span className="text-gray-400 text-xs font-mono">#{exp.experiment_id}</span>
                </div>
                <span className={`text-xs font-medium px-2 py-0.5 ${exp.lifecycle_stage === "active" ? "bg-green-50 text-green-700" : "bg-gray-100 text-gray-500"}`}>
                  {exp.lifecycle_stage}
                </span>
              </button>

              {expanded === exp.experiment_id && (
                <div className="border-t border-gray-100">
                  {!runs[exp.experiment_id] ? (
                    <p className="text-gray-400 text-sm p-6">Loading runs…</p>
                  ) : runs[exp.experiment_id].length === 0 ? (
                    <p className="text-gray-400 text-sm p-6">No runs in this experiment.</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-gray-50 text-gray-500 uppercase tracking-wider font-medium">
                            <th className="text-left px-6 py-3">Run ID</th>
                            <th className="text-left px-6 py-3">Job</th>
                            <th className="text-left px-6 py-3">Status</th>
                            <th className="text-left px-6 py-3">LR</th>
                            <th className="text-left px-6 py-3">Epochs</th>
                            <th className="text-left px-6 py-3">Val Loss</th>
                            <th className="text-left px-6 py-3">Train Loss</th>
                            <th className="text-left px-6 py-3">BLEU</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {runs[exp.experiment_id].map(run => (
                            <tr key={run.run_id} className="hover:bg-gray-50 transition-colors">
                              <td className="px-6 py-3 text-gray-500 font-mono">{run.run_id.slice(0, 8)}</td>
                              <td className="px-6 py-3 text-gray-700">{run.job_id ?? "—"}</td>
                              <td className="px-6 py-3">
                                <span className={`px-2 py-0.5 font-medium ${
                                  run.status === "FINISHED" ? "bg-green-50 text-green-700"
                                  : run.status === "FAILED" ? "bg-red-50 text-red-700"
                                  : "bg-blue-50 text-blue-700"}`}>
                                  {run.status}
                                </span>
                              </td>
                              <td className="px-6 py-3 text-gray-600">{run.params?.learning_rate ?? "—"}</td>
                              <td className="px-6 py-3 text-gray-600">{run.params?.epochs ?? "—"}</td>
                              <td className="px-6 py-3 text-gray-900 font-medium">{run.metrics?.val_loss?.toFixed(4) ?? "—"}</td>
                              <td className="px-6 py-3 text-gray-900 font-medium">{run.metrics?.train_loss?.toFixed(4) ?? "—"}</td>
                              <td className="px-6 py-3 text-gray-900 font-medium">{run.metrics?.["final.bleu"]?.toFixed(1) ?? "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

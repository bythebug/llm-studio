"use client";

import { useEffect, useState } from "react";
import { Server, Plus, Trash2, Zap, AlertCircle, CheckCircle, HelpCircle } from "lucide-react";
import {
  listCompute, addCompute, testCompute, deleteCompute,
  type ComputeInstance, type AddComputeRequest,
} from "@/lib/api";

function StatusIcon({ status }: { status: string }) {
  if (status === "connected") return <CheckCircle size={14} className="text-green-500" />;
  if (status === "error")     return <AlertCircle  size={14} className="text-red-500" />;
  return                             <HelpCircle   size={14} className="text-gray-400" />;
}

const EMPTY_FORM: AddComputeRequest = { name: "", host: "", port: 22, username: "", key_path: "" };

export default function ComputePage() {
  const [instances, setInstances] = useState<ComputeInstance[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<AddComputeRequest>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState<number | null>(null);
  const [testResults, setTestResults] = useState<Record<number, { success: boolean; message: string }>>({});
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const r = await listCompute();
      setInstances(r.instances);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleAdd = async () => {
    if (!form.name || !form.host || !form.username) return;
    setSaving(true);
    try {
      await addCompute({ ...form, key_path: form.key_path || undefined });
      setForm(EMPTY_FORM);
      setShowForm(false);
      await load();
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setSaving(false); }
  };

  const handleTest = async (id: number) => {
    setTesting(id);
    try {
      const r = await testCompute(id);
      setTestResults(prev => ({ ...prev, [id]: { success: r.success, message: r.message } }));
      await load();
    } catch (e: unknown) {
      setTestResults(prev => ({ ...prev, [id]: { success: false, message: (e as Error).message } }));
    } finally { setTesting(null); }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Remove this compute instance?")) return;
    try {
      await deleteCompute(id);
      await load();
    } catch (e: unknown) { setError((e as Error).message); }
  };

  const set = (k: keyof AddComputeRequest) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(f => ({ ...f, [k]: k === "port" ? parseInt(e.target.value) || 22 : e.target.value }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Compute Instances</h1>
          <p className="text-gray-500 text-sm mt-0.5">SSH into remote GPU machines for training</p>
        </div>
        <button onClick={() => setShowForm(s => !s)}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 text-sm font-medium hover:bg-blue-700 transition-colors">
          <Plus size={14} /> Add Instance
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-3 bg-red-50 border border-red-200 p-4">
          <AlertCircle size={15} className="text-red-500 mt-0.5 shrink-0" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {/* Add form */}
      {showForm && (
        <div className="bg-white border border-gray-200 shadow-sm p-6 space-y-4">
          <h2 className="text-sm font-semibold text-gray-700">New Compute Instance</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1.5">Name</label>
              <input value={form.name} onChange={set("name")} placeholder="My GPU Server"
                className="w-full border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1.5">Host / IP</label>
              <input value={form.host} onChange={set("host")} placeholder="192.168.1.100 or gpu.example.com"
                className="w-full border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1.5">Username</label>
              <input value={form.username} onChange={set("username")} placeholder="ubuntu"
                className="w-full border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1.5">Port</label>
              <input type="number" value={form.port} onChange={set("port")} placeholder="22"
                className="w-full border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div className="col-span-2">
              <label className="text-xs font-medium text-gray-500 block mb-1.5">
                SSH Key Path <span className="text-gray-400 font-normal">(path on the server running LLM Studio, e.g. ~/.ssh/id_rsa)</span>
              </label>
              <input value={form.key_path ?? ""} onChange={set("key_path")} placeholder="~/.ssh/id_rsa"
                className="w-full border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
          </div>
          <div className="flex gap-3 pt-2">
            <button onClick={handleAdd} disabled={saving || !form.name || !form.host || !form.username}
              className="bg-blue-600 text-white px-4 py-2 text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-40">
              {saving ? "Saving…" : "Save Instance"}
            </button>
            <button onClick={() => { setShowForm(false); setForm(EMPTY_FORM); }}
              className="border border-gray-300 text-gray-600 px-4 py-2 text-sm hover:bg-gray-50 transition-colors">
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Instance list */}
      {loading ? (
        <p className="text-gray-400 text-sm">Loading…</p>
      ) : instances.length === 0 ? (
        <div className="bg-white border border-gray-200 p-12 text-center shadow-sm">
          <Server size={32} className="text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500 text-sm font-medium">No compute instances yet</p>
          <p className="text-gray-400 text-xs mt-1">Add a remote GPU server to offload training</p>
        </div>
      ) : (
        <div className="space-y-3">
          {instances.map(inst => (
            <div key={inst.id} className="bg-white border border-gray-200 shadow-sm p-5">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <Server size={18} className="text-gray-400 shrink-0" />
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-gray-900 text-sm">{inst.name}</p>
                      <StatusIcon status={inst.last_status} />
                      <span className={`text-xs font-medium ${
                        inst.last_status === "connected" ? "text-green-600"
                        : inst.last_status === "error"   ? "text-red-600"
                        : "text-gray-400"
                      }`}>{inst.last_status}</span>
                    </div>
                    <p className="text-gray-500 text-xs mt-0.5 font-mono">
                      {inst.username}@{inst.host}:{inst.port}
                      {inst.key_path && <span className="ml-2 text-gray-400">key: {inst.key_path}</span>}
                    </p>
                    {inst.last_checked && (
                      <p className="text-gray-400 text-xs mt-0.5">
                        Last tested: {new Date(inst.last_checked).toLocaleString()}
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button onClick={() => handleTest(inst.id)} disabled={testing === inst.id}
                    className="flex items-center gap-1.5 border border-gray-300 text-gray-600 px-3 py-1.5 text-xs font-medium hover:bg-gray-50 transition-colors disabled:opacity-40">
                    <Zap size={11} />
                    {testing === inst.id ? "Testing…" : "Test Connection"}
                  </button>
                  <button onClick={() => handleDelete(inst.id)}
                    className="border border-red-200 text-red-500 px-2 py-1.5 text-xs hover:bg-red-50 transition-colors">
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>

              {/* Test result */}
              {testResults[inst.id] && (
                <div className={`mt-3 p-3 text-xs border ${testResults[inst.id].success ? "bg-green-50 border-green-200 text-green-700" : "bg-red-50 border-red-200 text-red-700"}`}>
                  {testResults[inst.id].success ? "✓ " : "✗ "}
                  {testResults[inst.id].message}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* How it works */}
      <div className="bg-gray-50 border border-gray-200 p-5 text-sm space-y-2">
        <p className="font-semibold text-gray-700">How remote training works</p>
        <ol className="space-y-1 text-gray-500 text-xs list-decimal list-inside">
          <li>LLM Studio SSHes into the remote instance using your key</li>
          <li>Uploads training data and a self-contained training script via SFTP</li>
          <li>Installs PyTorch + Transformers on the remote (cached after first run)</li>
          <li>Runs training; logs stream back in real time</li>
          <li>Downloads model artifacts back via SFTP when complete</li>
        </ol>
        <p className="text-gray-400 text-xs pt-1">
          Requirements on the remote: Python 3.8+, pip, internet access for pip install.
          GPU (CUDA) is auto-detected and used if available.
        </p>
      </div>
    </div>
  );
}

"use client";

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type { LossCurves } from "@/lib/api";

export function LossCurveChart({ curves }: { curves: LossCurves }) {
  if (!curves?.epochs?.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400 text-sm border-2 border-dashed border-gray-200 ">
        No training data yet
      </div>
    );
  }

  const data = curves.epochs.map((epoch, i) => ({
    epoch,
    "Train Loss": curves.train_loss?.[i] != null ? +curves.train_loss[i].toFixed(4) : undefined,
    "Val Loss":   curves.val_loss?.[i]   != null ? +curves.val_loss[i].toFixed(4)   : undefined,
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
        <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 6, fontSize: 12 }}
          labelStyle={{ color: "#6b7280" }}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "#6b7280" }} />
        <Line type="monotone" dataKey="Train Loss" stroke="#3b82f6" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="Val Loss"   stroke="#10b981" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

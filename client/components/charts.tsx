"use client";

// Chart conventions (applied everywhere): one y-axis per chart (two measures => two
// charts), 2px lines, >=8px markers, recessive grid, tooltips on hover, values in
// ink colours, and a table view for every chart so nothing depends on colour alone.

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AXIS = { stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 12 }, tickLine: false };
const TOOLTIP = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--ink)",
    fontSize: 12,
  },
  labelStyle: { color: "var(--ink-2)" },
  itemStyle: { color: "var(--ink)" },
};

function TableToggle({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button type="button" onClick={() => setOpen(!open)} className="text-xs text-muted underline-offset-2 hover:underline">
        {open ? "Hide table" : "Show as table"}
      </button>
      {open && <div className="mt-2 overflow-x-auto">{children}</div>}
    </div>
  );
}

function DataTable({ headers, rows }: { headers: string[]; rows: (string | number)[][] }) {
  return (
    <table className="tabular w-full text-left text-xs">
      <thead className="text-muted">
        <tr>{headers.map((h) => <th key={h} className="py-1 pr-4 font-medium">{h}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-t border-line">
            {r.map((c, j) => <td key={j} className="py-1 pr-4">{c}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** One measure over sessions (single series: the title names it, no legend). */
export function LineTrend({
  data,
  dataKey,
  domain,
  unit = "",
  height = 180,
}: {
  data: Record<string, string | number | null>[];
  dataKey: string;
  domain?: [number, number];
  unit?: string;
  height?: number;
}) {
  const points = data.filter((d) => d[dataKey] != null);
  if (points.length === 0) return <p className="text-sm text-muted">No data yet.</p>;
  return (
    <>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="label" {...AXIS} />
            <YAxis domain={domain ?? ["auto", "auto"]} {...AXIS} axisLine={false} width={44} />
            <Tooltip {...TOOLTIP} cursor={{ stroke: "var(--axis)" }} formatter={(v) => [`${v}${unit}`, ""]} />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke="var(--series-1)"
              strokeWidth={2}
              dot={{ r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 }}
              activeDot={{ r: 6 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <TableToggle>
        <DataTable headers={["Session", dataKey]} rows={points.map((p) => [String(p.label), `${p[dataKey]}${unit}`])} />
      </TableToggle>
    </>
  );
}

/** Horizontal bars for one measure across categories (e.g. score per topic). */
export function HBars({
  data,
  domain = [0, 5],
  height,
}: {
  data: { label: string; value: number }[];
  domain?: [number, number];
  height?: number;
}) {
  if (data.length === 0) return <p className="text-sm text-muted">No data yet.</p>;
  return (
    <>
      <div style={{ height: height ?? Math.max(120, data.length * 34) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 36, bottom: 0, left: 0 }} barCategoryGap={8}>
            <CartesianGrid stroke="var(--grid)" horizontal={false} />
            <XAxis type="number" domain={domain} {...AXIS} />
            <YAxis type="category" dataKey="label" {...AXIS} axisLine={false} width={130} />
            <Tooltip {...TOOLTIP} cursor={{ fill: "var(--surface-2)" }} formatter={(v) => [Number(v).toFixed(2), "score"]} />
            <Bar dataKey="value" fill="var(--series-1)" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false}>
              <LabelList dataKey="value" position="right" fill="var(--ink-2)" fontSize={12} formatter={(v) => Number(v).toFixed(1)} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <TableToggle>
        <DataTable headers={["", "Score"]} rows={data.map((d) => [d.label, d.value.toFixed(2)])} />
      </TableToggle>
    </>
  );
}

const STAGES = [
  { key: "turn", label: "Turn detection", color: "var(--series-1)" },
  { key: "llm", label: "LLM", color: "var(--series-2)" },
  { key: "tts", label: "TTS", color: "var(--series-3)" },
  { key: "output", label: "Output", color: "var(--series-4)" },
] as const;

/** Median latency per pipeline stage as one stacked bar, with legend, labels and table. */
export function LatencyStages({ stages }: { stages: Record<string, number | null> }) {
  const row = Object.fromEntries(STAGES.map((s) => [s.key, Math.round(stages[s.key] ?? 0)]));
  const total = STAGES.reduce((acc, s) => acc + (row[s.key] as number), 0);
  if (!total) return <p className="text-sm text-muted">No latency data.</p>;
  return (
    <>
      <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        {STAGES.map((s) => (
          <li key={s.key} className="flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} />
            {s.label} <span className="tabular text-ink">{row[s.key]} ms</span>
          </li>
        ))}
      </ul>
      <div style={{ height: 56 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={[{ name: "p50", ...row }]} layout="vertical" margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
            <XAxis type="number" hide domain={[0, total]} />
            <YAxis type="category" dataKey="name" hide />
            <Tooltip {...TOOLTIP} cursor={false} formatter={(v, name) => [`${v} ms`, STAGES.find((s) => s.key === name)?.label]} />
            {STAGES.map((s, i) => (
              <Bar
                key={s.key}
                dataKey={s.key}
                stackId="latency"
                fill={s.color}
                stroke="var(--surface)"
                strokeWidth={2}
                radius={i === STAGES.length - 1 ? [0, 4, 4, 0] : i === 0 ? [4, 0, 0, 4] : 0}
                isAnimationActive={false}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <TableToggle>
        <DataTable headers={["Stage", "p50 (ms)"]} rows={STAGES.map((s) => [s.label, row[s.key] as number])} />
      </TableToggle>
    </>
  );
}

"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { api, formatDate, formatDuration, type Progress, ROLE_LABELS, type SessionSummary } from "@/lib/api";

import { HBars, LineTrend } from "./charts";
import { Card, CardTitle, fmt, StatusBadge } from "./ui";

export function Dashboard() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.listSessions(), api.progress()])
      .then(([s, p]) => {
        setSessions(s);
        setProgress(p);
      })
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <Card>
        <p className="text-sm text-bad">Can&apos;t reach the server ({error}). Is the API running on port 7860?</p>
      </Card>
    );
  }

  const trend = (progress?.sessions ?? []).map((p, i) => ({
    label: `#${i + 1}`,
    score: p.score,
    wpm: p.wpm,
    fillers: p.fillers_per_min,
  }));

  return (
    <div className="space-y-6">
      {trend.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardTitle hint="1 to 5, averaged across answers">Overall score</CardTitle>
            <LineTrend data={trend} dataKey="score" domain={[1, 5]} />
          </Card>
          <Card>
            <CardTitle hint="Averaged across all your answers">Score by topic</CardTitle>
            <HBars data={(progress?.topics ?? []).map((t) => ({ label: t.topic, value: t.avg_score }))} />
          </Card>
          <Card>
            <CardTitle hint="Conversational pace is about 120 to 160">Speaking pace (words/min)</CardTitle>
            <LineTrend data={trend} dataKey="wpm" />
          </Card>
          <Card>
            <CardTitle hint="um, uh, you know, basically…">Filler words per minute</CardTitle>
            <LineTrend data={trend} dataKey="fillers" />
          </Card>
        </div>
      )}

      <Card>
        <div id="history" />
        <CardTitle hint={sessions ? `${sessions.length} total` : undefined}>History</CardTitle>
        {sessions === null ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : sessions.length === 0 ? (
          <p className="text-sm text-muted">No interviews yet. Your first one is a click away.</p>
        ) : (
          <ul className="divide-y divide-line">
            {sessions.map((s) => (
              <li key={s.id} className="flex items-center justify-between gap-4 py-2.5 text-sm">
                <Link href={`/report/${s.id}`} className="min-w-0 flex-1 hover:underline">
                  <span className="font-medium">{ROLE_LABELS[s.role]}</span>
                  <span className="text-ink-2"> · {s.level} · {s.mode}</span>
                </Link>
                <span className="tabular hidden text-ink-2 sm:inline">{formatDuration(s.duration_secs)}</span>
                <span className="tabular w-12 text-right font-medium">{fmt(s.overall_score)}</span>
                <StatusBadge status={s.status} />
                <span className="hidden w-32 text-right text-xs text-muted md:inline">{formatDate(s.created_at)}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

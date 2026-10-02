"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { type AnswerReport, api, formatDate, formatDuration, ROLE_LABELS, type SessionDetail } from "@/lib/api";

import { HBars, LatencyStages } from "./charts";
import { Button, Card, CardTitle, fmt, Stat, StatusBadge } from "./ui";

const PENDING = new Set(["created", "live", "ended", "analyzing"]);
const DIMENSIONS = ["correctness", "depth", "structure", "communication"] as const;

export function ReportView({ id }: { id: string }) {
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;
    const load = async () => {
      try {
        const s = await api.getSession(id);
        if (cancelled) return;
        setSession(s);
        if (PENDING.has(s.status)) timer = setTimeout(load, 3000);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id]);

  if (error) return <Card><p className="text-sm text-bad">{error}</p></Card>;
  if (!session) return <p className="text-sm text-muted">Loading…</p>;

  const report = session.report;
  const header = (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <Link href="/" className="text-sm text-ink-2 hover:underline">← All interviews</Link>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">
          {ROLE_LABELS[session.role]} · {session.level} · {session.mode}
        </h1>
        <p className="text-sm text-ink-2">
          {formatDate(session.created_at)} · {formatDuration(session.duration_secs)}
        </p>
      </div>
      <StatusBadge status={session.status} />
    </div>
  );

  if (report?.overall && report.answers.length === 0) {
    return (
      <div className="space-y-6">
        {header}
        <Card>
          <p className="text-sm text-ink-2">
            No questions were answered in this session ({report.end_reason ?? "the call ended early"}), so there is
            nothing to score. <Link href="/" className="text-accent underline">Start a new interview</Link>.
          </p>
        </Card>
      </div>
    );
  }

  if (PENDING.has(session.status) || !report?.overall) {
    return (
      <div className="space-y-6">
        {header}
        <Card>
          {session.status === "failed" ? (
            <div className="space-y-3">
              <p className="text-sm text-bad">Analysis failed: {session.error}</p>
              <Button variant="ghost" onClick={() => api.reanalyze(id).then(() => location.reload())}>
                Retry analysis
              </Button>
            </div>
          ) : session.status === "empty" ? (
            <p className="text-sm text-ink-2">Nothing was said in this session, so there is nothing to score.</p>
          ) : (
            <div className="flex items-center gap-3 text-sm text-ink-2">
              <span aria-hidden className="h-2 w-2 animate-pulse rounded-full bg-accent" />
              Scoring your answers and analysing your delivery. This usually takes a minute or two.
            </div>
          )}
        </Card>
      </div>
    );
  }

  const { overall, speech, latency } = report;
  return (
    <div className="space-y-6">
      {header}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Overall score" value={`${fmt(overall.score)} / 5`} />
        <Stat label="Key points covered" value={overall.coverage == null ? "–" : `${Math.round(overall.coverage * 100)}%`} />
        <Stat label="Speaking pace" value={fmt(speech.wpm, 0)} sub="words per minute" />
        <Stat label="Filler words" value={fmt(speech.fillers_per_min, 1)} sub="per minute" />
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardTitle hint="1 = weak · 5 = excellent">Score by dimension</CardTitle>
          <HBars data={DIMENSIONS.map((d) => ({ label: d[0].toUpperCase() + d.slice(1), value: overall.dimensions[d] ?? 0 }))} />
        </Card>
        <Card>
          <CardTitle hint={speech.source}>Delivery</CardTitle>
          <dl className="tabular grid grid-cols-2 gap-y-2 text-sm">
            <dt className="text-ink-2">Long pauses (over 3 s)</dt>
            <dd className="text-right">{speech.long_pauses}</dd>
            <dt className="text-ink-2">Longest pause</dt>
            <dd className="text-right">{fmt(speech.longest_pause_secs, 1, " s")}</dd>
            <dt className="text-ink-2">Avg. time before answering</dt>
            <dd className="text-right">{fmt(speech.avg_response_delay_secs, 1, " s")}</dd>
            <dt className="text-ink-2">Your talk time / interviewer&apos;s</dt>
            <dd className="text-right">{fmt(speech.talk_ratio ?? null, 1, "×")}</dd>
            <dt className="text-ink-2">Fillers used</dt>
            <dd className="text-right">
              {Object.entries(speech.filler_counts).length
                ? Object.entries(speech.filler_counts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([w, n]) => `“${w}” ×${n}`)
                    .join(", ")
                : "none"}
            </dd>
          </dl>
        </Card>
      </div>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold tracking-tight">Answers</h2>
        {report.answers.map((a, i) => (
          <AnswerCard key={a.question_id} index={i + 1} answer={a} />
        ))}
      </section>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardTitle hint={`p50 ${fmt(latency.p50_ms, 0, " ms")} · p95 ${fmt(latency.p95_ms, 0, " ms")} · ${latency.turns} turns`}>
            Interviewer response time
          </CardTitle>
          <LatencyStages stages={latency.stages_p50_ms} />
        </Card>
        {session.has_audio && (
          <Card>
            <CardTitle>Recording</CardTitle>
            <audio controls preload="none" className="w-full" src={`/api/sessions/${id}/audio`} />
          </Card>
        )}
      </div>

      <details className="rounded-xl border border-line bg-surface p-5">
        <summary className="cursor-pointer text-sm font-medium">Full transcript</summary>
        <ol className="mt-3 space-y-2 text-sm">
          {session.turns.map((t) => (
            <li key={t.idx} className="flex gap-3">
              <span className="tabular w-12 shrink-0 text-xs text-muted">{formatDuration(t.start_ms / 1000)}</span>
              <span className="w-12 shrink-0 font-medium">{t.speaker === "user" ? "You" : "Alex"}</span>
              <span className="text-ink-2">
                {t.text}
                {t.interrupted && <em className="text-muted"> (interrupted)</em>}
              </span>
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}

function AnswerCard({ index, answer: a }: { index: number; answer: AnswerReport }) {
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs text-muted">
            Q{index} · {a.topic}
            {a.type === "behavioral" && " · behavioral (STAR)"}
          </div>
          <h3 className="mt-0.5 font-medium">{a.question}</h3>
        </div>
        <div className="text-right">
          <div className="tabular text-xl font-semibold">{a.mean_score.toFixed(1)}</div>
          <div className="text-xs text-muted">{Math.round(a.coverage * 100)}% of key points</div>
        </div>
      </div>

      <div className="tabular mt-3 flex flex-wrap gap-2 text-xs">
        {DIMENSIONS.map((d) => (
          <span key={d} className="rounded-md bg-surface-2 px-2 py-1 text-ink-2">
            {d} <strong className="text-ink">{a.score[d]}</strong>
          </span>
        ))}
        {a.speech.wpm != null && (
          <span className="rounded-md bg-surface-2 px-2 py-1 text-ink-2">
            {Math.round(a.speech.wpm)} wpm · {a.speech.filler_count} fillers
          </span>
        )}
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <h4 className="mb-1 text-xs font-medium text-muted">Covered</h4>
          <ul className="space-y-1 text-sm">
            {a.score.covered_points.length === 0 && <li className="text-muted">None</li>}
            {a.score.covered_points.map((p) => (
              <li key={p} className="flex gap-2">
                <span aria-hidden className="text-good">✓</span>
                <span>{p}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="mb-1 text-xs font-medium text-muted">Missed</h4>
          <ul className="space-y-1 text-sm">
            {a.score.missed_points.length === 0 && <li className="text-muted">Nothing. Well done.</li>}
            {a.score.missed_points.map((p) => (
              <li key={p} className="flex gap-2">
                <span aria-hidden className="text-bad">✗</span>
                <span>{p}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="mt-4 rounded-lg bg-surface-2 px-4 py-3 text-sm">
        <span className="font-medium">A stronger answer: </span>
        <span className="text-ink-2">{a.score.better_answer_outline}</span>
      </div>

      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-xs text-muted">Your answer (transcript)</summary>
        <p className="mt-2 text-ink-2">{a.answer_text || <em>No answer recorded.</em>}</p>
        {a.follow_up_question && (
          <>
            <p className="mt-2 font-medium">Follow-up: {a.follow_up_question}</p>
            <p className="mt-1 text-ink-2">{a.follow_up_answer || <em>No answer recorded.</em>}</p>
          </>
        )}
      </details>
    </Card>
  );
}

// Typed client for the FastAPI backend (proxied at /api by next.config.ts).

export type Role = "ai-engineer" | "backend";
export type Level = "junior" | "mid" | "senior";
export type Mode = "technical" | "behavioral";
export type SessionStatus = "created" | "live" | "ended" | "empty" | "analyzing" | "done" | "failed";

export interface CreateSessionRequest {
  role: Role;
  level: Level;
  mode: Mode;
  job_posting?: string;
  num_questions: number;
  max_minutes: number;
}

export interface SessionSummary {
  id: string;
  created_at: string;
  role: Role;
  level: Level;
  mode: Mode;
  status: SessionStatus;
  duration_secs: number | null;
  overall_score: number | null;
}

export interface Turn {
  idx: number;
  speaker: "user" | "bot";
  text: string;
  phase: string;
  question_id: string | null;
  start_ms: number;
  end_ms: number;
  interrupted: boolean;
}

export interface SpeechMetrics {
  words: number;
  speaking_secs: number;
  wpm: number | null;
  filler_count: number;
  fillers_per_min: number | null;
  filler_counts: Record<string, number>;
  long_pauses: number;
  longest_pause_secs: number;
  avg_response_delay_secs: number | null;
  talk_ratio?: number | null;
  source?: string;
}

export interface AnswerScore {
  question_id: string;
  covered_points: string[];
  missed_points: string[];
  correctness: number;
  depth: number;
  structure: number;
  communication: number;
  better_answer_outline: string;
}

export interface AnswerReport {
  question_id: string;
  question: string;
  topic: string;
  type: string;
  key_points: string[];
  answer_text: string;
  follow_up_question: string | null;
  follow_up_answer: string;
  score: AnswerScore;
  mean_score: number;
  coverage: number;
  speech: SpeechMetrics;
}

export interface Report {
  overall: { score: number | null; coverage: number | null; dimensions: Record<string, number> };
  answers: AnswerReport[];
  speech: SpeechMetrics;
  latency: { turns: number; p50_ms: number | null; p95_ms: number | null; stages_p50_ms: Record<string, number | null> };
  barge_in: { count: number; median_ms: number | null };
  judge_model: string;
  end_reason?: string;
}

export interface SessionDetail extends SessionSummary {
  started_at: string | null;
  ended_at: string | null;
  job_posting: string | null;
  turns: Turn[];
  report: Report | null;
  error: string | null;
  has_audio: boolean;
}

export interface ProgressPoint {
  id: string;
  created_at: string;
  role: Role;
  level: Level;
  score: number | null;
  coverage: number | null;
  wpm: number | null;
  fillers_per_min: number | null;
  latency_p50_ms: number | null;
}

export interface Progress {
  sessions: ProgressPoint[];
  topics: { topic: string; avg_score: number; answers: number }[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createSession: (body: CreateSessionRequest) =>
    request<{ id: string; num_questions: number; topics: string[] }>("/api/sessions", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listSessions: () => request<SessionSummary[]>("/api/sessions"),
  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),
  deleteSession: (id: string) => request<{ status: string }>(`/api/sessions/${id}`, { method: "DELETE" }),
  reanalyze: (id: string) => request<{ status: string }>(`/api/sessions/${id}/analyze`, { method: "POST" }),
  progress: () => request<Progress>("/api/progress"),
};

export const ROLE_LABELS: Record<Role, string> = { "ai-engineer": "AI Engineer", backend: "Backend" };

export function formatDuration(secs: number | null | undefined): string {
  if (secs == null) return "–";
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

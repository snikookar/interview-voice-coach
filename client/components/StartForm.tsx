"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { api, type CreateSessionRequest, type Level, type Mode, type Role } from "@/lib/api";

import { Button, Card, CardTitle } from "./ui";

const ROLES: { value: Role; label: string }[] = [
  { value: "ai-engineer", label: "AI Engineer" },
  { value: "backend", label: "Backend Engineer" },
];
const LEVELS: Level[] = ["junior", "mid", "senior"];

function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <fieldset>
      <legend className="mb-1.5 text-sm font-medium">{label}</legend>
      <div className="inline-flex rounded-lg border border-line bg-surface-2 p-1">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            aria-pressed={value === o.value}
            onClick={() => onChange(o.value)}
            className={`rounded-md px-3 py-1.5 text-sm transition ${
              value === o.value ? "bg-surface font-medium shadow-sm" : "text-ink-2 hover:text-ink"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function StartForm() {
  const router = useRouter();
  const [form, setForm] = useState<CreateSessionRequest>({
    role: "ai-engineer",
    level: "mid",
    mode: "technical",
    num_questions: 5,
    max_minutes: 12,
    job_posting: "",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const session = await api.createSession({
        ...form,
        job_posting: form.job_posting?.trim() || undefined,
      });
      router.push(`/interview?session=${session.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardTitle hint="Questions are picked for you, never shown in advance">New interview</CardTitle>
      <form onSubmit={start} className="space-y-5">
        <div className="flex flex-wrap gap-6">
          <Segmented label="Role" options={ROLES} value={form.role} onChange={(role) => setForm({ ...form, role })} />
          <Segmented
            label="Level"
            options={LEVELS.map((l) => ({ value: l, label: l[0].toUpperCase() + l.slice(1) }))}
            value={form.level}
            onChange={(level) => setForm({ ...form, level })}
          />
          <Segmented<Mode>
            label="Mode"
            options={[
              { value: "technical", label: "Technical" },
              { value: "behavioral", label: "Behavioral" },
            ]}
            value={form.mode}
            onChange={(mode) => setForm({ ...form, mode })}
          />
        </div>

        <div className="flex flex-wrap gap-6">
          <label className="text-sm">
            <span className="mb-1.5 block font-medium">Questions</span>
            <input
              type="number"
              min={1}
              max={10}
              value={form.num_questions}
              onChange={(e) => setForm({ ...form, num_questions: Number(e.target.value) })}
              className="w-24 rounded-lg border border-line bg-surface px-3 py-2"
            />
          </label>
          <label className="text-sm">
            <span className="mb-1.5 block font-medium">Time limit (min)</span>
            <input
              type="number"
              min={3}
              max={45}
              value={form.max_minutes}
              onChange={(e) => setForm({ ...form, max_minutes: Number(e.target.value) })}
              className="w-24 rounded-lg border border-line bg-surface px-3 py-2"
            />
          </label>
        </div>

        <label className="block text-sm">
          <span className="mb-1.5 block font-medium">
            Job posting <span className="font-normal text-muted">(optional: questions will target it)</span>
          </span>
          <textarea
            rows={5}
            value={form.job_posting}
            onChange={(e) => setForm({ ...form, job_posting: e.target.value })}
            placeholder="Paste the job description here…"
            className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm"
          />
        </label>

        {error && <p className="rounded-lg bg-bad-bg px-3 py-2 text-sm text-bad">Couldn&apos;t start: {error}</p>}

        <div className="flex items-center gap-3">
          <Button type="submit" disabled={busy}>
            {busy ? "Preparing questions…" : "Start interview"}
          </Button>
          <span className="text-xs text-muted">You&apos;ll be asked for microphone access on the next page.</span>
        </div>
      </form>
    </Card>
  );
}

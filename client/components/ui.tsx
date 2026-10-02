import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-xl border border-line bg-surface p-5 ${className}`}>{children}</section>;
}

export function CardTitle({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-4 flex items-baseline justify-between gap-4">
      <h2 className="text-base font-semibold tracking-tight">{children}</h2>
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </div>
  );
}

/** A labelled number. Values stay in ink colours; colour never carries meaning alone. */
export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-lg bg-surface-2 px-4 py-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold tracking-tight">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-2">{sub}</div>}
    </div>
  );
}

const STATUS_STYLE: Record<string, string> = {
  done: "bg-good-bg text-good",
  failed: "bg-bad-bg text-bad",
  analyzing: "bg-warn-bg text-ink-2",
  ended: "bg-warn-bg text-ink-2",
  live: "bg-surface-2 text-accent",
};

export function StatusBadge({ status }: { status: string }) {
  const label = status === "ended" ? "queued" : status;
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLE[status] ?? "bg-surface-2 text-muted"}`}>
      {label}
    </span>
  );
}

export function Button({
  children,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "danger" }) {
  const styles = {
    primary: "bg-accent text-accent-ink hover:opacity-90",
    ghost: "border border-line bg-surface hover:bg-surface-2",
    danger: "bg-bad text-white hover:opacity-90",
  }[variant];
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${props.className ?? ""}`}
    >
      {children}
    </button>
  );
}

export function fmt(value: number | null | undefined, digits = 1, suffix = ""): string {
  return value == null ? "–" : `${value.toFixed(digits)}${suffix}`;
}

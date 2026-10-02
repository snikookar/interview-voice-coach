import { Dashboard } from "@/components/Dashboard";
import { StartForm } from "@/components/StartForm";

export default function Home() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Practice interviews, out loud.</h1>
        <p className="mt-1 max-w-2xl text-ink-2">
          A voice interviewer asks role-specific questions, follows up when you miss a key point, and lets you
          interrupt it like a real person. Afterwards you get rubric scores and delivery metrics.
        </p>
      </div>
      <StartForm />
      <Dashboard />
    </div>
  );
}

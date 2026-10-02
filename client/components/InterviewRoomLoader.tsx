"use client";

import dynamic from "next/dynamic";

// The voice client only exists in the browser, so skip server rendering entirely.
const InterviewRoom = dynamic(() => import("./InterviewRoom"), {
  ssr: false,
  loading: () => <p className="text-sm text-muted">Loading…</p>,
});

export function InterviewRoomLoader({ sessionId }: { sessionId: string }) {
  return <InterviewRoom sessionId={sessionId} />;
}

# Phase 7: Next.js client

**Goal:** a browser UI to start an interview, talk to the interviewer, and read the report and your progress over time.

## Files

| File | Purpose |
|---|---|
| `client/next.config.ts` | Rewrites `/api/*` to FastAPI (REST **and** the WebRTC SDP/ICE exchange) |
| `client/lib/api.ts` | Typed API client mirroring the server's report schema |
| `client/app/page.tsx` + `components/StartForm.tsx`, `Dashboard.tsx` | Start form, progress charts, history |
| `client/app/interview/page.tsx` + `components/InterviewRoom*.tsx` | The live call: Pipecat client, live transcript, timer, question progress, voice visualisers, mute and end, last-turn latency |
| `client/app/report/[id]/page.tsx` + `components/ReportView.tsx` | Polls until the analysis is done. Stat tiles, per-dimension scores, per-answer cards (covered ✓ / missed ✗ / stronger-answer outline), delivery, latency stages, recording, full transcript |
| `client/components/charts.tsx` | Recharts wrappers: `LineTrend`, `HBars`, `LatencyStages`, each with a "Show as table" view |
| `client/app/globals.css` | Design tokens for light and dark (Tailwind v4 `@theme`) |

## How the live page works
```
click "Start interview"
  → client.initDevices()                        (mic permission)
  → client.connect({ webrtcRequestParams: { endpoint: "/api/offer", requestData: { session_id } } })
        SDP offer → Next rewrite → FastAPI → SmallWebRTC answer; trickle ICE via PATCH /api/offer
RTVI events on the WebRTC data channel:
  userTranscript (final)          → your bubble
  botOutput (spoken_status=completed) → interviewer bubble
  serverMessage {type:"progress"}  → "Question 2 of 5 · rag · follow-up"
  serverMessage {type:"latency"}   → last-turn latency card
  disconnected (interviewer said goodbye, or you pressed End) → /report/<id>
```

## Technology choices

### Next.js (App Router) + TypeScript
The spec asks for it, and it fits: server components for the thin page shells (`await params` / `await searchParams`, Next 16 style), client components for everything interactive, and **rewrites** to put the API behind the same origin. One origin means no CORS configuration and no API URL compiled into the bundle.

### `@pipecat-ai/client-js` + `client-react` + `small-webrtc-transport`
These are the client half of the RTVI protocol the server already speaks. `PipecatClientAudio` plays the bot's track, `useRTVIClientEvent` subscribes to transcripts and server messages, `VoiceVisualizer` draws live levels, and `usePipecatClientMicControl` handles mute. Writing raw `RTCPeerConnection` code would duplicate all of that.

The Pipecat client touches `navigator` and WebRTC, so the room is loaded with `next/dynamic({ ssr: false })` and the client is created once in a `useState` initializer. That also satisfies React's `set-state-in-effect` lint rule, which the first version (creating it in `useEffect`) violated.

### Recharts
Declarative React charts, SVG output, and tooltips built in. Chart.js renders to canvas, which is harder to style with CSS variables and invisible to screen readers. D3 would mean hand-building axes for four simple chart types.

Chart rules applied (from a data-viz checklist):
- **One measure per chart, never dual axes.** Pace and fillers are two separate charts, not one chart with two y-scales.
- **Single-series charts get no legend** (the title names the measure). The multi-series latency stack gets a legend *and* direct value labels.
- **The stage colours were validated with a colour-vision-deficiency checker** in both modes (worst adjacent CVD ΔE 9.1 light / 8.4 dark). Two light-mode colours fall below 3:1 contrast, so the chart carries visible labels and a table view.
- **Every chart has a "Show as table" view**, so nothing relies on colour or hover alone.
- Values are written in text colours, never in the series colour.

### Tailwind v4 with CSS-variable tokens
All colours are tokens in `globals.css`, with a dark theme under `prefers-color-scheme`. Components use semantic names (`bg-surface`, `text-ink-2`, `text-good`), so the dark mode is designed rather than inverted, and charts use the same variables (`var(--series-1)`).

## Bugs found by testing in a real browser
1. **Every interviewer sentence appeared twice.** RTVI sends each sentence as `spoken_status: "new"` (queued for TTS), then `"completed"` (played). The transcript now shows only completed text, which also keeps interrupted replies truncated at the point you cut in.
2. **A session with no answers showed empty charts and dashes.** It now gets a clear "nothing to score" state.
3. **Sessions stuck as "queued" forever** (they were recorded before the analyzer existed, or the server restarted mid-analysis). The server now recovers them at startup: re-queues pending analyses and fails calls that were live. Recovering 11 sessions at once then exposed a **thread-safety bug**: `openai` imports `httpx` lazily, which raced with offline Whisper importing it in a worker thread ("partially initialized module"). The fix is eager imports plus running one analysis at a time.

# Phase 2: Latency measurement and tracing

**Goal:** every bot turn produces a per-stage latency breakdown, so the README claim ("sub-1.5 s, measured per stage") is backed by data rather than vibes.

## Files

| File | Purpose |
|---|---|
| `server/metrics_observer.py` | `LatencyRecorder` gives the per-turn breakdown folded into the spec's four stages (turn, STT, LLM, TTS). `BargeInObserver` measures interruption-to-silence time |
| `server/tracing.py` | Optional Langfuse export through OpenTelemetry |
| `server/tests/test_metrics_observer.py` | Unit tests with synthetic breakdowns and a fake clock |
| `server/bot.py` (updated) | Registers the observers and pushes each turn's latency to the browser as an RTVI server message |

## How it works

Pipecat **observers** see every frame passed between processors, without being in the pipeline (zero added latency).

1. Pipecat's `UserBotLatencyObserver` timestamps moments like *VAD stop*, *transcript*, *LLM request*, *first LLM chunk*, *first audio* and *bot speaking*. From those it builds named spans (`endpointing_wait`, `transcription`, `llm_inference`, `speech_synthesis` and so on) that **sum exactly to the total**.
2. `LatencyRecorder` maps those spans onto the spec's budget table:

   | Spec stage | Pipecat spans |
   |---|---|
   | End-of-turn detection | `endpointing_wait`, `turn_detection`, `turn_completion` |
   | STT | `transcription` |
   | LLM time-to-first-token | `llm_inference`, `llm_tool_call`, `function_handler` |
   | TTS first audio | `sentence_aggregation`, `speech_synthesis` |

3. Each turn is logged, sent live to the client (`{"type":"latency", ...}`), and saved with the session (Phase 5). The benchmark script (Phase 8) computes p50/p95 from those records.

`BargeInObserver` starts a stopwatch on an `InterruptionFrame` that arrives *while the bot is speaking*, and stops it at the next `BotStoppedSpeakingFrame`. The spec target is under 300 ms.

## Technology choices

### Build on `UserBotLatencyObserver` rather than writing timestamps by hand
My first idea was a custom observer that records `UserStoppedSpeakingFrame → TranscriptionFrame → LLMTextFrame → TTSAudioRawFrame`. Pipecat 1.x already does this, **and** it handles the awkward cases: function calls in the middle of a turn, the greeting (measured from client connect, not user silence), and tiny spans rolled together. Re-implementing it would be more code with more bugs. I keep only the *mapping to the spec's stages*, which is project-specific.

### Server-side measurement rather than client-side
A client-side "mic silent → audio heard" measurement includes network round-trips, which vary with Wi-Fi. Server-side spans are reproducible and attributable to a stage. The README reports server-side numbers and says so.

### Tracing: Langfuse over **OpenTelemetry**, not the Langfuse SDK
Pipecat already emits OpenTelemetry spans for STT, LLM and TTS calls (`enable_tracing=True`). Langfuse accepts OTLP natively at `/api/public/otel`, so a standard exporter with a Basic-auth header is all it takes. That means no vendor SDK, and the same code works with Jaeger, Honeycomb or Grafana Tempo. It's optional, enabled only when `LANGFUSE_*` keys are set.

### Pushing latency to the browser with RTVI
RTVI is Pipecat's client↔server message protocol, and it's already running on the WebRTC data channel. An `RTVIServerMessageFrame` reaches the React client's `onServerMessage` callback. No second WebSocket is needed.

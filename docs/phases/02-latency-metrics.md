# Phase 2: Latency measurement and tracing

**Goal:** every interviewer turn produces a per-stage latency breakdown, so the README claim ("sub-1.5 s, measured per stage") is backed by data rather than vibes.

## Files

| File | Purpose |
|---|---|
| `server/metrics_observer.py` | `LatencyRecorder` gives the per-turn stage breakdown. `BargeInObserver` gives interruption-to-silence time |
| `server/tracing.py` | Optional Langfuse export through OpenTelemetry |
| `server/tests/test_metrics_observer.py` | Unit tests with a fake clock: stage arithmetic, resumed speech, ignored upstream frames, barge-in |
| `server/bot.py` (updated) | Registers the observers and pushes each turn's latency to the browser as an RTVI server message |

## How it works

Pipecat **observers** see every frame passed between processors without being in the pipeline (zero added latency). `LatencyRecorder` timestamps five moments per turn:

```
speech_end ──► turn_end ──► llm_done ──► first_audio ──► bot_speaking
   │   turn      │    llm     │    tts      │   output     │
```

| Moment | Frame |
|---|---|
| speech_end | `VADUserStoppedSpeakingFrame.timestamp − stop_secs` (when the voice actually stopped) |
| turn_end | `UserStoppedSpeakingFrame` (Smart Turn + min-silence + transcript gate released the turn) |
| llm_done | first `FunctionCallInProgressFrame` or `LLMTextFrame` (the LLM's decision) |
| first_audio | first `TTSAudioRawFrame` of the reply |
| bot_speaking | `BotStartedSpeakingFrame` |

Each stage is the gap between consecutive moments, so **the stages always sum to the total**. STT runs *in parallel* with the turn-detection wait, so "STT done +X ms" is reported separately rather than double-counted. A `VADUserStartedSpeakingFrame` (the candidate kept talking) restarts the measurement, so totals are always from the *last* time the voice stopped.

`BargeInObserver` starts a stopwatch on an `InterruptionFrame` that arrives *while the bot is speaking* and stops it at `BotStoppedSpeakingFrame`. The client-side view of barge-in comes from the synthetic candidate (Phase 5), which hears the actual audio.

## Technology choices, and a reversal

### First attempt: Pipecat's `UserBotLatencyObserver` (abandoned)
Pipecat 1.x ships an observer that builds named spans (`endpointing_wait`, `transcription`, `llm_inference`, ...). I initially just mapped those spans onto the spec's stages. The end-to-end test (Phase 5) proved that wrong for interviews. It anchors the measurement to the **first** pause of a turn, so an answer with pauses reported totals of 26 s, 50 s, 70 s, growing all session, and attributed whole seconds of the candidate's *own talking* to "LLM". Clipping its spans helped the totals but not the attribution.

The replacement is simpler (five timestamps, no span bookkeeping) and correct by construction. It's also checked against an independent measurement: the synthetic candidate's client-side voice-to-voice latency agrees within about 50 ms.

### Server-side and client-side measurement, both reported
Server timestamps attribute time to stages. The client's "my audio ended → bot audio heard" includes WebRTC buffering, which is what a user actually experiences. The README reports both.

### Tracing: Langfuse over **OpenTelemetry**, not the Langfuse SDK
Pipecat already emits OpenTelemetry spans for STT, LLM and TTS calls (`enable_tracing=True`). Langfuse accepts OTLP natively at `/api/public/otel`, so a standard exporter with a Basic-auth header is all it takes. That means no vendor SDK, and the same code works with Jaeger, Honeycomb or Grafana Tempo. It's optional, enabled only when `LANGFUSE_*` keys are set.

### Pushing latency to the browser with RTVI
RTVI is Pipecat's client↔server message protocol, and it's already running on the WebRTC data channel. An `RTVIServerMessageFrame` reaches the React client's `onServerMessage` callback. No second WebSocket is needed.

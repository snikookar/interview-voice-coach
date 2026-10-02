# Phase 5: Sessions, recording, and the end-to-end harness

**Goal:** persist every interview (timestamped transcript, audio, latency) behind a REST API, and prove the *whole* system works with a repeatable, automated end-to-end test.

## Files

| File | Purpose |
|---|---|
| `server/session_recorder.py` | Observer that rebuilds turns (speaker, text, phase, question, ms timestamps, interrupted flag), records **separate user and bot audio tracks**, and saves everything to Postgres |
| `server/api.py` (rewritten) | `POST /api/sessions` (builds the plan), `GET /api/sessions[/{id}]`, `GET /api/sessions/{id}/audio`, `DELETE`, `/api/meta`, plus WebRTC signalling keyed by `session_id` |
| `server/turn_strategy.py` | `MinSilenceTurnStopStrategy`: Smart Turn's verdict plus a minimum silence before the turn ends |
| `eval/mock_llm.py` | A scripted OpenAI-compatible server (streams real tool calls), so the full stack runs **without an API key** |
| `eval/synthetic_candidate.py` | A Python WebRTC client (aiortc) that creates a session, connects via `/api/offer` exactly like the browser, *speaks* Kokoro-synthesised answers (different voice), and measures voice-to-voice latency and barge-in from the client side |
| `server/tests/test_session_recorder.py` | Turn reconstruction: merged answer segments, interruption handling, noise |

### Session lifecycle
```
POST /api/sessions ─► plan built (RAG + MMR), status=created
browser POST /api/offer {requestData:{session_id}} ─► bot starts, status=live
call ends ─► turns + audio + latency saved, status=ended ─► (Phase 6) analysis ─► status=done
```
Questions are hidden from the API until the interview is over (no spoilers). Practice sessions avoid questions from your last 3 sessions.

### Recording design
- **Timestamps are ms since the call connected**, which is also when `AudioBufferProcessor` starts recording. So transcript times line up with `user.wav` for the offline speech analysis (Phase 6).
- **Separate tracks** (`user.wav`, `bot.wav`, plus `conversation.wav` for playback): speech metrics must only hear the candidate.
- Consecutive speech segments with no bot reply in between are **one answer**. An interruption closes the bot turn and drops text that was synthesised but never played.

## The end-to-end harness, and what it found

Unit tests passed from Phase 4 on, but **the first real end-to-end run measured 8.4 s voice-to-voice, cut answers off after every sentence, and once froze mid-interview.** Every number below comes from `synthetic_candidate.py` against the real server (CPU only; mock LLM with a fixed 0.4 s time-to-first-token so the LLM provider doesn't distort the comparison):

| # | Problem found | Root cause | Fix | Voice-to-voice |
|---|---|---|---|---|
| 0 | first run | | | 4.8 – 8.4 s |
| 1 | Answers cut off after each sentence; a `record_answer` fired after the goodbye | VAD 0.2 s + Smart Turn treat every *finished sentence* as end-of-turn (measured pauses between sentences: up to 0.7 s) | Turn stop gate, stale-call guards, `cancel_on_interruption` | ~3.9 s |
| 2 | Interview **froze**; 1.7 s silences mid-question | Flows waits for each `tts_say` to *finish playing*, and an interruption meant it never did. Kokoro on CPU synthesises a long sentence slower than the short one before it plays | Custom non-blocking `speak` action (text streamed like LLM output) | – |
| 3 | TTS still dominant | Kokoro RTF 0.59 vs Piper 0.07 on this CPU | Piper as CPU default | ~2.5 s |
| 4 | STT ~1.6 s | A long VAD `stop_secs` makes the *whole answer* one Whisper segment, transcribed only after it ends | VAD back to 0.2 s (Whisper transcribes sentence by sentence *during* the answer); the 0.8 s silence moved into a custom **turn stop strategy** | – |
| 5 | +0.8 s on every turn | The final transcript ends the turn, and Pipecat's default `TranscriptionUserTurnStartStrategy` then opened a phantom new turn that **cancelled the LLM request** | VAD-only turn start | **~1.6 s** |

Final per-stage breakdown (server side, typical turn):

| Stage | ms | Notes |
|---|---|---|
| turn | ~1 100 | ≥ 800 ms silence gate; here bounded by CPU Whisper finishing the last sentence |
| llm | ~430 | mock 400 ms TTFT; a real LLM adds its own |
| tts | ~70 | Piper first audio |
| output | ~3 | |
| **total** | **~1 600** | the client measures the same ±50 ms |

Barge-in: the server stops the bot 0–30 ms after detecting the interruption. The client hears silence about 690 ms after it starts talking: 200 ms of VAD `start_secs`, plus WebRTC and aiortc audio buffering. Both are reported. Lowering `start_secs` trades faster barge-in for more false interruptions (coughs, "mm-hm").

## Technology choices

### Synthetic candidate over WebRTC rather than unit-testing the pipeline
Pipecat has an eval framework (`pipecat.evals`) with scenarios and personas, but it uses a WebSocket transport. Bugs #2 and #5 only exist in the real WebRTC + Flows + turn-strategy combination, and the barge-in measurement needs a real audio round-trip. So the harness uses **aiortc** (already a Pipecat dependency) to be a browser.

### A mock LLM instead of skipping the LLM
The point of a test harness is to run anywhere: CI, a laptop with no keys. The mock streams genuine OpenAI-format tool calls, so Pipecat's function-calling path, Flows transitions, cancellation and context updates are all exercised. A fixed TTFT makes latency comparisons between runs fair.

### Subclassing the turn stop strategy rather than raising VAD `stop_secs`
Raising `stop_secs` is the one-line fix every tutorial suggests. Measured, it moved 0.6 s from the turn stage into STT, because the whole answer became a single Whisper segment. Decoupling "segment for STT" from "decide the turn is over" kept both benefits. The subclass overrides two Pipecat internals, so the Pipecat version is pinned to `<2`, and `test_session_recorder.py` plus the harness catch regressions on upgrade.

# Interview Voice Coach

**A real-time voice interviewer you can talk to (and interrupt) in the browser.** It asks role-specific technical and behavioral questions, follows up when you miss a key point, and then scores every answer against a rubric and analyses how you spoke: pace, filler words and pauses.

> Built to answer one engineering question honestly: *how fast can a fully local, CPU-only voice agent respond, and where does the time go?* Every stage is measured, and every number below comes from a script in [`eval/`](eval/).

<!-- Demo: record a 60–90 s screen capture WITH audio (a GIF can't show a voice app) and link it here. -->

## What it does

- 🎙️ **Real-time voice conversation** over WebRTC, with barge-in (talk over the interviewer and it stops)
- 🧭 **A structured interview** (intro → questions → at most one follow-up each → wrap-up), driven by a Pipecat Flows state machine
- 🎯 **RAG question selection**: paste a job posting and the questions target its requirements (pgvector + MMR for topic diversity)
- 🧪 **Post-session report**: rubric scores (correctness, depth, structure, communication), covered/missed key points, a stronger-answer outline, WPM, filler words, long pauses and response delay
- 📈 **Progress over time**: score, pace and fillers across sessions, plus per-topic averages
- ⏱️ **Per-stage latency** on every turn (turn detection, LLM, TTS, output), live in the UI and in the report

## Results

Measured on an 8-core laptop CPU with **no GPU and no cloud services** (Whisper `base.en`, Piper TTS, Smart Turn v3). The LLM is a mock with a fixed 0.4 s time-to-first-token, so the numbers isolate the pipeline itself. 103 turns over 17 full interviews, driven by a synthetic candidate over real WebRTC ([method](eval/README.md)):

![Latency by stage](docs/img/latency_by_stage.png)

| Metric | Result | Spec target |
|---|---|---|
| Voice-to-voice latency, client-measured | **p50 1.81 s** · p95 3.72 s | p50 < 1.5 s · p95 < 2.5 s |
| · turn detection (incl. final STT) / LLM / TTS / output | 1 250 / 437 / 72 / 3 ms (p50) | |
| Barge-in: interviewer audio stops | median **656 ms** at the client (server reacts in < 30 ms) | < 300 ms |
| Filler-word counting (20 synthetic clips) | precision **1.00** · recall **0.93** | reported |
| Judge vs human agreement | ⏳ needs 30 human-labelled answers ([how](eval/judge_agreement/README.md)) | Spearman ≥ 0.7 |
| Follow-up targets the missed point | ⏳ needs a real LLM (`eval/follow_up_relevance.py`) | ≥ 80 % |

**Reading these honestly:**
- On CPU the latency floor is **speech recognition**: the final transcript lands about 1.2 s after you stop talking. The p95 tail is Smart Turn waiting 3 s when it thinks you're mid-thought (11 % of turns). Cloud STT (`STT_PROVIDER=cloud`) and a shorter `TURN_MAX_WAIT_SECS` are the levers. Re-run `eval/latency_bench.py` to measure them on your hardware.
- The **first** end-to-end run measured **8.4 s**. The [Phase 5 notes](docs/phases/05-sessions-and-e2e.md) document the five bugs and design changes that brought it to 1.8 s.
- Barge-in misses its target because VAD needs 200 ms of speech to confirm an interruption, and audio already queued in the WebRTC pipeline keeps playing.

## Architecture

```mermaid
flowchart LR
    subgraph Browser [Next.js client]
      MIC[Mic] --> RTC[WebRTC]
      RTC --> SPK[Speaker]
      UI[Live transcript · timer · latency]
    end
    subgraph Server [FastAPI + Pipecat, one Python process]
      IN[SmallWebRTC in] --> REC1[user track recorder]
      REC1 --> STT[faster-whisper<br/>Silero VAD 0.2 s segments]
      STT --> TURN[Smart Turn v3<br/>+ min-silence gate]
      TURN --> FLOW[Pipecat Flows<br/>intro · question · follow-up · wrap-up]
      FLOW --> LLM[LLM, OpenAI-compatible<br/>one call per turn]
      LLM --> TTS[Piper / Kokoro / Deepgram]
      TTS --> OUT[SmallWebRTC out]
      OBS[Observers: latency · barge-in · usage · transcript]
    end
    RTC <--> IN
    OUT --> RTC
    FLOW <--> QB[(Question bank<br/>Postgres + pgvector)]
    Server --> DB[(Sessions · turns · reports)]
    DB --> AN[Post-session analyzer<br/>LLM-as-judge + offline Whisper word timestamps]
```

Key design decisions (each explained in [`docs/phases/`](docs/phases/)):

- **One LLM call per candidate turn.** The LLM grades the answer *inside* a tool call and writes the follow-up as a tool argument. The next question is spoken straight from the bank, with no second LLM round-trip.
- **A min-silence turn gate on top of Smart Turn.** VAD segments stay short (0.2 s) so Whisper transcribes sentence by sentence *while you talk*, but the turn only ends after 0.8 s of silence, because interview answers pause between sentences.
- **Latency is measured from five timestamps per turn** (speech end → turn released → LLM decision → first TTS audio → bot audible). The stages sum to the total by construction, and the client-side measurement agrees with it.
- **Speech metrics come from an offline re-transcription** (word timestamps, filler-priming prompt) of a **wall-clock-aligned** candidate-only audio track. They're kept off the latency path.

## Quick start

**Prerequisites:** [uv](https://docs.astral.sh/uv/) and Node 20+. Docker is optional. An API key for any OpenAI-compatible LLM (DeepSeek, GLM, OpenAI, Groq…) or a local Ollama.

```bash
git clone <your-repo-url> interview-voice-coach && cd interview-voice-coach
cp .env.example .env          # set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL if not DeepSeek)
```

**Server** (Python 3.12 is installed automatically by uv):
```bash
cd server
uv sync --extra embedded-db            # embedded Postgres+pgvector, no Docker needed
uv run python download_models.py       # Whisper, Piper, embeddings (~400 MB, resumable)
uv run python api.py                   # http://localhost:7860, docs at /docs
```

**Client** (in a second terminal):
```bash
cd client
npm install
npm run dev                            # http://localhost:3000
```

Open http://localhost:3000, pick a role and level, optionally paste a job posting, and start talking. Headphones help: the browser cancels echo, but speakers can still trigger false barge-ins.

<details>
<summary>Using Docker for Postgres, or the full stack</summary>

```bash
docker compose up -d db
# then in .env: DATABASE_URL=postgresql+asyncpg://coach:coach@localhost:5432/coach

docker compose --profile app up --build    # full stack, Linux only (WebRTC needs host networking)
```
</details>

<details>
<summary>Try it without any API key</summary>

A scripted OpenAI-compatible mock LLM and a synthetic candidate (a Python WebRTC client that *speaks* generated answers) run the whole pipeline end to end:
```bash
cd server
uv run python ../eval/mock_llm.py &                                     # :8001
LLM_BASE_URL=http://127.0.0.1:8001/v1 LLM_API_KEY=mock LLM_MODEL=mock uv run python api.py &
uv run python ../eval/synthetic_candidate.py --turns 8 --barge-in-every 3
```
</details>

### Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | DeepSeek | Any OpenAI-compatible endpoint |
| `JUDGE_MODEL` | = `LLM_MODEL` | Optionally a stronger model for scoring |
| `STT_PROVIDER` | `local` | `local` (faster-whisper `base.en`) or `cloud` (Deepgram nova-3) |
| `TTS_PROVIDER` | `piper` | `piper` (fastest on CPU), `kokoro` (more natural) or `deepgram` |
| `TURN_STOP_SECS` | `0.8` | Silence before your turn ends. Lower is snappier but may cut you off |
| `TURN_MAX_WAIT_SECS` | `3.0` | Max wait when Smart Turn thinks you are mid-thought (drives p95) |
| `DATABASE_URL` | *(empty)* | Empty means embedded Postgres |
| `LANGFUSE_*` | *(empty)* | Optional tracing via OpenTelemetry |

## Repository layout

```
server/
  bot.py                  Pipecat pipeline (transport, STT, turn strategy, LLM, TTS, observers)
  turn_strategy.py        Smart Turn + minimum-silence end-of-turn strategy
  flows/                  interview state machine, tools (next_question, record_answer, end_interview)
  services.py             STT/LLM/TTS factory (local vs cloud via env)
  metrics_observer.py     per-stage latency, barge-in and usage observers
  session_recorder.py     timestamped turns + wall-clock-aligned audio tracks
  api.py                  FastAPI: sessions, WebRTC signalling, reports, progress
  analysis/               judge.py (LLM-as-judge), speech_metrics.py, analyzer.py
  question_bank/          questions.yaml (132 questions with rubrics), retriever.py (pgvector + MMR)
  tests/                  44 unit tests
client/                   Next.js: start page, live interview room, report, progress charts
eval/                     latency bench, judge agreement, filler accuracy, follow-up relevance, cost
docs/phases/              build log: what was built in each phase, why, and what testing found
```

## How it was built

The project was built in phases, each with a write-up of the files, the technology choices (and the alternatives considered), and what end-to-end testing changed:

| Phase | |
|---|---|
| [0 · Scaffolding](docs/phases/00-scaffolding.md) | uv, settings, repo layout |
| [1 · Voice pipeline](docs/phases/01-voice-pipeline.md) | Pipecat, SmallWebRTC, VAD + Smart Turn, STT/TTS benchmarks on CPU |
| [2 · Latency metrics](docs/phases/02-latency-metrics.md) | Per-stage measurement (and why Pipecat's built-in breakdown was replaced) |
| [3 · Question bank + RAG](docs/phases/03-question-bank-rag.md) | pgvector, bge-small, posting chunking, MMR |
| [4 · Interview flow](docs/phases/04-interview-flow.md) | Pipecat Flows, one LLM call per turn |
| [5 · Sessions + E2E harness](docs/phases/05-sessions-and-e2e.md) | Recording, API, synthetic candidate: **8.4 s → 1.8 s** |
| [6 · Analysis](docs/phases/06-analysis.md) | LLM-as-judge, speech metrics, audio timeline fix |
| [7 · Client](docs/phases/07-client.md) | Next.js, Pipecat React SDK, accessible charts |
| [8 · Evaluation](docs/phases/08-evaluation.md) | Benchmarks and how to reproduce them |
| [9 · Release](docs/phases/09-release.md) | Docker, CI |

## Limitations

- **CPU-only latency is bounded by STT.** Whisper on a laptop CPU finishes the last sentence about 1.1 s after you stop. GPU or cloud STT (`STT_PROVIDER=cloud`) is the lever.
- **Single-user and no auth.** It's a local practice tool. Add authentication before exposing it on a network.
- **SmallWebRTC is peer-to-peer.** For public deployment behind strict NATs, use a TURN server or Daily.
- **English only**, by design (the question bank, filler lists and embedding model).

## License

MIT

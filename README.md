<div align="center">

# 🎙️ Interview Voice Coach

**A real-time voice interviewer you can talk to (and interrupt) in the browser.**

It asks role-specific technical and behavioral questions, follows up when you miss a key point,<br/>
then scores every answer against a rubric and analyses *how* you spoke: pace, filler words and pauses.

[![CI](https://github.com/snikookar/interview-voice-coach/actions/workflows/ci.yml/badge.svg)](https://github.com/snikookar/interview-voice-coach/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Pipecat](https://img.shields.io/badge/Pipecat-voice%20pipeline-6E56CF)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Postgres + pgvector](https://img.shields.io/badge/Postgres-pgvector-4169E1?logo=postgresql&logoColor=white)
![Runs on CPU](https://img.shields.io/badge/runs%20on-CPU%20only-1baf7a)
![Tests](https://img.shields.io/badge/tests-44%20passing-1baf7a)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

</div>

> Built to answer one engineering question honestly: *how fast can a fully local, CPU-only voice agent respond, and where does the time go?* Every stage is measured, and every number below comes from a script in [`eval/`](eval/).

<!-- Demo: record a 60–90 s screen capture WITH audio (a GIF can't show a voice app) and link it here. -->

**Contents:** [What it does](#-what-it-does) · [Results](#-results) · [Architecture](#%EF%B8%8F-architecture) · [Life of a turn](#-life-of-one-turn) · [Interview flow](#-the-interview-flow) · [Quick start](#-quick-start) · [How it was built](#-how-it-was-built) · [Limitations](#%EF%B8%8F-limitations)

## ✨ What it does

| | |
|---|---|
| 🎙️ **Real-time voice** | Conversation over WebRTC with **barge-in**: talk over the interviewer and it stops |
| 🧭 **Structured interview** | Intro → questions → at most one follow-up each → wrap-up, driven by a Pipecat Flows state machine |
| 🎯 **RAG question selection** | Paste a job posting and the questions target its requirements (pgvector + MMR for topic diversity) |
| 🧪 **Post-session report** | Rubric scores (correctness, depth, structure, communication), covered/missed key points, a stronger-answer outline, WPM, filler words, long pauses and response delay |
| 📈 **Progress over time** | Score, pace and fillers across sessions, plus per-topic averages |
| ⏱️ **Per-stage latency** | Turn detection, LLM, TTS and output timing on every turn, live in the UI and in the report |

## 📊 Results

Measured on an 8-core laptop CPU with **no GPU and no cloud services** (Whisper `base.en`, Piper TTS, Smart Turn v3). The LLM is a mock with a fixed 0.4 s time-to-first-token, so the numbers isolate the pipeline itself. 103 turns over 17 full interviews, driven by a synthetic candidate over real WebRTC ([method](eval/README.md)).

![Latency by stage](docs/img/latency_by_stage.png)

| Metric | Result | Spec target |
|---|---|---|
| Voice-to-voice latency, client-measured | **p50 1.81 s** · p95 3.72 s | p50 < 1.5 s · p95 < 2.5 s |
| · turn detection (incl. final STT) / LLM / TTS / output | 1 250 / 437 / 72 / 3 ms (p50) | |
| Barge-in: interviewer audio stops | median **656 ms** at the client (server reacts in < 30 ms) | < 300 ms |
| Filler-word counting (20 synthetic clips) | precision **1.00** · recall **0.93** | reported |
| Judge vs human agreement | ⏳ needs 30 human-labelled answers ([how](eval/judge_agreement/README.md)) | Spearman ≥ 0.7 |
| Follow-up targets the missed point | ⏳ needs a real LLM (`eval/follow_up_relevance.py`) | ≥ 80 % |

### Where the tail comes from

Latency is **bimodal**: most turns land around 1.8 s, and a separate cluster sits 2–3.5 s later. That cluster is Smart Turn deciding you are mid-thought and waiting for more speech before releasing the turn.

![Latency distribution](docs/img/latency_distribution.png)

### From 8.4 s to 1.8 s

Unit tests passed from Phase 4 on, but the **first real end-to-end run measured 8.4 s**, cut answers off after every sentence and once froze mid-interview. A synthetic candidate that *speaks* over WebRTC found five bugs and design problems; each bar below is one fix ([full write-up](docs/phases/05-sessions-and-e2e.md)).

![Latency journey](docs/img/latency_journey.png)

### Speech analysis accuracy

Filler words are counted from an offline Whisper re-transcription with word timestamps. Short "uh"s are the hardest case:

<p align="center"><img src="docs/img/filler_accuracy.png" alt="Filler-word detection per filler" width="560"/></p>

**Reading these honestly:**
- On CPU the latency floor is **speech recognition**: the final transcript lands about 1.2 s after you stop talking. The p95 tail is Smart Turn waiting 3 s when it thinks you're mid-thought. Cloud STT (`STT_PROVIDER=cloud`) and a shorter `TURN_MAX_WAIT_SECS` are the levers. Re-run `eval/latency_bench.py` to measure them on your hardware.
- Barge-in misses its target because VAD needs 200 ms of speech to confirm an interruption, and audio already queued in the WebRTC pipeline keeps playing.
- All charts are regenerated from `eval/results/*.json` by [`eval/plot_results.py`](eval/plot_results.py).

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Browser [Next.js client]
      MIC[🎤 Mic] --> RTC[WebRTC]
      RTC --> SPK[🔊 Speaker]
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

## ⏱️ Life of one turn

What happens between the moment you stop talking and the moment you hear the interviewer, with the five timestamps that the latency breakdown is built from:

```mermaid
sequenceDiagram
    autonumber
    participant C as 🧑 Candidate (browser)
    participant S as STT + VAD
    participant T as Turn strategy
    participant F as Flows + LLM
    participant V as TTS
    C->>S: speech (Whisper transcribes sentence by sentence)
    Note over C,S: ⏱ t0 · candidate stops speaking
    S->>T: final transcript (~1.2 s on CPU)
    T->>T: Smart Turn v3 + ≥ 0.8 s silence gate
    Note over T: ⏱ t1 · turn released
    T->>F: user turn
    F->>F: record_answer(covered, missing, follow_up)
    Note over F: ⏱ t2 · LLM decision (one call)
    F->>V: acknowledgement + next question from the bank
    Note over V: ⏱ t3 · first TTS audio
    V->>C: audio over WebRTC
    Note over C: ⏱ t4 · interviewer audible
    C-->>V: barge-in: candidate talks over the bot
    V-->>V: stop output in < 30 ms
```

`turn = t1 − t0`, `llm = t2 − t1`, `tts = t3 − t2`, `output = t4 − t3`, so the stages always add up to the total.

## 🧭 The interview flow

Each node exposes exactly the tools that make sense at that point, so the LLM can't skip ahead or ask two follow-ups:

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Intro: greeting
    Intro --> Question: next_question
    Question --> FollowUp: record_answer (key point missed)
    Question --> Question: record_answer (next question)
    FollowUp --> Question: record_answer
    Question --> WrapUp: out of questions or time
    FollowUp --> WrapUp: out of questions or time
    Intro --> WrapUp: end_interview
    Question --> WrapUp: end_interview
    WrapUp --> [*]
    WrapUp --> Analysis: session saved
    state Analysis {
        direction LR
        Judge: LLM-as-judge rubric scores
        Speech: Offline Whisper → WPM, fillers, pauses
    }
```

### The question bank

132 questions with rubrics (key points + follow-ups) for **AI engineer** and **backend** roles at junior, mid and senior level. When you paste a job posting it is chunked, embedded with `bge-small` and matched against the bank in pgvector; MMR keeps the selection from clustering on one topic.

```mermaid
pie showData title 132 questions by topic
    "LLMs" : 16
    "RAG" : 13
    "Python" : 12
    "System design" : 12
    "Behavioral" : 12
    "ML basics" : 11
    "Agents" : 11
    "Evaluation" : 11
    "Databases" : 10
    "MLOps" : 8
    "API design" : 8
    "Distributed systems" : 8
```

## 🧰 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Voice pipeline | [Pipecat](https://github.com/pipecat-ai/pipecat) + SmallWebRTC | Frame-based pipeline with observers for measurement; peer-to-peer WebRTC with no media server |
| Turn detection | Silero VAD + Smart Turn v3 + custom silence gate | Semantic end-of-turn, tuned so pauses between sentences don't cut you off |
| STT | faster-whisper `base.en` (local) · Deepgram nova-3 (cloud) | Swappable through one env var |
| TTS | Piper (local default) · Kokoro · Deepgram | Piper RTF 0.07 vs Kokoro 0.59 on CPU |
| LLM | Any OpenAI-compatible API (DeepSeek, OpenAI, Groq, GLM, Ollama…) | One tool call per turn |
| Interview logic | Pipecat Flows | Explicit state machine instead of one giant prompt |
| Storage + RAG | Postgres + pgvector (embedded or Docker), fastembed `bge-small` | One database for sessions, reports and vectors |
| Analysis | LLM-as-judge + offline Whisper word timestamps | Off the latency path |
| Client | Next.js, Pipecat React SDK, Recharts | Live room, report and progress charts |
| Ops | uv, Docker Compose, GitHub Actions, optional Langfuse tracing | |

## 🚀 Quick start

**Prerequisites:** [uv](https://docs.astral.sh/uv/) and Node 20+. Docker is optional. An API key for any OpenAI-compatible LLM (DeepSeek, GLM, OpenAI, Groq…) or a local Ollama.

```bash
git clone https://github.com/snikookar/interview-voice-coach.git && cd interview-voice-coach
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

### Reproduce the numbers

```bash
cd server
uv sync --extra eval
uv run python ../eval/latency_bench.py --help         # latency + barge-in over real WebRTC
uv run python ../eval/plot_results.py                 # regenerate every chart in docs/img/
uv run pytest -q                                      # 44 unit tests
```

## 📁 Repository layout

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

## 🛠️ How it was built

The project was built in ten phases, each with a write-up of the files, the technology choices (and the alternatives considered), and what end-to-end testing changed:

```mermaid
timeline
    title Build log
    Foundations : 0 · Scaffolding : 1 · Voice pipeline : 2 · Latency metrics
    Intelligence : 3 · Question bank + RAG : 4 · Interview flow
    Reality check : 5 · Sessions + E2E harness (8.4 s → 1.8 s) : 6 · Post-session analysis
    Product : 7 · Next.js client : 8 · Evaluation suite : 9 · Docker + CI
```

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

## ⚠️ Limitations

- **CPU-only latency is bounded by STT.** Whisper on a laptop CPU finishes the last sentence about 1.1 s after you stop. GPU or cloud STT (`STT_PROVIDER=cloud`) is the lever.
- **Single-user and no auth.** It's a local practice tool. Add authentication before exposing it on a network.
- **SmallWebRTC is peer-to-peer.** For public deployment behind strict NATs, use a TURN server or Daily.
- **English only**, by design (the question bank, filler lists and embedding model).

## 📄 License

[MIT](LICENSE)

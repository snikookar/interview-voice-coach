# Phase 1: Real-time voice pipeline

**Goal:** talk to an LLM interviewer in the browser and interrupt it mid-sentence (barge-in).

## Files

| File | Purpose |
|---|---|
| `server/services.py` | Factory that builds the **STT, LLM and TTS** services from settings, so local and cloud are a `.env` switch |
| `server/bot.py` | The Pipecat pipeline: `transport.input → STT → user aggregator → LLM → TTS → transport.output → assistant aggregator` |
| `server/api.py` | FastAPI app. `POST /api/offer` does the WebRTC SDP exchange, `PATCH /api/offer` receives trickle-ICE candidates, and it starts one bot per connection |
| `server/download_models.py` | Resumable, size-checked model downloads (Kokoro, Whisper, embeddings). Written after a real 325 MB Kokoro download dropped halfway and Pipecat's own downloader left a truncated file that would never load |

## How a turn flows

```
mic ─WebRTC/Opus─► SmallWebRTC input ─► Silero VAD (speech yes/no, 200 ms)
                                        └► Smart Turn v3 (is the *answer* finished?)
                   ► faster-whisper (segment → text)
                   ► user aggregator (adds the text to LLMContext)
                   ► LLM (streams tokens)
                   ► TTS (synthesises sentence by sentence while the LLM is still writing)
                   ► SmallWebRTC output ─WebRTC─► speaker
```
**Barge-in:** the default *user turn start* strategy is VAD-based. When you start speaking while the bot talks, an `InterruptionFrame` flushes the LLM, TTS and output queues, and the bot stops mid-word.

## Technology choices (and the alternatives)

### Voice framework: **Pipecat**
| Option | Why not / why |
|---|---|
| Hand-rolled asyncio + WebSockets | You'd re-implement VAD gating, interruption, sentence aggregation and audio resampling. That's weeks of work |
| LiveKit Agents | Excellent, but it needs a LiveKit server (SFU) even locally |
| OpenAI Realtime / speech-to-speech | Lowest latency, but it's one opaque box. You **can't measure per-stage latency or swap STT/TTS**, and both are core goals of this project |
| **Pipecat** ✅ | Open source, frame-based pipeline. Every stage is a swappable processor. It ships Smart Turn, Flows (Phase 4) and a per-stage latency observer (Phase 2) |

Note: Pipecat 1.x replaced `PipelineTask`/`PipelineRunner` with `PipelineWorker`/`WorkerRunner`, and moved Flows into `pipecat.flows`. The code uses the 1.12 API.

### Transport: **SmallWebRTC** (local) rather than WebSockets or Daily
- **WebSocket audio** is TCP. One lost packet stalls everything behind it (head-of-line blocking), and you'd have to implement echo cancellation and jitter buffering yourself.
- **WebRTC** uses UDP + Opus, and the browser gives you **echo cancellation for free**. That's essential for barge-in, because without AEC the bot hears itself through your speakers and interrupts itself.
- **SmallWebRTC** is a peer-to-peer connection straight into the Python process (aiortc), with no account and no media server. **Daily** is the drop-in swap for public deployment behind strict NATs (see Risks in the spec).

### VAD + turn detection: **Silero VAD + Smart Turn v3**
A plain silence timeout (say 800 ms) cuts off people who pause to think. That's exactly what candidates do in interviews. Pipecat 1.x splits the job in two:
1. **Silero VAD** (`stop_secs=0.2`) only answers "is there speech right now?". It's fast and runs on CPU.
2. **Smart Turn v3**, a small ONNX audio model, looks at the *prosody* of the last segment and answers "did they finish the thought?". A trailing "and then we... um" is classified as *incomplete*, so the bot waits.

### STT: **faster-whisper** (local) / **Deepgram nova-3** (cloud)
- faster-whisper is CTranslate2 with int8 quantisation, roughly 4× faster than openai-whisper on CPU. **The default model came from a measurement, not a guess.** On the dev machine (8-core CPU, no GPU, warm model, 4.4 s clip):

  | Model | Transcribe time | Kept "um/uh"? |
  |---|---|---|
  | tiny.en | 0.57 s | yes |
  | **base.en** (default) | 1.10 s | yes |
  | distil-whisper-small.en | 3.27 s | yes |

  Distil models are fast on GPU but slow on CPU, because they keep the large encoder. `base.en` is the CPU sweet spot. Even so, CPU STT alone eats most of the 1.5 s budget, which is exactly why cloud mode exists (spec, Risks table).
- The decoder is primed with an `initial_prompt` full of "um, uh, like" because Whisper otherwise silently drops fillers (spec §6).
- Deepgram is streaming STT, so transcripts are ready almost immediately after you stop. It's the cloud comparison point.

### LLM: **any OpenAI-compatible endpoint** (DeepSeek by default)
I used `OpenAILLMService` with a configurable `base_url` instead of a vendor-specific class. The same code then runs DeepSeek, GLM, OpenAI, Groq or a local Ollama, and the latency benchmark can compare them. `max_tokens=220` plus "2–3 sentences" in the prompt keeps turns short, which helps both latency and naturalness.

### TTS: **Kokoro** (default) / **Piper** / **Deepgram Aura**
| | Kokoro-82M | Piper | Deepgram Aura |
|---|---|---|---|
| Runs | local ONNX | local ONNX | cloud |
| Voice quality | near-human | robotic but clear | human |
| CPU speed | real-time ×3–5 | real-time ×10+ | network-bound |

Kokoro is the default because voice quality matters in a demo video. Piper is the fallback for slow CPUs. Deepgram reuses the STT key, so cloud mode needs **one** API key.

### Backend: **FastAPI**
SmallWebRTC peer connections live **inside the Python process**, so the signalling endpoint must run in the same event loop as the bot. FastAPI is async-native and is what Pipecat's own runner uses. Its Pydantic models also generate the OpenAPI docs at `/docs` for free.

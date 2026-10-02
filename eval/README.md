# Evaluation

Everything the README claims is produced by a script in this folder. Results land in `eval/results/`, and `plot_results.py` draws the charts in `docs/img/`.

All commands run from the repo root. Add `--extra eval` for scipy, scikit-learn and matplotlib.

| Spec metric | Script | Needs | Status |
|---|---|---|---|
| Voice-to-voice latency, p50/p95 per stage, 100 turns | `latency_bench.py` | running server | ✅ local CPU measured · ⏳ cloud mode (needs Deepgram key) |
| Barge-in: agent stops in < 300 ms | `latency_bench.py` (every 3rd turn) | running server | ✅ measured |
| Filler count accuracy | `filler_clips/filler_accuracy.py` | labelled clips | ✅ synthetic baseline · ⏳ 20 clips of your voice |
| Judge agreement (Spearman ≥ 0.7), key-point F1 ≥ 0.8 | `judge_agreement/judge_agreement.py` | 30 human-labelled answers + LLM key | ⏳ needs your labels |
| Follow-up relevance ≥ 80 % | `follow_up_relevance.py` | LLM key | ⏳ needs a real LLM |
| Cost per 20-min session | `cost.py` + `prices.yaml` | finished sessions | ⏳ needs real-LLM sessions for token counts |

## The tools

- **`mock_llm.py`**: a scripted OpenAI-compatible server (streams real tool calls, answers judge requests), so the full pipeline runs without a key. It has a fixed time-to-first-token (`--ttft-ms`), so latency comparisons isolate everything *except* the LLM provider.
- **`synthetic_candidate.py`**: a Python WebRTC client (aiortc) that takes an interview exactly like the browser does. It speaks Kokoro-synthesised answers and measures voice-to-voice latency and barge-in from the client side.

## Reproduce

```bash
# 1. latency, local mode (any LLM: mock or real)
cd server && uv run python api.py                         # in its own terminal
uv run --project server python eval/latency_bench.py --label local-cpu --turns 100

# 2. latency, cloud mode: restart the server with STT_PROVIDER=cloud TTS_PROVIDER=deepgram
uv run --project server python eval/latency_bench.py --label cloud --turns 100

# 3. fillers
uv run --project server --extra eval python eval/filler_clips/make_synthetic_clips.py
uv run --project server --extra eval python eval/filler_clips/filler_accuracy.py --labels eval/filler_clips/synthetic_labels.csv

# 4. follow-ups (real LLM from .env)
uv run --project server --extra eval python eval/follow_up_relevance.py -n 50

# 5. judge agreement: see judge_agreement/README.md for the labelling workflow
# 6. cost
uv run --project server --extra eval python eval/cost.py

uv run --project server --extra eval python eval/plot_results.py
```

## Measured so far (dev machine: 8-core laptop CPU, no GPU)

**Latency, local mode** (Whisper `base.en`, Piper, Smart Turn v3, mock LLM at 0.4 s TTFT) over 103 turns in 17 interviews:

| | p50 | p95 |
|---|---|---|
| Client voice-to-voice | **1 810 ms** | 3 723 ms |
| Server total | 1 772 ms | 3 715 ms |
| · turn (silence gate + final STT) | 1 250 ms | 3 203 ms |
| · LLM (mock) | 437 ms | 452 ms |
| · TTS first audio | 72 ms | 113 ms |
| · output | 3 ms | 5 ms |

- The **turn** stage is bounded by CPU Whisper finishing the last sentence (final transcript p50 **1 235 ms** after the voice stops). It already overlaps the 0.8 s silence gate, so STT, not the gate, is the floor.
- The **p95 tail** is 11 of 103 turns at exactly ~3.2 s: Smart Turn judged the answer *unfinished* and waited its 3 s fallback (`TURN_MAX_WAIT_SECS`). STT had finished on time in every one of them.
- With a real LLM, add (its TTFT − 0.4 s) to the LLM stage.

**Barge-in** (17 interruptions): the server stops sending audio within 0–30 ms of detecting the interruption, but the client hears silence after a **median 656 ms** (max 703 ms). That's 200 ms of VAD `start_secs` before an interruption is recognised, plus audio already buffered in the WebRTC pipeline. The spec's < 300 ms target is **not met** end to end.

**Filler counting, synthetic baseline** (20 Kokoro clips, 27 fillers, Whisper `small.en`): precision **1.00**, recall **0.93**. Both misses are one clip where "uh" was transcribed as the article "a". "metrics like MRR" was correctly *not* counted. The priming prompt made no difference on synthetic speech (TTS pronounces fillers crisply), so the real test is recorded human clips.

# Phase 8: Evaluation

**Goal:** every claim in the README is backed by a reproducible script, and the numbers are reported honestly, including where targets are missed.

## Files

| File | Purpose |
|---|---|
| `eval/latency_bench.py` | N synthetic interviews over real WebRTC. Client voice-to-voice plus server per-stage p50/p95, plus barge-in |
| `eval/judge_agreement/export_answers.py` | Exports finished answers for **blind** human labelling (judge scores withheld) |
| `eval/judge_agreement/judge_agreement.py` | Spearman ρ and quadratic-weighted κ per dimension, key-point precision/recall/F1 |
| `eval/judge_agreement/labels.example.jsonl` | 5 rows with **made-up** labels, only to demo the pipeline |
| `eval/filler_clips/make_synthetic_clips.py` | 20 Kokoro clips (4 voices) with known filler counts, including negative cases |
| `eval/filler_clips/filler_accuracy.py` | Per-filler precision/recall through the real offline path, `--no-prompt` ablation |
| `eval/follow_up_relevance.py` | Replays the live interviewer prompt and tool schema on answers missing one key point. Detected / asked / targeted rates |
| `eval/cost.py` + `eval/prices.yaml` | Cost from **measured** usage (tokens, STT seconds, TTS characters) × editable unit prices, projected to 20 min |
| `eval/plot_results.py` | `docs/img/latency_by_stage.png`, `docs/img/judge_vs_human.png` |
| `server/metrics_observer.py` (`UsageRecorder`), `analysis/judge.py` (usage) | Usage capture that `cost.py` relies on |

## Results (local CPU, see `eval/README.md` for the full tables)

- **Voice-to-voice p50 1.81 s / p95 3.72 s** (client-measured, 103 turns, mock LLM at 0.4 s). Turn stage 1.25 s (CPU Whisper bound), LLM 0.44 s, TTS 0.07 s.
- **The p95 tail is diagnosed, not guessed.** 11 % of turns sat at exactly 0.2 + 3.0 s: Smart Turn's "unfinished" fallback. That's now configurable (`TURN_MAX_WAIT_SECS`).
- **Barge-in: 656 ms median at the client.** It **misses** the spec's 300 ms. The server reacts in 0–30 ms, but VAD needs 200 ms of speech to confirm, and buffered WebRTC audio keeps playing.
- **Filler counting (synthetic): precision 1.00, recall 0.93.** The priming-prompt ablation showed no difference on TTS speech, which is itself a finding: it must be re-measured on human speech.

### Not yet measured, and why
| Metric | Blocker | One command once unblocked |
|---|---|---|
| Judge agreement | Needs 30 answers labelled by a human; a judge "validated" against labels I wrote would be circular | `judge_agreement.py` |
| Follow-up relevance | Needs a real LLM key; the mock always asks the same follow-up | `follow_up_relevance.py -n 50` |
| Cloud latency | Needs a Deepgram key | `latency_bench.py --label cloud` |
| Cost | Needs sessions with real token counts | `cost.py` |

All four scripts were smoke-tested end to end against the mock LLM. Their mock outputs are deliberately not committed.

## Design choices

### Synthetic candidate instead of "100 turns of me talking"
The spec's 100-turn benchmark would take a human about an hour per configuration, and humans aren't repeatable. The synthetic candidate gives the **same audio** to every configuration, so local and cloud differ only in what's being measured. Its limitation (TTS speech is cleaner than human speech, especially for Smart Turn and filler detection) is stated wherever it matters.

### Mock LLM with a fixed TTFT for the latency benchmark
LLM time-to-first-token varies by provider, region and minute. Fixing it at 0.4 s isolates the part of the pipeline this project controls. With a real provider, the LLM stage is simply that provider's TTFT for a short tool call.

### Agreement metrics: Spearman *and* quadratic κ
Spearman only checks ordering, so a judge that is consistently one point harsher scores ρ = 1. Quadratic-weighted κ penalises that offset, which matters because users read absolute scores. Key-point F1 is reported separately because "you missed X" is the most actionable feedback, and it can be wrong even when overall scores agree.

### Blind labelling
`export_answers.py` never writes the judge's scores into the labelling file, so the human isn't anchored by them.

### Cost from measured usage, priced separately
Token counts, STT seconds and TTS characters are recorded per session by observers. Unit prices live in a YAML file with an `as_of` field, because prices go stale and measured usage doesn't.

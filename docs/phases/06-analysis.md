# Phase 6: Post-session analysis (LLM-as-judge + speech metrics)

**Goal:** within a minute or two of hanging up, produce a report with a score per answer, the missed key points, and delivery metrics (pace, fillers, pauses), plus progress across sessions.

## Files

| File | Purpose |
|---|---|
| `server/analysis/judge.py` | `AnswerScore` (the spec's schema). Anchored 1–5 rubric, one few-shot example, JSON mode with a fallback, one repair retry, and **re-alignment of key points onto exact rubric strings** |
| `server/analysis/speech_metrics.py` | Offline word-level transcription of `user.wav`, then pure functions: WPM, fillers per minute, long pauses (> 3 s), response delay, talk ratio |
| `server/analysis/analyzer.py` | Orchestrates both (judge calls in parallel, transcription in a worker thread), plus the latency summary (p50/p95 per stage), and writes the report |
| `server/api.py` (updated) | Analysis starts automatically after each call. `POST /api/sessions/{id}/analyze` re-runs it. `GET /api/progress` returns scores, WPM and fillers over time, plus per-topic averages |
| `server/session_recorder.py` (updated) | `TrackRecorder`: wall-clock-aligned audio tracks (see "What the end-to-end run caught") |
| `eval/mock_llm.py` (updated) | Answers judge requests heuristically, so the whole pipeline including reports runs without a key |
| `tests/test_judge.py`, `tests/test_speech_metrics.py` | A fake OpenAI client covering retry, JSON-mode fallback, alignment and clamping. Filler rules, pause exclusion, WPM, response delay |

## The report (stored as JSONB on the session)
```jsonc
{
  "overall": {"score": 3.4, "coverage": 0.62, "dimensions": {"correctness": 3.8, "depth": 3.0, ...}},
  "answers": [{ "question", "topic", "answer_text", "follow_up_question", "follow_up_answer",
                "score": { covered_points, missed_points, correctness, depth, structure, communication, better_answer_outline },
                "coverage": 0.75, "speech": { wpm, fillers_per_min, long_pauses, ... } }],
  "speech":  { wpm, fillers_per_min, filler_counts, long_pauses, longest_pause_secs,
               avg_response_delay_secs, talk_ratio, source },
  "latency": { p50_ms, p95_ms, stages_p50_ms },
  "barge_in": { count, median_ms },
  "live_notes": { ... what the interviewer noticed live ... }
}
```

## Technology choices

### Why the judge re-runs after the call instead of reusing the live `record_answer` notes
The live LLM call is optimised for **speed** (it's on the latency path, one short tool call). The judge is optimised for **accuracy**: full answer plus follow-up, anchored scales, a few-shot example, temperature 0, and optionally a stronger `JUDGE_MODEL`. The live notes are kept in the report for comparison.

### Key-point alignment in code, not trust in the judge
LLMs paraphrase ("dropout or weight decay" for "Regularisation such as L1, L2, dropout or weight decay"). `align_points` fuzzy-matches claims back onto the exact rubric strings, discards invented points, and *derives* `missed = rubric − covered`. That makes coverage consistent and lets key-point detection be scored with precision, recall and F1 against human labels (Phase 8).

### JSON mode with a graceful fallback, not a structured-output library
`response_format={"type":"json_object"}` works on DeepSeek, OpenAI, GLM and Ollama. Libraries like `instructor` add value for complex nested schemas, but here a Pydantic `model_validate` plus one "your output was invalid, reply with JSON only" retry covers it with no extra dependency. If an endpoint rejects JSON mode, the judge switches to plain text and extracts the `{...}`.

### Offline re-transcription for speech metrics
| | Live STT | Offline analysis |
|---|---|---|
| Model | `base.en` (fast) | `small.en` (more accurate) |
| Word timestamps | no | **yes** |
| Filler priming | yes | yes |
| On the latency path | yes | **no**, after the call, in a thread, one at a time |

Whisper drops "um" and "uh" unless primed. Both passes use an `initial_prompt` full of disfluencies. "like" only counts when set off by commas ("it was, like, slow"), not in comparisons ("metrics like MRR"). "kind of" and "sort of" are handled the same way.

### What the end-to-end run caught
1. **Timeline drift.** Pipecat's `AudioBufferProcessor` keeps user and bot tracks aligned *with each other* by padding. A 140 s call produced a **164 s** `user.wav`, so word timestamps drifted out of their turns and **40% of words were lost** (116 of 197). The new `TrackRecorder` places every frame at its wall-clock offset from connect. Result: 206 of 206 words, and WPM matches the synthetic voice's rate (~155).
2. **Duplicated bot turns.** `BotStartedSpeakingFrame` is broadcast as two frames (up and down), and both counted as a turn start. Only the downstream copy is used now.
3. **Filler accuracy on the synthetic run:** um 4/4, basically 1/1, you know 1/1, uh **1/2**. Phase 8 measures this properly on labelled clips.

### Background task, not Celery or RQ
A single-user app with one analysis every ~10 minutes doesn't need a broker. `asyncio.create_task` with a strong reference set, a semaphore so only one Whisper job saturates the CPU, and a status field (`analyzing → done | failed`) that the UI polls. For multi-user deployment, the same `analyze_session(id)` function drops into a queue worker unchanged.

# Phase 4: Interview flow (state machine with Pipecat Flows)

**Goal:** a real interview structure (intro → questions → at most one follow-up each → wrap-up), not a free-form chat that drifts.

## Files

| File | Purpose |
|---|---|
| `server/flows/state.py` | `InterviewState`: plan, current index, phase, follow-up flag, time budget, live notes |
| `server/flows/interview_flow.py` | Node factories: `intro`, `question_N`, `follow_up_N`, `wrap_up`, plus the interviewer persona |
| `server/flows/tools.py` | LLM tools: `next_question`, `record_answer`, `end_interview` (global) |
| `server/tests/test_interview_flow.py` | State-machine tests with no LLM: follow-up rules, time limit, early exit, tool schemas |
| `server/bot.py`, `server/api.py` (updated) | `FlowManager` drives the pipeline. The offer request carries role and level, and a plan is built per connection |

```
 intro ──next_question──► question ──record_answer──► follow_up ──record_answer──► question ...
                              │   (key point missing, first time)                     │
                              └───────────record_answer (complete)────────────────────┤
                                                                                      ▼
                                       end_interview (any node) ───────────────►  wrap_up ─► end call
```

## Key design decision: **one LLM call per candidate turn**

A naive Flows design costs **two** LLM round-trips per turn:
1. LLM reads the answer and calls `record_answer`, which triggers a node transition.
2. The new node runs the LLM again to *say* the next question.

On top of STT and turn detection, the second call alone (~0.5–1 s time-to-first-token) breaks the 1.5 s budget. So:
- `record_answer` takes **`follow_up_question`** and **`acknowledgement`** as tool *arguments*. The LLM grades the answer *and* writes what to say, all in one call.
- The next node has `respond_immediately=False` and a **`speak` pre-action** that says `"<ack> <lead-in> <question from the bank>"`. The text is pushed into the pipeline like streamed LLM output, so the LLM context records what was asked.
- Why a custom `speak` action instead of Flows' built-in `tts_say`? The end-to-end test found two problems with `tts_say`. (1) Flows **blocks the node transition until the speech has been played**, and if the candidate interrupts, the "finished" event never fires: the node never got its new tools and the interview **froze**. (2) The whole text goes to the TTS as one unit, so on CPU nothing plays until all of it is synthesised. Pushing `LLMFullResponseStart + LLMTextFrame + LLMFullResponseEnd` instead lets Pipecat's sentence aggregator pipeline synthesis with playback, and custom actions are never awaited.
- The next question is already in memory (the plan is built before the call). This is the spec's *"prefetch the next question while the user is still talking"*, taken to the limit: the lookup costs zero.

Bonus: questions are spoken **verbatim from the bank**, so the rubric always matches what was actually asked.

The greeting and the goodbye are also `speak` actions, so the first audio doesn't wait on the LLM at all. The goodbye is followed by `end_conversation`, whose `EndFrame` queues behind the farewell, so the call ends after it has played.

## Behaviour rules (in prompts and code)
- **At most one follow-up per question.** This is enforced in code (`state.follow_up_asked`), not just requested in the prompt. LLMs don't reliably count.
- **Time budget.** When `elapsed > max_minutes − 90 s`, no new questions or follow-ups are started, and the flow goes to wrap-up.
- **Stale tool calls are ignored.** A candidate can pause, let the LLM start, then keep talking. Tools use `cancel_on_interruption=True`, and a call that arrives in the wrong phase returns `NO_RESPONSE`. Without this guard, a second `record_answer` once fired after the goodbye.
- "I don't know" means no follow-up, a kind acknowledgement, and moving on.
- "Let me think..." gets "Take your time." Smart Turn usually prevents the bot from jumping in at all.
- The rubric is in the node's developer message, marked *secret*. The persona forbids revealing it or judging answers live (judging happens after the session, in Phase 6).

## Technology choices

### Pipecat Flows rather than one big prompt or LangGraph
| Option | Problem |
|---|---|
| One system prompt with all the rules | The LLM forgets which question it's on, asks two follow-ups, invents questions. Hard to test |
| LangGraph | A great agent graph framework, but it isn't frame-aware. You'd have to bridge its state into the real-time pipeline yourself |
| **Pipecat Flows** ✅ | Built for voice pipelines. Each node swaps the prompt and tools in place, and supports pre/post actions (`end_conversation`, custom ones like our `speak`). In Pipecat 1.x it ships inside `pipecat.flows` |

### Direct functions rather than hand-written JSON schemas
Flows reads the function's **type hints and Google-style docstring** to build the tool schema. The code *is* the schema, so they can't drift apart. A test asserts every parameter has a description, because the LLM relies on those.

### Role and level chosen in the UI, not asked by voice
The spec has `intro` asking for role and level. I moved that to the start form because (a) the question plan must exist *before* the call for the zero-latency prefetch above, and (b) parsing "mid-to-senior back-end-ish" from speech is error-prone. The intro now asks for a short self-introduction instead, which is what real interviews do.

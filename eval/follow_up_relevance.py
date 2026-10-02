"""Does the interviewer's follow-up target the key point the candidate missed?

    uv run --project server --extra eval python eval/follow_up_relevance.py -n 50

For N questions from the bank we build an answer that covers every key point but
ONE (the "omitted" point), then send exactly what the live interviewer sees (the
same role prompt, question-node task with rubric, and record_answer tool schema
that Pipecat Flows sends) to the configured LLM. We check:

* detected: the omitted point is in the tool call's missing_points (re-aligned to
  the rubric the same way the judge does)
* asked:    a follow-up question was produced
* targeted: embedding similarity puts the follow-up closest to the omitted point
  among all of the question's key points (spec target: >= 80%)

Answers are built from the rubric text itself (deterministic, free). That makes
detection easier than with real speech, so treat the result as an upper bound and
review eval/results/follow_up_relevance.json by hand.
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from openai import AsyncOpenAI  # noqa: E402
from pipecat.flows.types import FlowsDirectFunctionWrapper  # noqa: E402

from analysis.judge import align_points  # noqa: E402
from config import get_settings  # noqa: E402
from flows import interview_flow, tools  # noqa: E402
from flows.state import InterviewState  # noqa: E402
from question_bank.bank import load_bank  # noqa: E402
from question_bank.retriever import embed_passages  # noqa: E402

CONNECTORS = ["First,", "Also,", "On top of that,", "Another thing is that", "And finally,"]


def build_answer(points: list[str]) -> str:
    sentences = [
        f"{CONNECTORS[min(i, len(CONNECTORS) - 1)]} {p[0].lower() + p[1:]}."
        for i, p in enumerate(points)
    ]
    return "So, I'd approach it like this. " + " ".join(sentences)


def record_answer_tool() -> dict:
    schema = FlowsDirectFunctionWrapper(function=tools.record_answer).to_function_schema()
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": {
                "type": "object",
                "properties": schema.properties,
                "required": schema.required,
            },
        },
    }


def interviewer_messages(q: dict, answer: str) -> list[dict]:
    state = InterviewState(session_id="eval", role=q["role"][0], level=q["level"][-1], plan=[q])
    state.advance()
    node = interview_flow.create_question_node(state, q, "", is_first=True, is_last=False)
    return [
        {"role": "system", "content": interview_flow._role_message(state)},
        *node["task_messages"],
        {"role": "assistant", "content": f"Let's get started. First question. {q['question']}"},
        {"role": "user", "content": answer},
    ]


async def run_case(client: AsyncOpenAI, model: str, q: dict, omitted: int) -> dict:
    points = q["key_points"]
    answer = build_answer([p for i, p in enumerate(points) if i != omitted])
    resp = await client.chat.completions.create(
        model=model,
        messages=interviewer_messages(q, answer),
        tools=[record_answer_tool()],
        temperature=0.6,  # same as the live interviewer
    )
    msg = resp.choices[0].message
    call = next((c for c in msg.tool_calls or [] if c.function.name == "record_answer"), None)
    args = json.loads(call.function.arguments) if call else {}
    missing = align_points(args.get("missing_points", []), points)
    follow_up = (args.get("follow_up_question") or "").strip()
    return {
        "question_id": q["id"],
        "omitted": points[omitted],
        "omitted_index": omitted,
        "answer": answer,
        "tool_called": call is not None,
        "missing_points": missing,
        "follow_up": follow_up,
        "detected": points[omitted] in missing,
        "asked": bool(follow_up),
        "text_reply": msg.content,
    }


def add_targeting(results: list[dict], questions: dict[str, dict]) -> None:
    """targeted = the follow-up is semantically closest to the omitted key point."""
    for r in results:
        r["targeted"] = False
        if not r["follow_up"]:
            continue
        points = questions[r["question_id"]]["key_points"]
        vecs = embed_passages([r["follow_up"], *points])
        sims = vecs[1:] @ vecs[0]
        r["similarities"] = [round(float(s), 3) for s in sims]
        r["targeted"] = int(np.argmax(sims)) == r["omitted_index"]


async def main_async(n: int, seed: int, model: str | None) -> dict:
    s = get_settings()
    model = model or s.llm_model
    client = AsyncOpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    rng = random.Random(seed)
    bank = [q.model_dump() for q in load_bank() if q.type == "technical" and len(q.key_points) >= 3]
    cases = [(q, rng.randrange(len(q["key_points"]))) for q in rng.sample(bank, min(n, len(bank)))]

    sem = asyncio.Semaphore(4)

    async def guarded(q, i):
        async with sem:
            return await run_case(client, model, q, i)

    results = await asyncio.gather(*(guarded(q, i) for q, i in cases))
    add_targeting(results, {q["id"]: q for q, _ in cases})

    def rate(key: str) -> float:
        return round(sum(r[key] for r in results) / len(results), 3)

    return {
        "model": model,
        "n": len(results),
        "tool_called": rate("tool_called"),
        "detected": rate("detected"),
        "asked": rate("asked"),
        "targeted": rate("targeted"),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model")
    args = parser.parse_args()
    result = asyncio.run(main_async(args.n, args.seed, args.model))
    out = ROOT / "eval" / "results" / "follow_up_relevance.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n{result['model']} on {result['n']} incomplete answers")
    for k in ("tool_called", "detected", "asked", "targeted"):
        print(f"  {k:<12} {result[k]:.0%}")
    print("  (spec target: targeted >= 80%)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

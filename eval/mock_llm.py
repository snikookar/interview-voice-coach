"""A scripted OpenAI-compatible LLM server for end-to-end tests without an API key.

It speaks just enough of the Chat Completions streaming protocol for Pipecat:
it reads which interview tools the current Flows node offers and answers with the
tool call a sensible interviewer would make. A configurable delay before the
first chunk emulates a real model's time-to-first-token.

    uv run --project server python eval/mock_llm.py --port 8001 --ttft-ms 400
    # then: LLM_BASE_URL=http://localhost:8001/v1 LLM_API_KEY=mock LLM_MODEL=mock
"""

import argparse
import asyncio
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
TTFT_SECS = 0.4


def _last_developer_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") in ("developer", "system") and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _rubric_points(prompt: str) -> list[str]:
    lines = prompt.split("Rubric key points (secret):")[-1].split("Suggested follow-ups:")[0]
    return [ln.strip("- ").strip() for ln in lines.strip().splitlines() if ln.strip()]


def decide(messages: list[dict], tools: list[dict]) -> tuple[str | None, dict | None, str]:
    """Return (tool_name, tool_args, text) for the next assistant message."""
    names = {t["function"]["name"] for t in tools or []}
    last = messages[-1] if messages else {}
    if last.get("role") != "user":
        return None, None, "Take your time."

    if "next_question" in names:
        return "next_question", {"acknowledgement": "Nice to meet you."}, ""

    if "record_answer" in names:
        prompt = _last_developer_message(messages)
        points = _rubric_points(prompt)
        if "You asked a follow-up" in prompt or not points:
            return (
                "record_answer",
                {
                    "covered_points": points[:2],
                    "missing_points": points[2:],
                    "follow_up_question": "",
                    "acknowledgement": "Thanks.",
                },
                "",
            )
        return (
            "record_answer",
            {
                "covered_points": points[:1],
                "missing_points": points[1:],
                "follow_up_question": "Could you say a bit more about how you would measure that?",
                "acknowledgement": "Okay.",
            },
            "",
        )
    return None, None, "Could you tell me a little more?"


def _chunk(cid: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    tool, args, text = decide(body.get("messages", []), body.get("tools", []))
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def stream():
        await asyncio.sleep(TTFT_SECS)
        yield _chunk(cid, {"role": "assistant"})
        if tool:
            call = {
                "index": 0,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }
            yield _chunk(cid, {"tool_calls": [call]})
            yield _chunk(cid, {}, "tool_calls")
        else:
            for word in text.split(" "):
                yield _chunk(cid, {"content": word + " "})
            yield _chunk(cid, {}, "stop")
        usage = {"prompt_tokens": 500, "completion_tokens": 30, "total_tokens": 530}
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> None:
    global TTFT_SECS
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--ttft-ms", type=int, default=400)
    args = parser.parse_args()
    TTFT_SECS = args.ttft_ms / 1000
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

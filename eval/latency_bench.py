"""Voice-to-voice latency benchmark: N synthetic interviews against a running server.

Each run is a full interview through the real WebRTC path (see synthetic_candidate.py).
Per turn we keep both views:
* client: candidate audio ended -> first audible interviewer audio (what a user feels)
* server: the per-stage breakdown recorded by LatencyRecorder (turn / llm / tts / output)

    # start the server in the mode you want to measure, then:
    uv run --project server python eval/latency_bench.py --label local-cpu --turns 100
    uv run --project server python eval/latency_bench.py --label cloud --turns 100

Results go to eval/results/latency_<label>.json; plot_results.py turns them into charts.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from synthetic_candidate import run_interview

RESULTS = Path(__file__).parent / "results"


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def summarise(client_ms: list[float], server_turns: list[dict], barge_in_ms: list[float]) -> dict:
    stages = sorted({k for t in server_turns for k in t["stages_ms"]})
    return {
        "turns": len(server_turns),
        "client_voice_to_voice": {
            "n": len(client_ms),
            "p50_ms": pct(client_ms, 50),
            "p95_ms": pct(client_ms, 95),
        },
        "server_total": {
            "p50_ms": pct([t["total_ms"] for t in server_turns], 50),
            "p95_ms": pct([t["total_ms"] for t in server_turns], 95),
        },
        "server_stages": {
            s: {
                "p50_ms": pct([t["stages_ms"][s] for t in server_turns], 50),
                "p95_ms": pct([t["stages_ms"][s] for t in server_turns], 95),
            }
            for s in stages
        },
        "stt_done_after_speech_p50_ms": pct(
            [t["stt_ms"] for t in server_turns if t.get("stt_ms") is not None], 50
        ),
        "barge_in_client_ms": {
            "n": len(barge_in_ms),
            "p50_ms": pct(barge_in_ms, 50),
            "max_ms": max(barge_in_ms, default=None),
            "under_300ms": sum(1 for b in barge_in_ms if b < 300),
        },
    }


async def bench(api: str, label: str, target_turns: int, barge_in_every: int) -> dict:
    client_ms: list[float] = []
    barge_in_ms: list[float] = []
    server_turns: list[dict] = []
    sessions: list[str] = []
    http = httpx.AsyncClient(base_url=api, timeout=30)
    health = (await http.get("/api/health")).json()
    started = time.time()

    while len(server_turns) < target_turns:
        run = await run_interview(api, turns=8, barge_in_every=barge_in_every, num_questions=3)
        sessions.append(run.session_id)
        for t in run.turns:
            if t.kind == "answer" and t.voice_to_voice_ms is not None:
                client_ms.append(t.voice_to_voice_ms)
            if t.kind == "barge_in" and t.barge_in_stop_ms is not None:
                barge_in_ms.append(t.barge_in_stop_ms)
        await asyncio.sleep(3)  # let the server save the session
        detail = (await http.get(f"/api/sessions/{run.session_id}")).json()
        server_turns.extend(detail.get("latency") or [])
        print(f"[{label}] sessions={len(sessions)} turns={len(server_turns)}/{target_turns}")

    await http.aclose()
    return {
        "label": label,
        "config": health,
        "duration_min": round((time.time() - started) / 60, 1),
        "sessions": sessions,
        "summary": summarise(client_ms, server_turns, barge_in_ms),
        "raw": {"client_ms": client_ms, "barge_in_ms": barge_in_ms, "server_turns": server_turns},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:7860")
    parser.add_argument("--label", required=True, help="e.g. local-cpu, cloud")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--barge-in-every", type=int, default=3)
    args = parser.parse_args()

    result = asyncio.run(bench(args.api, args.label, args.turns, args.barge_in_every))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"latency_{args.label}.json"
    out.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(json.dumps(s, indent=2))
    print(
        f"\nclient voice-to-voice p50 {s['client_voice_to_voice']['p50_ms']} ms, "
        f"p95 {s['client_voice_to_voice']['p95_ms']} ms over {s['client_voice_to_voice']['n']} turns"
    )
    stages = ", ".join(f"{k} {v['p50_ms']}" for k, v in s["server_stages"].items())
    print(f"stages p50: {stages}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

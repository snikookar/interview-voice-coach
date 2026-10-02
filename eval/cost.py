"""Cost per interview from MEASURED usage, projected to a 20-minute session.

    uv run --project server --extra eval python eval/cost.py

Every finished session stores what Pipecat reported (LLM tokens, STT audio seconds,
TTS characters) plus the judge's token usage. This script prices that usage with
eval/prices.yaml, normalises to cost per minute, and projects a 20-minute session
for two configurations:

* local: faster-whisper + Piper on your machine; you pay only for LLM tokens
* cloud: the same LLM plus Deepgram STT and Aura TTS
"""

import argparse
import json
import statistics
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).parent


def session_cost(report: dict, duration_min: float, prices: dict) -> dict:
    u = report.get("usage", {}) or {}
    j = report.get("judge_usage", {}) or {}
    llm = prices["llm"]
    cached = u.get("cached_prompt_tokens", 0)
    live_llm = (
        (u.get("prompt_tokens", 0) - cached) * llm["input_per_m"]
        + cached * llm["cached_input_per_m"]
        + u.get("completion_tokens", 0) * llm["output_per_m"]
    ) / 1e6
    judge = (
        j.get("prompt_tokens", 0) * llm["input_per_m"]
        + j.get("completion_tokens", 0) * llm["output_per_m"]
    ) / 1e6
    stt = u.get("stt_audio_secs", 0) / 60 * prices["stt_cloud"]["per_minute"]
    tts = u.get("tts_characters", 0) / 1000 * prices["tts_cloud"]["per_1k_chars"]
    return {
        "duration_min": duration_min,
        "live_llm": live_llm,
        "judge": judge,
        "stt_cloud": stt,
        "tts_cloud": tts,
        "local_total": live_llm + judge,
        "cloud_total": live_llm + judge + stt + tts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:7860")
    parser.add_argument("--minutes", type=float, default=20)
    args = parser.parse_args()
    prices = yaml.safe_load((HERE / "prices.yaml").read_text())

    sessions = httpx.get(f"{args.api}/api/sessions", timeout=30).json()
    costs = []
    for s in sessions:
        if s["status"] != "done" or not s["duration_secs"]:
            continue
        detail = httpx.get(f"{args.api}/api/sessions/{s['id']}", timeout=30).json()
        report = detail.get("report") or {}
        if not report.get("usage"):
            continue  # recorded before usage tracking existed
        costs.append(session_cost(report, s["duration_secs"] / 60, prices))
    if not costs:
        raise SystemExit("No finished sessions with usage data yet.")

    def per_min(key: str) -> float:
        return statistics.median(c[key] / c["duration_min"] for c in costs)

    projection = {
        k: round(per_min(k) * args.minutes, 4)
        for k in ("live_llm", "judge", "stt_cloud", "tts_cloud", "local_total", "cloud_total")
    }
    result = {
        "sessions": len(costs),
        "prices_as_of": prices.get("as_of"),
        f"projected_{int(args.minutes)}_min_usd": projection,
    }
    out = HERE / "results" / "cost.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("NOTE: sessions run against eval/mock_llm.py report the mock's fixed token counts.")


if __name__ == "__main__":
    main()

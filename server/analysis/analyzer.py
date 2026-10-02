"""Post-session analysis: judge every answer, compute speech metrics, write the report.

Runs as a background task when a call ends (and on demand via the API), so the
CPU-heavy offline transcription never competes with a live call's latency path.
"""

import asyncio
import statistics
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from analysis.judge import AnswerScore, Judge
from analysis.speech_metrics import (
    TurnWindow,
    Word,
    compute_metrics,
    talk_ratio,
    transcribe_words,
    words_from_text,
)
from config import get_settings
from db import SessionRow, TurnRow, session_scope

# One analysis at a time: offline transcription saturates the CPU, and a burst
# (e.g. sessions re-queued at startup) should drain in order, not all at once.
_analysis_lock = asyncio.Semaphore(1)
DIMENSIONS = ("correctness", "depth", "structure", "communication")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def summarise_latency(latency: list[dict]) -> dict:
    turns = [t for t in latency if t.get("total_ms") is not None]
    totals = [t["total_ms"] for t in turns]
    stages = sorted({k for t in turns for k in t.get("stages_ms", {})})
    return {
        "turns": len(turns),
        "p50_ms": percentile(totals, 50),
        "p95_ms": percentile(totals, 95),
        "stages_p50_ms": {
            s: percentile([t["stages_ms"][s] for t in turns if s in t["stages_ms"]], 50)
            for s in stages
        },
    }


def user_windows(turns: list[TurnRow]) -> list[TurnWindow]:
    windows, last_bot_end = [], None
    for t in turns:
        if t.speaker == "bot":
            last_bot_end = t.end_ms
        else:
            windows.append(TurnWindow(t.start_ms, t.end_ms, t.question_id, last_bot_end))
            last_bot_end = None  # only the first reply after a bot turn has a "delay"
    return windows


def collect_answers(plan: list[dict], turns: list[TurnRow]) -> list[dict]:
    """Group transcript turns by question: answer text and the follow-up asked."""
    answers = []
    for q in plan:
        user = [t for t in turns if t.speaker == "user" and t.question_id == q["id"]]
        if not user:
            continue  # never reached
        follow = [
            t.text
            for t in turns
            if t.speaker == "bot" and t.question_id == q["id"] and t.phase == "follow_up"
        ]
        answers.append(
            {
                "question": q,
                "answer_text": " ".join(t.text for t in user if t.phase == "question"),
                "follow_up_question": " ".join(follow) or None,
                "follow_up_answer": " ".join(t.text for t in user if t.phase == "follow_up"),
            }
        )
    return answers


async def load_words(row: SessionRow, turns: list[TurnRow]) -> tuple[list[Word], str]:
    audio = Path(row.audio_dir) / "user.wav" if row.audio_dir else None
    if audio and audio.exists():
        try:
            model = get_settings().analysis_whisper_model
            words = await asyncio.to_thread(transcribe_words, audio, model)
            return words, f"offline whisper ({model}, word timestamps)"
        except Exception as e:
            logger.warning(f"Offline transcription failed, using live transcript: {e}")
    words = [
        w
        for t in turns
        if t.speaker == "user"
        for w in words_from_text(t.text, t.start_ms, t.end_ms)
    ]
    return words, "live transcript (approximate timing)"


async def analyze_session(session_id: str, judge: Judge | None = None) -> dict | None:
    async with _analysis_lock:
        return await _analyze(session_id, judge)


async def _analyze(session_id: str, judge: Judge | None) -> dict | None:
    async with session_scope() as db:
        row = await db.scalar(
            select(SessionRow)
            .where(SessionRow.id == session_id)
            .options(selectinload(SessionRow.turns))
        )
        if row is None:
            return None
        row.status = "analyzing"
        await db.commit()
        turns = list(row.turns)

    try:
        report = await _build_report(row, turns, judge or Judge())
        status, error = "done", None
    except Exception as e:
        logger.exception(f"Analysis failed for {session_id}")
        report, status, error = None, "failed", str(e)

    async with session_scope() as db:
        row = await db.get(SessionRow, session_id)
        if row is not None:
            if report is not None:
                row.report = {**(row.report or {}), **report}
                row.overall_score = report["overall"]["score"]
            row.status, row.error = status, error
            await db.commit()
    logger.info(f"Analysis {status} for {session_id}")
    return report


async def _build_report(row: SessionRow, turns: list[TurnRow], judge: Judge) -> dict:
    answers = collect_answers(row.plan or [], turns)

    sem = asyncio.Semaphore(4)

    async def score(a: dict) -> AnswerScore:
        async with sem:
            text = a["answer_text"]
            if a["follow_up_answer"]:
                text = f"{text}\n{a['follow_up_answer']}"
            return await judge.score(a["question"], text, a["follow_up_question"])

    scores, (words, speech_source) = await asyncio.gather(
        asyncio.gather(*(score(a) for a in answers)), load_words(row, turns)
    )

    windows = user_windows(turns)
    bot_spans = [(t.start_ms, t.end_ms) for t in turns if t.speaker == "bot"]
    overall_speech = compute_metrics(words, windows).as_dict()
    overall_speech["talk_ratio"] = talk_ratio(windows, bot_spans)
    overall_speech["source"] = speech_source

    answer_reports = []
    for a, s in zip(answers, scores, strict=True):
        q = a["question"]
        q_windows = [w for w in windows if w.question_id == q["id"]]
        answer_reports.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "topic": q["topic"],
                "type": q.get("type", "technical"),
                "key_points": q["key_points"],
                "answer_text": a["answer_text"],
                "follow_up_question": a["follow_up_question"],
                "follow_up_answer": a["follow_up_answer"],
                "score": s.model_dump(),
                "mean_score": round(s.mean, 2),
                "coverage": round(len(s.covered_points) / max(1, len(q["key_points"])), 2),
                "speech": compute_metrics(words, q_windows).as_dict(),
            }
        )

    if scores:
        overall = {
            "score": round(statistics.mean(s.mean for s in scores), 2),
            "coverage": round(statistics.mean(a["coverage"] for a in answer_reports), 2),
            "dimensions": {
                d: round(statistics.mean(getattr(s, d) for s in scores), 2) for d in DIMENSIONS
            },
        }
    else:
        overall = {"score": None, "coverage": None, "dimensions": {}}

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "judge_model": judge.model,
        "judge_usage": judge.usage,
        "overall": overall,
        "answers": answer_reports,
        "speech": overall_speech,
        "latency": summarise_latency(row.latency or []),
        "barge_in": {
            "count": len(row.barge_in_ms or []),
            "median_ms": percentile(row.barge_in_ms or [], 50),
        },
    }

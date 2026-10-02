"""FastAPI app: WebRTC signalling for the voice bot, sessions, reports and progress."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from bot import run_bot
from config import get_settings
from db import SessionRow, close_db, init_db, session_scope
from flows.state import InterviewState
from question_bank.bank import LEVELS, ROLES
from question_bank.retriever import build_plan, sync_bank

settings = get_settings()

webrtc_handler = SmallWebRTCRequestHandler(
    ice_servers=[IceServer(urls=url) for url in settings.ice_server_urls] or None
)

# Hook run after a call ends (the post-session analyzer plugs in here).
on_session_end = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    updated = await sync_bank()
    logger.info(f"Question bank ready ({updated} re-embedded)")
    yield
    await webrtc_handler.close()
    await close_db()


app = FastAPI(title="Interview Voice Coach", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── schemas ─────────────────────────


class CreateSessionRequest(BaseModel):
    role: Literal["ai-engineer", "backend"] = "ai-engineer"
    level: Literal["junior", "mid", "senior"] = "mid"
    mode: Literal["technical", "behavioral"] = "technical"
    job_posting: str | None = Field(default=None, max_length=20_000)
    num_questions: int = Field(default=settings.default_num_questions, ge=1, le=10)
    max_minutes: int = Field(default=settings.default_max_minutes, ge=3, le=45)


class CreateSessionResponse(BaseModel):
    id: str
    num_questions: int
    topics: list[str]


class SessionSummary(BaseModel):
    id: str
    created_at: datetime
    role: str
    level: str
    mode: str
    status: str
    duration_secs: float | None
    overall_score: float | None


# ───────────────────────── sessions ─────────────────────────


async def _recent_question_ids(role: str, limit: int = 3) -> set[str]:
    """Questions from the last few sessions, so practice sessions don't repeat."""
    async with session_scope() as db:
        rows = await db.scalars(
            select(SessionRow)
            .where(SessionRow.role == role, SessionRow.status != "created")
            .order_by(SessionRow.created_at.desc())
            .limit(limit)
        )
        return {q["id"] for row in rows for q in (row.plan or [])}


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    """Build the question plan up front, so the call never waits on retrieval."""
    plan = await build_plan(
        req.role,
        req.level,
        job_posting=req.job_posting,
        num_questions=req.num_questions,
        mode=req.mode,
        exclude_ids=await _recent_question_ids(req.role),
    )
    if not plan:
        raise HTTPException(422, "No questions match this role and level")
    row = SessionRow(
        id=str(uuid.uuid4()),
        role=req.role,
        level=req.level,
        mode=req.mode,
        job_posting=req.job_posting,
        max_minutes=req.max_minutes,
        plan=plan,
    )
    async with session_scope() as db:
        db.add(row)
        await db.commit()
    # Topics only: showing the questions in advance would spoil the practice.
    return CreateSessionResponse(
        id=row.id, num_questions=len(plan), topics=[q["topic"] for q in plan]
    )


@app.get("/api/sessions", response_model=list[SessionSummary])
async def list_sessions():
    async with session_scope() as db:
        rows = await db.scalars(select(SessionRow).order_by(SessionRow.created_at.desc()))
        return [SessionSummary.model_validate(r, from_attributes=True) for r in rows]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    async with session_scope() as db:
        row = await db.scalar(
            select(SessionRow)
            .where(SessionRow.id == session_id)
            .options(selectinload(SessionRow.turns))
        )
        if row is None:
            raise HTTPException(404, "Session not found")
        return {
            **SessionSummary.model_validate(row, from_attributes=True).model_dump(),
            "started_at": row.started_at,
            "ended_at": row.ended_at,
            "job_posting": row.job_posting,
            # Questions are revealed once the interview is over.
            "plan": row.plan if row.status not in ("created", "live") else [],
            "latency": row.latency,
            "barge_in_ms": row.barge_in_ms,
            "report": row.report,
            "error": row.error,
            "has_audio": bool(row.audio_dir),
            "turns": [
                {
                    "idx": t.idx,
                    "speaker": t.speaker,
                    "text": t.text,
                    "phase": t.phase,
                    "question_id": t.question_id,
                    "start_ms": t.start_ms,
                    "end_ms": t.end_ms,
                    "interrupted": t.interrupted,
                }
                for t in row.turns
            ],
        }


@app.get("/api/sessions/{session_id}/audio")
async def get_session_audio(session_id: str):
    path = settings.sessions_dir / session_id / "conversation.wav"
    if not path.exists():
        raise HTTPException(404, "No recording for this session")
    return FileResponse(path, media_type="audio/wav", filename=f"interview-{session_id[:8]}.wav")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    async with session_scope() as db:
        await db.execute(delete(SessionRow).where(SessionRow.id == session_id))
        await db.commit()
    return {"status": "deleted"}


# ───────────────────────── WebRTC ─────────────────────────


@app.post("/api/offer")
async def offer(background_tasks: BackgroundTasks, body: dict = Body(...)):
    """SDP offer/answer exchange. The bot starts once the peer connection exists.

    The browser sends ``requestData: {session_id}`` (from ``POST /api/sessions``).
    ``from_dict`` accepts both ``requestData`` and ``request_data``. A renegotiation
    of an existing peer connection carries ``pc_id`` and starts no new bot.
    """
    request = SmallWebRTCRequest.from_dict(body)
    state: InterviewState | None = None
    if not request.pc_id:
        session_id = (request.request_data or {}).get("session_id")
        async with session_scope() as db:
            row = await db.get(SessionRow, session_id) if session_id else None
        if row is None:
            raise HTTPException(404, "Unknown session_id; create one with POST /api/sessions")
        if row.status != "created":
            raise HTTPException(409, f"Session already {row.status}")
        state = InterviewState(
            session_id=row.id,
            role=row.role,
            level=row.level,
            mode=row.mode,
            plan=row.plan,
            max_minutes=row.max_minutes,
        )

    async def on_connection(connection: SmallWebRTCConnection):
        if state is not None:
            background_tasks.add_task(run_bot, connection, state, on_session_end)

    return await webrtc_handler.handle_web_request(
        request=request, webrtc_connection_callback=on_connection
    )


@app.patch("/api/offer")
async def ice_candidate(body: dict = Body(...)):
    """Trickle ICE: candidates the browser discovers after the initial offer."""
    request = SmallWebRTCPatchRequest(
        pc_id=body["pc_id"],
        candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
    )
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


# ───────────────────────── misc ─────────────────────────


@app.get("/api/meta")
async def meta():
    return {
        "roles": ROLES,
        "levels": LEVELS,
        "modes": ["technical", "behavioral"],
        "defaults": {
            "num_questions": settings.default_num_questions,
            "max_minutes": settings.default_max_minutes,
        },
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "stt": settings.stt_provider,
        "tts": settings.tts_provider,
        "llm": settings.llm_model,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=7860)

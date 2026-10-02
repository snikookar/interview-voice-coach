"""PostgreSQL + pgvector storage: question bank, sessions, turns and reports.

If DATABASE_URL is empty, an embedded Postgres (pgserver, ships with pgvector) is
started under ``data/pg`` so the app runs without Docker.
"""

import subprocess
import time
import uuid
from datetime import UTC, datetime

from loguru import logger
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import get_settings

JsonType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class QuestionRow(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    roles: Mapped[list[str]] = mapped_column(ARRAY(String))
    levels: Mapped[list[str]] = mapped_column(ARRAY(String))
    topic: Mapped[str] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(16))  # technical | behavioral
    difficulty: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16), default="bank")  # bank | generated
    data: Mapped[dict] = mapped_column(JsonType)  # full question incl. key_points, follow_ups
    content_hash: Mapped[str] = mapped_column(String(64))
    # No fixed dimension: switching EMBEDDING_MODEL just re-indexes. With ~120 rows an
    # exact scan is faster than maintaining an HNSW index.
    embedding = mapped_column(Vector())


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    role: Mapped[str] = mapped_column(String(32))
    level: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(16), default="technical")
    job_posting: Mapped[str | None] = mapped_column(Text)
    max_minutes: Mapped[int] = mapped_column(Integer, default=12)
    plan: Mapped[list] = mapped_column(JsonType, default=list)
    status: Mapped[str] = mapped_column(String(16), default="created")
    duration_secs: Mapped[float | None] = mapped_column(Float)
    latency: Mapped[list] = mapped_column(JsonType, default=list)
    barge_in_ms: Mapped[list] = mapped_column(JsonType, default=list)
    audio_dir: Mapped[str | None] = mapped_column(String(512))
    report: Mapped[dict | None] = mapped_column(JsonType)
    overall_score: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)

    turns: Mapped[list["TurnRow"]] = relationship(
        back_populates="session", order_by="TurnRow.idx", cascade="all, delete-orphan"
    )


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(8))  # user | bot
    text: Mapped[str] = mapped_column(Text)
    phase: Mapped[str] = mapped_column(String(16))  # intro | question | follow_up | wrap_up
    question_id: Mapped[str | None] = mapped_column(String(64))
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    interrupted: Mapped[bool] = mapped_column(Boolean, default=False)

    session: Mapped[SessionRow] = relationship(back_populates="turns")


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_embedded_server = None


def _resolve_database_url() -> str:
    global _embedded_server
    url = get_settings().database_url
    if url:
        return url
    try:
        import pgserver
    except ImportError as e:
        raise RuntimeError(
            "DATABASE_URL is empty and the embedded DB is not installed. Either run "
            "`docker compose up db` and set DATABASE_URL, or `uv sync --extra embedded-db`."
        ) from e
    pg_dir = get_settings().data_dir / "pg"
    pg_dir.mkdir(parents=True, exist_ok=True)
    # After an unclean shutdown Postgres runs crash recovery, which can take longer
    # than pgserver's 10 s start timeout on Windows; the server keeps starting in
    # the background, so retrying attaches to it once it is ready.
    for attempt in range(1, 7):
        try:
            _embedded_server = pgserver.get_server(str(pg_dir))
            break
        except subprocess.TimeoutExpired:
            logger.warning(f"Embedded Postgres still starting (attempt {attempt}), retrying...")
            time.sleep(5)
    else:
        raise RuntimeError(f"Embedded Postgres did not start; see {pg_dir / 'log'}")
    uri = _embedded_server.get_uri()
    logger.info(f"Using embedded Postgres at {uri}")
    return uri.replace("postgresql://", "postgresql+asyncpg://", 1)


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_async_engine(_resolve_database_url(), pool_pre_ping=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def session_scope() -> AsyncSession:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker()


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None

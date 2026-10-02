"""Records a live interview: timestamped turns, separate audio tracks, and persistence.

Timestamps are milliseconds since the call connected, which is also when audio
recording starts, so transcript times line up with the WAV files for the
post-session speech analysis.
"""

import asyncio
import time
import wave
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from db import SessionRow, TurnRow, session_scope, utcnow
from flows.state import InterviewState

# Bot speech separated by a gap shorter than this (and no user turn in between)
# is one turn: TTS on CPU can leave small gaps between sentences.
BOT_MERGE_GAP_MS = 1500


@dataclass
class RecordedTurn:
    speaker: str  # user | bot
    start_ms: int
    end_ms: int
    phase: str
    question_id: str | None
    text: str = ""
    interrupted: bool = False


class SessionRecorder(BaseObserver):
    def __init__(self, state: InterviewState, session_dir: Path):
        super().__init__()
        self.state = state
        self.session_dir = session_dir
        self.turns: list[RecordedTurn] = []
        self._t0: float | None = None
        self._user_turn: RecordedTurn | None = None
        self._bot_turn: RecordedTurn | None = None
        self._last_bot_turn: RecordedTurn | None = None
        self._pending_bot_text: list[str] = []
        self.audio = AudioBufferProcessor(num_channels=1)
        self.audio.add_event_handler("on_track_audio_data", self._on_track_audio)
        self.audio_saved = asyncio.Event()

    # ───────────── timeline ─────────────

    async def start(self) -> None:
        self._t0 = time.monotonic()
        await self.audio.start_recording()

    def _now_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000) if self._t0 else 0

    def _open(self, speaker: str) -> RecordedTurn:
        turn = RecordedTurn(
            speaker=speaker,
            start_ms=self._now_ms(),
            end_ms=self._now_ms(),
            phase=self.state.phase,
            question_id=self.state.current_id,
        )
        self.turns.append(turn)
        return turn

    async def on_push_frame(self, data: FramePushed) -> None:
        if not data.first_push or self._t0 is None:
            return
        frame = data.frame
        now = self._now_ms()

        if isinstance(frame, UserStartedSpeakingFrame):
            # Speech segments with no bot reply in between are one answer.
            last = self.turns[-1] if self.turns else None
            self._user_turn = last if last and last.speaker == "user" else self._open("user")
            self._last_bot_turn = None  # a user turn breaks bot-turn merging
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._user_turn:
            self._user_turn.end_ms = now
        elif isinstance(frame, TranscriptionFrame):
            # Transcripts can land shortly after the turn ends; they belong to the
            # latest user turn until the next one starts.
            target = self._user_turn or self._open("user")
            target.text = f"{target.text} {frame.text.strip()}".strip()
        elif isinstance(frame, BotStartedSpeakingFrame):
            last = self._last_bot_turn
            if last and now - last.end_ms < BOT_MERGE_GAP_MS:
                self._bot_turn = last
            else:
                self._bot_turn = self._open("bot")
            self._flush_bot_text()
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._bot_turn:
            self._bot_turn.end_ms = now
            self._flush_bot_text()
            self._last_bot_turn, self._bot_turn = self._bot_turn, None
        elif isinstance(frame, TTSTextFrame):
            self._pending_bot_text.append(frame.text)
            if self._bot_turn:
                self._flush_bot_text()
        elif isinstance(frame, InterruptionFrame):
            self._pending_bot_text.clear()  # synthesized but never spoken
            if self._bot_turn:
                # The candidate cut in: this bot turn is over, whatever is still in flight.
                self._bot_turn.interrupted = True
                self._bot_turn.end_ms = now
                self._last_bot_turn, self._bot_turn = None, None

    def _flush_bot_text(self) -> None:
        if self._bot_turn and self._pending_bot_text:
            joined = "".join(
                t if t.startswith((" ", ",", ".", "?", "!")) else f" {t}"
                for t in self._pending_bot_text
            )
            self._bot_turn.text = f"{self._bot_turn.text}{joined}".strip()
            self._pending_bot_text.clear()

    # ───────────── audio ─────────────

    async def _on_track_audio(
        self, _buffer, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
    ):
        try:
            await asyncio.to_thread(self._write_wavs, user_audio, bot_audio, sample_rate)
            logger.info(f"Saved audio tracks to {self.session_dir}")
        except Exception as e:  # never let recording break the call
            logger.error(f"Failed to save audio: {e}")
        finally:
            self.audio_saved.set()

    def _write_wavs(self, user_audio: bytes, bot_audio: bytes, sample_rate: int) -> None:
        import numpy as np

        self.session_dir.mkdir(parents=True, exist_ok=True)
        user = np.frombuffer(user_audio, dtype=np.int16)
        bot = np.frombuffer(bot_audio, dtype=np.int16)
        n = max(len(user), len(bot))
        mix = np.zeros(n, dtype=np.int32)
        mix[: len(user)] += user
        mix[: len(bot)] += bot
        tracks = {
            "user.wav": user,
            "bot.wav": bot,
            "conversation.wav": np.clip(mix, -32768, 32767).astype(np.int16),
        }
        for name, samples in tracks.items():
            with wave.open(str(self.session_dir / name), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())

    # ───────────── persistence ─────────────

    def finished_turns(self) -> list[RecordedTurn]:
        """Drop empty user turns (noise that never produced a transcript)."""
        return [t for t in self.turns if t.text.strip()]

    async def save(self, latency: list[dict], barge_in_ms: list[float]) -> None:
        # Give the audio writer a moment: it runs while the pipeline is cancelled.
        try:
            await asyncio.wait_for(self.audio_saved.wait(), timeout=10)
        except TimeoutError:
            logger.warning("Audio tracks were not written before save")

        turns = self.finished_turns()
        async with session_scope() as db:
            row = await db.get(SessionRow, self.state.session_id)
            if row is None:
                logger.error(f"Session {self.state.session_id} vanished before save")
                return
            row.ended_at = utcnow()
            row.duration_secs = round(self.state.elapsed_secs, 1)
            row.latency = latency
            row.barge_in_ms = barge_in_ms
            row.audio_dir = (
                str(self.session_dir) if (self.session_dir / "user.wav").exists() else None
            )
            row.status = "ended" if turns else "empty"
            row.report = {
                "live_notes": {
                    qid: [asdict(n) for n in notes] for qid, notes in self.state.notes.items()
                },
                "end_reason": self.state.end_reason or "disconnected",
                "questions_asked": self.state.index + 1,
            }
            for i, t in enumerate(turns):
                db.add(
                    TurnRow(
                        session_id=row.id,
                        idx=i,
                        speaker=t.speaker,
                        text=t.text,
                        phase=t.phase,
                        question_id=t.question_id,
                        start_ms=t.start_ms,
                        end_ms=max(t.end_ms, t.start_ms),
                        interrupted=t.interrupted,
                    )
                )
            await db.commit()
        logger.info(f"Saved session {self.state.session_id}: {len(turns)} turns")


def mark_started(row: SessionRow) -> None:
    row.started_at = datetime.now(UTC)
    row.status = "live"

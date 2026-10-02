"""Records a live interview: timestamped turns, separate audio tracks, and persistence.

Timestamps are milliseconds since the call connected, and every audio frame is
written at its wall-clock position since that same moment, so transcript times
line up with ``user.wav`` for the post-session speech analysis.

(Pipecat's AudioBufferProcessor keeps the user and bot tracks aligned with *each
other* by padding, which stretched a 140 s call into a 164 s user track: word
timestamps then drifted out of their turns. Hence the wall-clock TrackRecorder.)
"""

import asyncio
import time
import wave
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from db import SessionRow, TurnRow, session_scope, utcnow
from flows.state import InterviewState

# Bot speech separated by a gap shorter than this (and no user turn in between)
# is one turn: TTS on CPU can leave small gaps between sentences.
BOT_MERGE_GAP_MS = 1500


class TrackRecorder(FrameProcessor):
    """Pass-through processor that records one audio stream on a wall-clock timeline.

    Each frame is placed at (now - t0) minus its own duration; gaps (e.g. no bot
    audio while the candidate talks) are filled with silence. Frames arriving in a
    burst are appended back to back, so the track never runs ahead of real time.
    """

    def __init__(self, frame_type: type, name: str):
        super().__init__(name=name)
        self._frame_type = frame_type
        self._t0: float | None = None
        self.sample_rate: int | None = None
        self._chunks: list[np.ndarray] = []
        self._length = 0

    def start(self, t0: float) -> None:
        self._t0 = t0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            self._t0 is not None
            and direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, self._frame_type)
        ):
            self.add(frame.audio, frame.sample_rate, frame.num_channels, time.monotonic())
        await self.push_frame(frame, direction)

    def add(self, audio: bytes, sample_rate: int, num_channels: int, now: float) -> None:
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        elif sample_rate != self.sample_rate:
            return  # rates are fixed per call; skip rather than corrupt the track
        samples = np.frombuffer(audio, dtype=np.int16)
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels).mean(axis=1).astype(np.int16)
        starts_at = int((now - self._t0) * sample_rate) - len(samples)
        if starts_at > self._length + sample_rate // 50:  # > 20 ms behind: insert silence
            self._chunks.append(np.zeros(starts_at - self._length, dtype=np.int16))
            self._length = starts_at
        self._chunks.append(samples)
        self._length += len(samples)

    def samples(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(x) == 0:
        return x
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)


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
        # user_track goes right after transport.input(), bot_track right after
        # transport.output() (which re-emits frames as they are played).
        self.user_track = TrackRecorder(InputAudioRawFrame, "UserTrackRecorder")
        self.bot_track = TrackRecorder(OutputAudioRawFrame, "BotTrackRecorder")

    # ───────────── timeline ─────────────

    async def start(self) -> None:
        self._t0 = time.monotonic()
        self.user_track.start(self._t0)
        self.bot_track.start(self._t0)

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
        # Pipecat broadcasts speaking frames as two frames (upstream and downstream);
        # counting both opened every bot turn twice.
        if not data.first_push or self._t0 is None:
            return
        if data.direction != FrameDirection.DOWNSTREAM:
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

    def write_audio(self) -> bool:
        """Write user.wav, bot.wav and a mixed conversation.wav. Returns True on success."""
        user, bot = self.user_track.samples(), self.bot_track.samples()
        if len(user) == 0:
            return False
        self.session_dir.mkdir(parents=True, exist_ok=True)
        user_sr = self.user_track.sample_rate or 16000
        bot_sr = self.bot_track.sample_rate or user_sr
        bot_at_user_rate = _resample(bot, bot_sr, user_sr)
        n = max(len(user), len(bot_at_user_rate))
        mix = np.zeros(n, dtype=np.int32)
        mix[: len(user)] += user
        mix[: len(bot_at_user_rate)] += bot_at_user_rate
        tracks = {
            "user.wav": (user, user_sr),
            "bot.wav": (bot, bot_sr),
            "conversation.wav": (np.clip(mix, -32768, 32767).astype(np.int16), user_sr),
        }
        for name, (samples, sr) in tracks.items():
            with wave.open(str(self.session_dir / name), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())
        return True

    # ───────────── persistence ─────────────

    def finished_turns(self) -> list[RecordedTurn]:
        """Drop empty user turns (noise that never produced a transcript)."""
        return [t for t in self.turns if t.text.strip()]

    async def save(self, latency: list[dict], barge_in_ms: list[float]) -> None:
        try:
            has_audio = await asyncio.to_thread(self.write_audio)
        except Exception as e:  # a failed recording must not lose the transcript
            logger.error(f"Failed to save audio: {e}")
            has_audio = False

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
            row.audio_dir = str(self.session_dir) if has_audio else None
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

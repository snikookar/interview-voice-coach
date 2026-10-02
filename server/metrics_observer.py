"""Per-stage latency and barge-in measurement.

Two observers watch frames flowing through the pipeline without changing them.

``LatencyRecorder`` timestamps five moments of every candidate -> interviewer turn:

    speech_end ──► turn_end ──► llm_done ──► first_audio ──► bot_speaking
        │  turn detection  │  LLM   │   TTS     │  output   │
        │  (STT runs in    │        │           │           │
        │   parallel; the  │        │           │           │
        │   transcript gate│        │           │           │
        │   is reported)   │        │           │           │

* speech_end: the candidate's voice last stopped (VAD stop minus its stop_secs)
* turn_end: the turn is released to the LLM (Smart Turn + min silence + transcript)
* llm_done: the LLM's decision arrives (a tool call, or the first text token)
* first_audio: the TTS produces the first audio of the reply
* bot_speaking: the output transport starts playing it

Each stage is the gap between two consecutive moments, so the stages always sum
to the total. (An earlier version reused Pipecat's UserBotLatencyObserver spans;
for answers with pauses they were anchored to the first pause and attributed
whole seconds to the wrong stage, so they're no longer used.)

``BargeInObserver`` measures how long the bot keeps talking after an interruption.
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, STTUsageMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

STAGES = ("turn", "llm", "tts", "output")
# Anything longer is not one response (e.g. a stalled turn); keep it out of the stats.
MAX_VALID_TURN_MS = 15_000


@dataclass
class TurnLatency:
    turn_index: int
    total_ms: float
    stages_ms: dict[str, float]
    # How long after speech end the final transcript arrived. STT overlaps the
    # turn-detection wait, so it is reported but not added to the total.
    stt_ms: float | None = None


LatencyCallback = Callable[[TurnLatency], Awaitable[None]]


class LatencyRecorder(BaseObserver):
    """Collects one ``TurnLatency`` per interviewer response."""

    def __init__(
        self, on_turn: LatencyCallback | None = None, clock: Callable[[], float] = time.time
    ):
        super().__init__()
        self._clock = clock
        self._on_turn = on_turn
        self.turns: list[TurnLatency] = []
        self._reset()

    def _reset(self) -> None:
        self._speech_end: float | None = None
        self._transcript_at: float | None = None
        self._turn_end: float | None = None
        self._llm_done: float | None = None
        self._first_audio: float | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        # Pipecat broadcasts some frames in both directions; the downstream copy is
        # the one that travels mic -> speaker.
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        frame, now = data.frame, self._clock()

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._reset()  # the candidate (still) talking: start over
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._reset()
            self._speech_end = frame.timestamp - frame.stop_secs
        elif self._speech_end is None:
            return
        elif isinstance(frame, TranscriptionFrame) and self._turn_end is None:
            self._transcript_at = now
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._turn_end is None:
            self._turn_end = now
        elif self._turn_end is None:
            return
        elif isinstance(frame, (FunctionCallInProgressFrame, LLMTextFrame)):
            self._llm_done = self._llm_done or now
        elif isinstance(frame, TTSAudioRawFrame) and self._llm_done:
            self._first_audio = self._first_audio or now
        elif isinstance(frame, BotStartedSpeakingFrame) and self._first_audio:
            await self._emit(now)

    async def _emit(self, bot_speaking: float) -> None:
        points = [self._speech_end, self._turn_end, self._llm_done, self._first_audio, bot_speaking]
        gaps = [max(0.0, (b - a) * 1000) for a, b in zip(points, points[1:], strict=False)]
        stages = {name: round(ms, 1) for name, ms in zip(STAGES, gaps, strict=True)}
        total = round(sum(gaps), 1)
        stt = (
            round(max(0.0, (self._transcript_at - self._speech_end) * 1000), 1)
            if self._transcript_at
            else None
        )
        self._reset()
        if total > MAX_VALID_TURN_MS:
            logger.debug(f"latency: discarded {total / 1000:.1f}s turn")
            return
        turn = TurnLatency(len(self.turns), total, stages, stt)
        self.turns.append(turn)
        logger.info(
            f"latency turn={turn.turn_index} total={total:.0f}ms "
            + " ".join(f"{k}={v:.0f}" for k, v in stages.items())
            + (f" (stt done +{stt:.0f}ms)" if stt is not None else "")
        )
        if self._on_turn:
            await self._on_turn(turn)

    def as_dicts(self) -> list[dict]:
        return [asdict(t) for t in self.turns]


class BargeInObserver(BaseObserver):
    """Time from an interruption to the bot actually going silent (server side)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        super().__init__()
        self._clock = clock
        self._bot_speaking = False
        self._interrupted_at: float | None = None
        self.stop_times_ms: list[float] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        # A frame is pushed once per hop; only count its first push.
        if not data.first_push:
            return
        frame = data.frame
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, InterruptionFrame) and self._bot_speaking:
            self._interrupted_at = self._clock()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._interrupted_at is not None:
                ms = (self._clock() - self._interrupted_at) * 1000
                self.stop_times_ms.append(round(ms, 1))
                logger.info(f"barge-in: bot stopped {ms:.0f} ms after interruption")
                self._interrupted_at = None


class UsageRecorder(BaseObserver):
    """Totals the usage Pipecat reports per service, for cost-per-session numbers."""

    def __init__(self):
        super().__init__()
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.stt_audio_secs = 0.0
        self.tts_characters = 0

    async def on_push_frame(self, data: FramePushed) -> None:
        if not data.first_push or not isinstance(data.frame, MetricsFrame):
            return
        for m in data.frame.data:
            if isinstance(m, LLMUsageMetricsData):
                self.llm_calls += 1
                self.prompt_tokens += m.value.prompt_tokens
                self.completion_tokens += m.value.completion_tokens
                self.cached_prompt_tokens += m.value.cache_read_input_tokens or 0
            elif isinstance(m, STTUsageMetricsData):
                self.stt_audio_secs += m.value.audio_seconds
            elif isinstance(m, TTSUsageMetricsData):
                self.tts_characters += m.value

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "stt_audio_secs": round(self.stt_audio_secs, 1),
            "tts_characters": self.tts_characters,
        }

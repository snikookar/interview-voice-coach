"""Per-stage latency and barge-in measurement.

Two observers watch frames flowing through the pipeline without changing them:

* ``LatencyRecorder`` wraps Pipecat's ``UserBotLatencyObserver``. For every turn it
  takes the breakdown of "user stopped speaking -> first bot audio" and folds
  Pipecat's fine-grained spans into the four stages of the spec's latency budget:
  turn detection, STT, LLM and TTS.
* ``BargeInObserver`` measures how long the bot keeps talking after the user
  starts speaking over it (target: under 300 ms).
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    MeasuredFrom,
    UserBotLatencyObserver,
)

# Pipecat span key -> spec stage. Anything unmapped is reported as "other".
STAGE_OF_SPAN = {
    "endpointing_wait": "turn",
    "turn_detection": "turn",
    "turn_completion": "turn",
    "waiting_for_user": "turn",
    "transcription": "stt",
    "first_request": "llm",
    "llm_inference": "llm",
    "llm_tool_call": "llm",
    "function_handler": "llm",
    "sentence_aggregation": "tts",
    "awaiting_speakable_text": "tts",
    "speech_synthesis": "tts",
    "output_transport": "transport",
}
STAGES = ("turn", "stt", "llm", "tts", "transport", "other")


@dataclass
class TurnLatency:
    turn_index: int
    total_ms: float
    stages_ms: dict[str, float]
    spans_ms: dict[str, float] = field(default_factory=dict)
    first_turn: bool = False


def fold_breakdown(breakdown: LatencyBreakdown) -> tuple[dict[str, float], dict[str, float]]:
    """Return (stage totals, raw span totals) in milliseconds."""
    stages = dict.fromkeys(STAGES, 0.0)
    spans: dict[str, float] = {}
    for c in breakdown.contributions:
        ms = c.duration_secs * 1000
        stages[STAGE_OF_SPAN.get(c.key, "other")] += ms
        spans[c.key] = spans.get(c.key, 0.0) + ms
    return {k: round(v, 1) for k, v in stages.items()}, {k: round(v, 1) for k, v in spans.items()}


LatencyCallback = Callable[[TurnLatency], Awaitable[None]]


class LatencyRecorder:
    """Collects one ``TurnLatency`` per bot response."""

    def __init__(self, on_turn: LatencyCallback | None = None):
        self.observer = UserBotLatencyObserver()
        self.turns: list[TurnLatency] = []
        self._on_turn = on_turn
        self.observer.add_event_handler("on_latency_breakdown", self._on_breakdown)

    async def _on_breakdown(self, _observer, breakdown: LatencyBreakdown) -> None:
        if not breakdown.contributions:
            return
        stages, spans = fold_breakdown(breakdown)
        turn = TurnLatency(
            turn_index=len(self.turns),
            total_ms=round(breakdown.total_secs * 1000, 1),
            stages_ms=stages,
            spans_ms=spans,
            first_turn=breakdown.measured_from == MeasuredFrom.CLIENT_CONNECTED,
        )
        self.turns.append(turn)
        logger.info(
            f"latency turn={turn.turn_index} total={turn.total_ms:.0f}ms "
            + " ".join(f"{k}={v:.0f}" for k, v in stages.items() if v)
        )
        if self._on_turn:
            await self._on_turn(turn)

    def as_dicts(self) -> list[dict]:
        return [asdict(t) for t in self.turns]


class BargeInObserver(BaseObserver):
    """Time from an interruption to the bot actually going silent."""

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

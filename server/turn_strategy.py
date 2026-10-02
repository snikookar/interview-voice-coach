"""End-of-turn detection tuned for interview answers.

Two needs pull VAD's ``stop_secs`` in opposite directions:

* STT wants SHORT voice segments: the segmented Whisper service transcribes each
  segment when VAD reports a stop, so with 0.2 s segments each sentence is
  transcribed while the candidate is still talking, and only the last sentence
  is left to transcribe after they finish.
* Turn-taking wants a LONG silence: interview answers pause up to ~0.7 s between
  complete sentences (measured on our clips), and Smart Turn calls a finished
  sentence "complete", so a 0.2 s trigger cuts candidates off mid-answer.

``MinSilenceTurnStopStrategy`` keeps VAD at 0.2 s and Smart Turn's verdict, but
only ends the turn once the silence has lasted ``min_silence_secs``. If the
candidate starts speaking again first, the pending end-of-turn is dropped.

Measured with the synthetic candidate (same answers, CPU, base.en):
VAD stop 0.8 s -> STT stage ~1.6 s; this strategy -> see docs/phases/05.
"""

import asyncio
import time

from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy


class MinSilenceTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    def __init__(self, *, min_silence_secs: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self._min_silence_secs = min_silence_secs
        self._silence_started: float | None = None
        self._deferred: asyncio.Task | None = None

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        # VAD reports the stop stop_secs after the voice actually ended.
        self._silence_started = time.monotonic() - frame.stop_secs
        await super()._handle_vad_user_stopped_speaking(frame)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._silence_started = None
        await self._cancel_deferred()
        await super()._handle_vad_user_started_speaking(frame)

    async def trigger_user_turn_stopped(self, **kwargs):
        if self._silence_started is None:
            return await super().trigger_user_turn_stopped(**kwargs)
        remaining = self._silence_started + self._min_silence_secs - time.monotonic()
        if remaining <= 0:
            return await super().trigger_user_turn_stopped(**kwargs)
        if self._deferred is None:
            self._deferred = self.task_manager.create_task(
                self._trigger_later(remaining, kwargs), f"{self}::min_silence"
            )

    async def _trigger_later(self, delay: float, kwargs: dict) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        # Clear first: ending the turn resets the strategy, which would otherwise
        # try to cancel this very task.
        self._deferred = None
        await super().trigger_user_turn_stopped(**kwargs)

    async def _cancel_deferred(self) -> None:
        if self._deferred:
            await self.task_manager.cancel_task(self._deferred)
            self._deferred = None

    async def cleanup(self):
        await self._cancel_deferred()
        await super().cleanup()

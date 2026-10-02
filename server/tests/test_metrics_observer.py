from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from metrics_observer import BargeInObserver, LatencyRecorder


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


async def _push(obs, frame, direction=FrameDirection.DOWNSTREAM, first=True):
    await obs.on_push_frame(SimpleNamespace(frame=frame, direction=direction, first_push=first))


def _vad_stop(clock: Clock, stop_secs: float = 0.2) -> VADUserStoppedSpeakingFrame:
    return VADUserStoppedSpeakingFrame(stop_secs=stop_secs, timestamp=clock.t)


def _audio() -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\0\0" * 160, sample_rate=16000, num_channels=1)


def _tool_call() -> FunctionCallInProgressFrame:
    return FunctionCallInProgressFrame(
        function_name="record_answer", tool_call_id="c1", arguments={}
    )


async def _one_turn(rec: LatencyRecorder, clock: Clock) -> None:
    clock.t = 1000.2  # voice ended at 1000.0, VAD confirms 0.2 s later
    await _push(rec, _vad_stop(clock))
    clock.t = 1000.6
    await _push(rec, TranscriptionFrame("answer", "u", "t"))
    clock.t = 1000.9
    await _push(rec, UserStoppedSpeakingFrame())
    clock.t = 1001.3
    await _push(rec, _tool_call())
    clock.t = 1001.5
    await _push(rec, _audio())
    clock.t = 1001.55
    await _push(rec, BotStartedSpeakingFrame())


async def test_stages_are_consecutive_gaps_that_sum_to_total():
    clock = Clock()
    seen = []

    async def on_turn(t):
        seen.append(t)

    rec = LatencyRecorder(on_turn=on_turn, clock=clock)
    await _one_turn(rec, clock)
    turn = rec.turns[0]
    assert turn.stages_ms == {"turn": 900.0, "llm": 400.0, "tts": 200.0, "output": 50.0}
    assert turn.total_ms == 1550.0
    assert turn.stt_ms == 600.0
    assert seen == [turn]


async def test_resumed_speech_restarts_the_measurement():
    clock = Clock()
    rec = LatencyRecorder(clock=clock)
    clock.t = 990.2
    await _push(rec, _vad_stop(clock))  # an earlier pause...
    await _push(rec, VADUserStartedSpeakingFrame())  # ...then the candidate went on
    await _one_turn(rec, clock)
    assert rec.turns[0].total_ms == 1550.0  # measured from the LAST speech end


async def test_upstream_copies_and_greeting_are_ignored():
    clock = Clock()
    rec = LatencyRecorder(clock=clock)
    await _push(rec, BotStartedSpeakingFrame())  # greeting: no candidate speech before it
    await _push(rec, _vad_stop(clock), direction=FrameDirection.UPSTREAM)
    await _push(rec, UserStoppedSpeakingFrame())
    assert rec.turns == []


async def test_barge_in_measures_interruption_to_silence():
    now = [0.0]
    obs = BargeInObserver(clock=lambda: now[0])
    await _push(obs, BotStartedSpeakingFrame())
    now[0] = 1.0
    await _push(obs, InterruptionFrame())
    now[0] = 1.18
    await _push(obs, BotStoppedSpeakingFrame())
    await _push(obs, BotStoppedSpeakingFrame(), first=False)  # repeated hop: ignored
    assert obs.stop_times_ms == [180.0]

    # An interruption while the bot is silent is not a barge-in.
    await _push(obs, InterruptionFrame())
    await _push(obs, BotStoppedSpeakingFrame())
    assert obs.stop_times_ms == [180.0]

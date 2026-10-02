from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
)
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    LatencyContribution,
    LatencyOwnerKind,
    MeasuredFrom,
)

from metrics_observer import BargeInObserver, LatencyRecorder, fold_breakdown


def _contribution(key: str, secs: float) -> LatencyContribution:
    return LatencyContribution(
        key=key,
        label=key,
        owner="test",
        owner_kind=list(LatencyOwnerKind)[0],
        start_time=0.0,
        duration_secs=secs,
    )


def _breakdown(**spans: float) -> LatencyBreakdown:
    contributions = [_contribution(k, v) for k, v in spans.items()]
    return LatencyBreakdown(
        contributions=contributions,
        total_secs=sum(spans.values()),
        measured_from=MeasuredFrom.USER_SILENCE,
    )


def test_fold_breakdown_maps_spans_to_spec_stages():
    stages, spans = fold_breakdown(
        _breakdown(
            endpointing_wait=0.2,
            turn_detection=0.1,
            transcription=0.3,
            llm_inference=0.5,
            sentence_aggregation=0.05,
            speech_synthesis=0.15,
            mystery_span=0.01,
        )
    )
    assert stages["turn"] == 300.0
    assert stages["stt"] == 300.0
    assert stages["llm"] == 500.0
    assert stages["tts"] == 200.0
    assert stages["other"] == 10.0
    assert spans["transcription"] == 300.0


async def test_latency_recorder_collects_turns_and_calls_back():
    seen = []

    async def on_turn(turn):
        seen.append(turn)

    rec = LatencyRecorder(on_turn=on_turn)
    await rec._on_breakdown(None, _breakdown(transcription=0.2, llm_inference=0.4))
    await rec._on_breakdown(None, LatencyBreakdown())  # empty breakdown is ignored
    assert len(rec.turns) == 1
    assert rec.turns[0].total_ms == 600.0
    assert seen[0].stages_ms["llm"] == 400.0
    assert rec.as_dicts()[0]["turn_index"] == 0


async def test_barge_in_measures_interruption_to_silence():
    now = [0.0]
    obs = BargeInObserver(clock=lambda: now[0])

    async def push(frame, first=True):
        await obs.on_push_frame(SimpleNamespace(frame=frame, first_push=first))

    await push(BotStartedSpeakingFrame())
    now[0] = 1.0
    await push(InterruptionFrame())
    now[0] = 1.18
    await push(BotStoppedSpeakingFrame())
    # Repeated hops of the same frame are ignored.
    await push(BotStoppedSpeakingFrame(), first=False)
    assert obs.stop_times_ms == [180.0]

    # An interruption while the bot is silent is not a barge-in.
    await push(InterruptionFrame())
    await push(BotStoppedSpeakingFrame())
    assert obs.stop_times_ms == [180.0]

import time
from pathlib import Path
from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.utils.text.base_text_aggregator import AggregationType

from flows.state import InterviewState
from session_recorder import SessionRecorder


def _recorder() -> SessionRecorder:
    state = InterviewState(
        session_id="s1",
        role="ai-engineer",
        level="mid",
        plan=[{"id": "q1", "question": "Q?", "key_points": ["a", "b"]}],
    )
    rec = SessionRecorder(state, Path("unused"))
    rec._t0 = time.monotonic()  # start() would also start audio recording
    return rec


async def _push(rec: SessionRecorder, frame) -> None:
    await rec.on_push_frame(SimpleNamespace(frame=frame, first_push=True))


def _tts(text: str) -> TTSTextFrame:
    return TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)


async def test_answer_segments_merge_into_one_user_turn_tagged_with_question():
    rec = _recorder()
    rec.state.advance()  # question q1 is being asked
    await _push(rec, BotStartedSpeakingFrame())
    await _push(rec, _tts("First question."))
    await _push(rec, BotStoppedSpeakingFrame())

    # The candidate pauses mid-answer: two speech segments, no bot reply in between.
    await _push(rec, UserStartedSpeakingFrame())
    await _push(rec, TranscriptionFrame("I would use recall at k.", "u", "t"))
    await _push(rec, UserStoppedSpeakingFrame())
    await _push(rec, UserStartedSpeakingFrame())
    await _push(rec, TranscriptionFrame("And MRR.", "u", "t"))
    await _push(rec, UserStoppedSpeakingFrame())

    turns = rec.finished_turns()
    assert [t.speaker for t in turns] == ["bot", "user"]
    assert turns[0].text == "First question."
    assert turns[1].text == "I would use recall at k. And MRR."
    assert turns[1].question_id == "q1" and turns[1].phase == "question"


async def test_interrupted_bot_turn_is_flagged_and_unspoken_text_dropped():
    rec = _recorder()
    await _push(rec, BotStartedSpeakingFrame())
    await _push(rec, _tts("Hello there."))
    await _push(rec, InterruptionFrame())
    await _push(rec, _tts("This was never played."))  # flushed on interruption
    await _push(rec, BotStoppedSpeakingFrame())
    bot = rec.finished_turns()[0]
    assert bot.interrupted
    assert bot.text == "Hello there."


async def test_empty_user_noise_is_dropped():
    rec = _recorder()
    await _push(rec, UserStartedSpeakingFrame())
    await _push(rec, UserStoppedSpeakingFrame())
    assert rec.finished_turns() == []

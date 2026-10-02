from types import SimpleNamespace

import pytest

from flows import interview_flow, tools
from flows.state import InterviewState


def _plan(n: int) -> list[dict]:
    return [
        {
            "id": f"q{i}",
            "topic": "rag",
            "question": f"Question number {i}?",
            "key_points": ["point A", "point B"],
            "follow_ups": ["Hint?"],
        }
        for i in range(n)
    ]


def _fm(n: int = 3, max_minutes: int = 12) -> SimpleNamespace:
    state = InterviewState(
        session_id="s1", role="ai-engineer", level="mid", plan=_plan(n), max_minutes=max_minutes
    )
    return SimpleNamespace(state={"interview": state})


def _spoken(node) -> str:
    return " ".join(a.get("text", "") for a in node.get("pre_actions", []))


def test_intro_greets_and_waits_for_introduction():
    fm = _fm()
    node = interview_flow.create_intro_node(fm.state["interview"])
    assert node["respond_immediately"] is False
    assert "introduce yourself" in _spoken(node)
    assert tools.next_question in node["functions"]


async def test_next_question_asks_first_question_from_bank():
    fm = _fm()
    _, node = await tools.next_question(fm, "Nice to meet you.")
    state = fm.state["interview"]
    assert state.index == 0 and state.phase == "question"
    assert "Question number 0?" in _spoken(node)
    assert _spoken(node).startswith("Nice to meet you.")
    # The question is spoken via tts_say: no second LLM call.
    assert node["respond_immediately"] is False


async def test_missing_point_triggers_exactly_one_follow_up():
    fm = _fm()
    await tools.next_question(fm, "Hi.")
    _, node = await tools.record_answer(fm, ["point A"], ["point B"], "What about B?", "Okay.")
    state = fm.state["interview"]
    assert state.phase == "follow_up"
    assert node["name"].startswith("follow_up")
    assert "What about B?" in _spoken(node)

    # Even if the follow-up answer still misses something, we move on.
    _, node = await tools.record_answer(fm, [], ["point B"], "Another hint?", "Thanks.")
    assert state.index == 1 and state.phase == "question"
    assert "Question number 1?" in _spoken(node)
    assert len(state.notes["q0"]) == 2


async def test_complete_answer_moves_on_without_follow_up():
    fm = _fm()
    await tools.next_question(fm, "Hi.")
    _, node = await tools.record_answer(fm, ["point A", "point B"], [], "", "Got it.")
    assert fm.state["interview"].index == 1
    assert node["name"] == "question_2"


async def test_last_answer_wraps_up_and_ends_conversation():
    fm = _fm(n=1)
    await tools.next_question(fm, "Hi.")
    _, node = await tools.record_answer(fm, ["point A"], [], "", "Thanks.")
    state = fm.state["interview"]
    assert node["name"] == "wrap_up" and state.ended
    assert state.end_reason == "completed"
    assert node["pre_actions"][-1]["type"] == "end_conversation"
    assert "goodbye" in _spoken(node)


async def test_out_of_time_skips_follow_up_and_wraps_up():
    fm = _fm(n=3, max_minutes=1)  # 60 s budget minus 90 s reserve: already out of time
    await tools.next_question(fm, "Hi.")
    state = fm.state["interview"]
    # next_question itself already wrapped up because there is no time left.
    assert state.ended and state.end_reason == "time"


async def test_candidate_can_end_early():
    fm = _fm()
    await tools.next_question(fm, "Hi.")
    _, node = await tools.end_interview(fm, "has to leave")
    assert node["name"] == "wrap_up"
    assert "let's stop here" in _spoken(node)


async def test_stale_tool_calls_are_ignored():
    from pipecat.flows import NO_RESPONSE

    fm = _fm(n=1)
    await tools.next_question(fm, "Hi.")
    # A second next_question from an in-flight response must not skip a question.
    _, node = await tools.next_question(fm, "Hi again.")
    assert node is NO_RESPONSE and fm.state["interview"].index == 0

    await tools.record_answer(fm, ["point A"], [], "", "Thanks.")  # -> wrap_up
    _, node = await tools.record_answer(fm, ["point A"], [], "", "Thanks.")
    assert node is NO_RESPONSE
    _, node = await tools.end_interview(fm, "bye")
    assert node is NO_RESPONSE


def test_speech_is_one_non_blocking_action():
    actions = interview_flow.say("Got it.", "", "Next question.", "What is RAG?")
    assert len(actions) == 1
    # A custom action type: Flows never waits for it, so node transitions can't stall.
    assert actions[0]["type"] == "speak" and callable(actions[0]["handler"])
    assert actions[0]["text"] == "Got it. Next question. What is RAG?"
    assert interview_flow.say("", "  ") == []


@pytest.mark.parametrize("fn", [tools.next_question, tools.record_answer, tools.end_interview])
def test_tools_have_llm_readable_schemas(fn):
    from pipecat.flows.types import FlowsDirectFunctionWrapper

    schema = FlowsDirectFunctionWrapper(function=fn).to_function_schema()
    assert schema.description
    assert all(p.get("description") for p in schema.properties.values())

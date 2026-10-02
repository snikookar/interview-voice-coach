"""LLM-callable tools of the interview flow: next_question, record_answer, end_interview.

Each tool is a Pipecat Flows "direct function": the signature and docstring become
the JSON schema the LLM sees, and the return value is ``(result, next_node)``.

Latency design: the LLM is called once per candidate turn. It grades the answer
inside the ``record_answer`` tool call (and writes the follow-up question as a tool
argument when needed). The next node then *speaks* prepared text with ``tts_say``
instead of running the LLM again, so the next question comes straight from the
bank with no second LLM round-trip.
"""

from loguru import logger
from pipecat.flows import FlowManager, NodeConfig

from flows import interview_flow as nodes
from flows.state import AnswerNote, InterviewState


def get_state(flow_manager: FlowManager) -> InterviewState:
    return flow_manager.state["interview"]


def _advance(state: InterviewState, acknowledgement: str) -> NodeConfig:
    """Move to the next question, or wrap up when out of questions or time."""
    if not state.has_next():
        state.end_reason = "time" if state.out_of_time else "completed"
        return nodes.create_wrap_up_node(state, acknowledgement)
    is_first = state.index < 0
    question = state.advance()
    is_last = state.index == len(state.plan) - 1
    return nodes.create_question_node(state, question, acknowledgement, is_first, is_last)


async def next_question(flow_manager: FlowManager, acknowledgement: str) -> tuple[dict, NodeConfig]:
    """Start the questions. Call this as soon as the candidate has finished introducing themselves.

    Args:
        acknowledgement: A warm reaction to their introduction, 3 to 12 words, for example "Nice to meet you, that's a great background."
    """
    state = get_state(flow_manager)
    logger.info(f"[flow] next_question (intro done) session={state.session_id}")
    return {"status": "starting questions"}, _advance(state, acknowledgement)


async def record_answer(
    flow_manager: FlowManager,
    covered_points: list[str],
    missing_points: list[str],
    follow_up_question: str,
    acknowledgement: str,
) -> tuple[dict, NodeConfig]:
    """Record the candidate's finished answer. Call this once they have clearly finished answering.

    Args:
        covered_points: Rubric key points the answer covered, copied from the rubric.
        missing_points: Rubric key points the answer missed, copied from the rubric, most important first.
        follow_up_question: One short spoken question that nudges the candidate toward the most important missing point without revealing it. Empty string if nothing important is missing, if they said they don't know, or if this was already a follow-up answer.
        acknowledgement: A brief neutral reaction of 2 to 6 words, like "Okay, thanks." or "Got it." Never say whether the answer was right.
    """
    state = get_state(flow_manager)
    state.add_note(
        AnswerNote(
            phase=state.phase,
            covered_points=covered_points,
            missing_points=missing_points,
            follow_up_question=follow_up_question,
        )
    )
    logger.info(
        f"[flow] record_answer q={state.current_id} phase={state.phase} "
        f"covered={len(covered_points)} missing={len(missing_points)}"
    )

    wants_follow_up = (
        state.phase == "question"
        and not state.follow_up_asked
        and bool(missing_points)
        and bool(follow_up_question.strip())
        and not state.out_of_time
    )
    if wants_follow_up:
        state.phase = "follow_up"
        state.follow_up_asked = True
        return {"status": "follow-up asked"}, nodes.create_follow_up_node(
            state, follow_up_question.strip(), acknowledgement
        )
    return {"status": "answer recorded"}, _advance(state, acknowledgement)


async def end_interview(flow_manager: FlowManager, reason: str) -> tuple[dict, NodeConfig]:
    """End the interview early. Only call this if the candidate explicitly asks to stop or end the interview.

    Args:
        reason: Why the interview is ending, in a few words.
    """
    state = get_state(flow_manager)
    state.end_reason = f"candidate: {reason}"
    logger.info(f"[flow] end_interview reason={reason!r}")
    return {"status": "ending"}, nodes.create_wrap_up_node(state, "")

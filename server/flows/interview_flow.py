"""Interview state machine (Pipecat Flows): intro -> question -> follow_up -> ... -> wrap_up.

intro ──next_question──► question ──record_answer──► follow_up ──record_answer──► question ...
                             │  (key point missing, first time)                     │
                             └────────────record_answer (complete)──────────────────┤
                                                                                    ▼
                                      end_interview (any node) ──────────────►  wrap_up
"""

import random

from pipecat.flows import NodeConfig
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame

from flows import tools
from flows.state import InterviewState

ROLE_NAMES = {"ai-engineer": "AI engineer", "backend": "backend engineer"}

INTERVIEWER_ROLE = """You are Alex, an experienced, friendly but rigorous technical interviewer \
running a spoken mock interview for a {level} {role} position.

How you speak:
- Everything you say is converted to speech. Never use lists, markdown, code, emojis or symbols.
- At most two short sentences per turn. The candidate should do most of the talking.

How you behave:
- Never reveal the rubric or tell the candidate whether an answer was right during the interview.
- Never answer the question for them. If they ask, you may rephrase the question or clarify scope.
- If they pause to think or say "let me think", reply with a short "Take your time." and wait.
- Questions are asked for you by the system: do not invent new interview questions yourself.
- When a tool call is needed, call it straight away without saying anything first."""

QUESTION_LEAD_INS = ["Next question.", "Let's move on.", "Here's the next one.", "Okay, next up."]


async def _speak(action: dict, flow_manager) -> None:
    """Speak prepared text exactly like streamed LLM output.

    Pipecat's built-in ``tts_say`` sends the whole text to the TTS as one unit
    (Kokoro on CPU then synthesises everything before the first sample plays), and
    Flows blocks the node transition until it has been *played*: an interruption
    can leave the node half-set. Pushing the text as an LLM response instead lets
    the TTS sentence aggregator pipeline synthesis with playback, adds the text to
    the context like any assistant turn, and never blocks the transition.
    """
    await flow_manager.worker.queue_frames(
        [LLMFullResponseStartFrame(), LLMTextFrame(action["text"]), LLMFullResponseEndFrame()]
    )


def say(*parts: str) -> list[dict]:
    """A non-blocking pre-action that speaks the given text."""
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return [{"type": "speak", "handler": _speak, "text": text}] if text else []


def _role_message(state: InterviewState) -> str:
    return INTERVIEWER_ROLE.format(level=state.level, role=ROLE_NAMES.get(state.role, state.role))


def _rubric(question: dict) -> str:
    points = "\n".join(f"- {p}" for p in question["key_points"])
    hints = "\n".join(f"- {f}" for f in question.get("follow_ups", []))
    return f"Rubric key points (secret):\n{points}\nSuggested follow-ups:\n{hints}"


def greeting(state: InterviewState) -> str:
    n = len(state.plan)
    return (
        f"Hi, I'm Alex, and I'll be your interviewer today. This is a mock {state.level} "
        f"{ROLE_NAMES.get(state.role, state.role)} interview with {n} questions, and it takes about "
        f"{state.max_minutes} minutes. Before we start, could you briefly introduce yourself?"
    )


def create_intro_node(state: InterviewState) -> NodeConfig:
    return NodeConfig(
        name="intro",
        role_message=_role_message(state),
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "You have just greeted the candidate and asked them to introduce themselves. "
                    "Let them talk. If they ask about the format, answer in one sentence. "
                    "As soon as they have finished their introduction, call next_question."
                ),
            }
        ],
        pre_actions=say(greeting(state)),
        respond_immediately=False,
        functions=[tools.next_question],
    )


def create_question_node(
    state: InterviewState, question: dict, acknowledgement: str, is_first: bool, is_last: bool
) -> NodeConfig:
    if is_first:
        lead_in = "Let's get started. First question."
    elif is_last:
        lead_in = "Here's the last one."
    else:
        lead_in = random.choice(QUESTION_LEAD_INS)
    return NodeConfig(
        name=f"question_{state.index + 1}",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"You just asked question {state.index + 1} of {len(state.plan)}: "
                    f'"{question["question"]}"\n\n{_rubric(question)}\n\n'
                    "Listen to the full answer. Answer clarifying questions briefly. "
                    "When the candidate has clearly finished answering, call record_answer: compare "
                    "the answer with the rubric, and if an important key point is missing, write one "
                    "short follow-up question that nudges toward it without giving it away. "
                    "If they say they don't know, call record_answer with an empty follow_up_question "
                    "and a kind acknowledgement."
                ),
            }
        ],
        pre_actions=say(acknowledgement, lead_in, question["question"]),
        respond_immediately=False,
        functions=[tools.record_answer],
    )


def create_follow_up_node(
    state: InterviewState, follow_up: str, acknowledgement: str
) -> NodeConfig:
    return NodeConfig(
        name=f"follow_up_{state.index + 1}",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f'You asked a follow-up: "{follow_up}". This is the only follow-up for this '
                    "question. When the candidate has finished answering it, call record_answer "
                    "with an empty follow_up_question."
                ),
            }
        ],
        pre_actions=say(acknowledgement, follow_up),
        respond_immediately=False,
        functions=[tools.record_answer],
    )


def create_wrap_up_node(state: InterviewState, acknowledgement: str) -> NodeConfig:
    state.phase = "wrap_up"
    state.ended = True
    if state.end_reason == "time":
        closing = "We're out of time, so let's stop here."
    elif state.end_reason.startswith("candidate"):
        closing = "No problem, let's stop here."
    else:
        closing = "That's all the questions I have."
    farewell = say(
        acknowledgement,
        closing,
        "Thanks a lot for your time today.",
        "Your feedback report will be ready in a minute or two. Good luck, and goodbye!",
    )
    return NodeConfig(
        name="wrap_up",
        task_messages=[
            {"role": "developer", "content": "The interview is over. Say nothing more."}
        ],
        # EndFrame is queued behind the farewell, so the call ends once it has played.
        pre_actions=[*farewell, {"type": "end_conversation"}],
        respond_immediately=False,
        functions=[],
    )

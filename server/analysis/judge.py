"""LLM-as-judge: score one answer against its rubric.

Design choices (validated against human scores in eval/judge_agreement):
* The rubric's key points are the scoring backbone; the judge must copy them
  verbatim into covered/missed, and code re-aligns them to the rubric afterwards,
  so key-point detection can be measured with precision/recall/F1.
* Anchored 1-5 scales for each dimension, temperature 0, JSON output.
* The transcript comes from speech recognition, so the judge is told to ignore
  transcription errors and filler words when scoring content.
"""

import difflib
import json
import re

# openai imports httpx lazily on first client construction; doing that on the event
# loop while a worker thread (offline Whisper) imports it too raced ("partially
# initialized module 'httpx'"). Import it here, at startup, on the main thread.
import httpx  # noqa: F401
from loguru import logger
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, Field, ValidationError

from config import get_settings


class AnswerScore(BaseModel):
    question_id: str
    covered_points: list[str]
    missed_points: list[str]
    correctness: int = Field(ge=1, le=5)
    depth: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)  # clear, logical; STAR for behavioral
    communication: int = Field(ge=1, le=5)
    better_answer_outline: str

    @property
    def mean(self) -> float:
        return (self.correctness + self.depth + self.structure + self.communication) / 4


SYSTEM_PROMPT = """You are a calibrated senior interviewer grading ONE answer from a mock job interview.
The answer is a speech-to-text transcript: ignore transcription errors, filler words and
grammar slips when judging content (but they may affect "communication").

Return ONLY a JSON object with these keys:
  covered_points: rubric key points the candidate clearly covered (copy the rubric text exactly)
  missed_points: rubric key points not covered (copy exactly). Every key point goes in exactly one list.
  correctness: 1-5   depth: 1-5   structure: 1-5   communication: 1-5
  better_answer_outline: 2-4 short sentences describing what a strong answer would add or change

A key point is covered only if its essential idea is stated, even in different words.
Naming a term without showing understanding does not count.

Scales (anchor your scores to these):
correctness   1 wrong or no answer · 2 major errors · 3 mostly right with gaps · 4 right, minor slips · 5 fully right
depth         1 none · 2 superficial buzzwords · 3 explains the main idea · 4 trade-offs or examples · 5 expert nuance, trade-offs and real experience
structure     1 rambling · 2 hard to follow · 3 understandable · 4 clear and logical · 5 crisp, well organised{star}
communication 1 incomprehensible · 2 unclear · 3 adequate · 4 clear and concise · 5 excellent, confident and precise

Use the full range. A candidate who says they don't know gets 1 for correctness and depth."""

STAR_NOTE = (
    "\n  (behavioral question: structure means STAR. 5 = clear Situation, Task, Action "
    "using 'I', and a measurable Result; deduct for each missing part)"
)

FEW_SHOT_USER = """Question: What is overfitting and how do you prevent it?
Rubric key points:
- The model memorises training data and fails to generalise
- Detected by a gap between training and validation performance
- Regularisation such as L1, L2, dropout or weight decay
- More data or data augmentation
- Early stopping and simpler models

Candidate answer (transcript):
Overfitting is when the model, um, learns the training set too well, like memorises it, so it does badly on new data. You see it when validation loss goes up while training loss keeps going down. To fix it I'd add dropout or L2, and stop training early."""

FEW_SHOT_ASSISTANT = json.dumps(
    {
        "covered_points": [
            "The model memorises training data and fails to generalise",
            "Detected by a gap between training and validation performance",
            "Regularisation such as L1, L2, dropout or weight decay",
            "Early stopping and simpler models",
        ],
        "missed_points": ["More data or data augmentation"],
        "correctness": 5,
        "depth": 3,
        "structure": 4,
        "communication": 4,
        "better_answer_outline": "Add that more data or augmentation reduces overfitting, and mention cross-validation for detection. A concrete example from a past project would show depth.",
    }
)


def build_messages(question: dict, answer_text: str, follow_up: str | None = None) -> list[dict]:
    behavioral = question.get("type") == "behavioral"
    rubric = "\n".join(f"- {p}" for p in question["key_points"])
    convo = answer_text.strip() or "(no answer)"
    if follow_up:
        convo += f"\n\n[Interviewer follow-up: {follow_up}]\n(The answer above includes the reply to it.)"
    user = (
        f"Question: {question['question']}\nRubric key points:\n{rubric}\n\n"
        f"Candidate answer (transcript):\n{convo}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(star=STAR_NOTE if behavioral else "")},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": user},
    ]


def align_points(claimed: list[str], rubric: list[str]) -> list[str]:
    """Map the judge's (possibly paraphrased) points back onto exact rubric strings."""
    aligned = []
    for c in claimed:
        if c in rubric:
            aligned.append(c)
            continue
        match = difflib.get_close_matches(c, rubric, n=1, cutoff=0.55)
        if match:
            aligned.append(match[0])
    return list(dict.fromkeys(aligned))  # dedupe, keep order


def parse_score(raw: str, question: dict) -> AnswerScore:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in judge output")
    data = json.loads(m.group(0))
    rubric = question["key_points"]
    covered = align_points(data.get("covered_points", []), rubric)
    # Every rubric point is either covered or missed, whatever the judge listed.
    data["covered_points"] = covered
    data["missed_points"] = [p for p in rubric if p not in covered]
    data["question_id"] = question["id"]
    for k in ("correctness", "depth", "structure", "communication"):
        data[k] = min(5, max(1, int(round(float(data.get(k, 1))))))
    data.setdefault("better_answer_outline", "")
    return AnswerScore.model_validate(data)


class Judge:
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        s = get_settings()
        self.model = model or s.effective_judge_model
        self.client = client or AsyncOpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
        self._json_mode = True
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def _complete(self, messages: list[dict]) -> str:
        kwargs = {"model": self.model, "messages": messages, "temperature": 0}
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            if not self._json_mode:
                raise
            logger.info("Judge endpoint rejected JSON mode; falling back to plain text")
            self._json_mode = False
            return await self._complete(messages)
        usage = getattr(resp, "usage", None)  # optional in the API, absent on some endpoints
        if usage:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += usage.prompt_tokens
            self.usage["completion_tokens"] += usage.completion_tokens
        return resp.choices[0].message.content or ""

    async def score(
        self, question: dict, answer_text: str, follow_up: str | None = None
    ) -> AnswerScore:
        if not answer_text.strip():
            return AnswerScore(
                question_id=question["id"],
                covered_points=[],
                missed_points=list(question["key_points"]),
                correctness=1,
                depth=1,
                structure=1,
                communication=1,
                better_answer_outline="No answer was given.",
            )
        messages = build_messages(question, answer_text, follow_up)
        last_error: Exception | None = None
        for _ in range(2):  # one retry on malformed output
            raw = await self._complete(messages)
            try:
                return parse_score(raw, question)
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                last_error = e
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"Invalid output ({e}). Reply with the JSON object only.",
                    },
                ]
        raise RuntimeError(f"Judge failed for {question['id']}: {last_error}")

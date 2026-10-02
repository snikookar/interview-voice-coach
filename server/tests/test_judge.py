import json
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from analysis.judge import Judge, align_points, parse_score

QUESTION = {
    "id": "ml-002",
    "type": "technical",
    "question": "What is overfitting and how do you prevent it?",
    "key_points": [
        "The model memorises training data and fails to generalise",
        "Regularisation such as L1, L2, dropout or weight decay",
        "More data or data augmentation",
    ],
}


class FakeCompletions:
    def __init__(self, outputs, reject_json_mode=False):
        self.outputs = list(outputs)
        self.calls = []
        self.reject_json_mode = reject_json_mode

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_json_mode and "response_format" in kwargs:
            req = httpx.Request("POST", "http://x")
            raise BadRequestError(
                "no json mode", response=httpx.Response(400, request=req), body=None
            )
        content = self.outputs.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _judge(outputs, **kw) -> tuple[Judge, FakeCompletions]:
    completions = FakeCompletions(outputs, **kw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return Judge(client=client, model="test-model"), completions


GOOD = json.dumps(
    {
        "covered_points": [
            "Model memorises training data and fails to generalise",
            "dropout or weight decay",
        ],
        "missed_points": [],
        "correctness": 4,
        "depth": 3.4,
        "structure": 9,
        "communication": 4,
        "better_answer_outline": "Mention data augmentation.",
    }
)


def test_parse_aligns_paraphrased_points_and_derives_missed():
    score = parse_score(GOOD, QUESTION)
    # Paraphrases ("dropout or weight decay") map onto the exact rubric strings.
    assert score.covered_points == QUESTION["key_points"][:2]
    # The judge listed nothing as missed, but uncovered rubric points always are.
    assert score.missed_points == QUESTION["key_points"][2:]
    assert score.depth == 3 and score.structure == 5  # rounded and clamped


def test_align_points_drops_inventions():
    assert align_points(["Talks about GPUs"], QUESTION["key_points"]) == []


async def test_judge_retries_once_on_malformed_output():
    judge, calls = _judge(["Sure! Here's my evaluation.", f"```json\n{GOOD}\n```"])
    score = await judge.score(QUESTION, "It memorises the data, so use dropout.")
    assert score.correctness == 4
    assert len(calls.calls) == 2
    assert calls.calls[0]["temperature"] == 0
    assert calls.calls[0]["response_format"] == {"type": "json_object"}


async def test_judge_falls_back_when_json_mode_unsupported():
    judge, calls = _judge([GOOD], reject_json_mode=True)
    await judge.score(QUESTION, "answer")
    assert "response_format" not in calls.calls[-1]


async def test_empty_answer_scores_minimum_without_calling_llm():
    judge, calls = _judge([])
    score = await judge.score(QUESTION, "   ")
    assert score.correctness == 1 and score.missed_points == QUESTION["key_points"]
    assert calls.calls == []


async def test_judge_gives_up_after_two_bad_outputs():
    judge, _ = _judge(["nope", "still nope"])
    with pytest.raises(RuntimeError):
        await judge.score(QUESTION, "answer")

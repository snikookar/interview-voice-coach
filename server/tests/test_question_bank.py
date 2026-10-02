from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from question_bank.bank import LEVELS, ROLES, load_bank
from question_bank.retriever import chunk_posting, mmr_select

EXAMPLE_POSTING = Path(__file__).parents[2] / "examples" / "job_posting_ai_engineer.txt"


def test_bank_loads_and_has_mvp_size():
    bank = load_bank()
    assert len(bank) >= 120
    assert len({q.id for q in bank}) == len(bank)


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("level", LEVELS)
def test_every_role_level_can_fill_an_interview(role, level):
    bank = load_bank()
    technical = [q for q in bank if role in q.role and level in q.level and q.type == "technical"]
    behavioral = [q for q in bank if role in q.role and level in q.level and q.type == "behavioral"]
    assert len(technical) >= 12, f"{role}/{level} has only {len(technical)} technical questions"
    assert len(behavioral) >= 3
    # At least three distinct topics, so MMR can diversify.
    assert len({q.topic for q in technical}) >= 3


def test_rubrics_are_specific():
    for q in load_bank():
        assert all(len(kp) >= 10 for kp in q.key_points), q.id
        assert q.follow_ups, f"{q.id} has no follow-up"


def test_chunk_posting_splits_requirements_and_drops_headers():
    chunks = chunk_posting(EXAMPLE_POSTING.read_text(encoding="utf-8"))
    assert len(chunks) >= 5
    assert not any(c.strip().lower() in {"requirements:", "responsibilities:"} for c in chunks)
    assert any("reranking" in c for c in chunks)


def test_mmr_prefers_diverse_results():
    # Items 0 and 1 are near-duplicates; item 2 is less relevant but different.
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    relevance = np.array([1.0, 0.95, 0.6])
    picks = mmr_select(relevance, vectors, ["a", "b", "c"], k=2, lambda_=0.5)
    assert picks == [0, 2]


def test_mmr_respects_topic_cap_and_still_fills_k():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(6, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    relevance = np.linspace(1, 0.5, 6)
    topics = ["rag", "rag", "rag", "rag", "llm", "llm"]
    picks = mmr_select(relevance, vectors, topics, k=3, max_per_topic=2)
    counts = Counter(topics[i] for i in picks)
    assert len(picks) == 3 and counts["rag"] <= 2

    # Only one topic available: the cap relaxes rather than returning too few.
    picks = mmr_select(relevance[:4], vectors[:4], topics[:4], k=4, max_per_topic=1)
    assert len(picks) == 4

"""Question selection with RAG: embed the role and job posting, search pgvector, diversify with MMR.

uv run python -m question_bank.retriever index          # (re)embed the YAML bank
uv run python -m question_bank.retriever plan --role ai-engineer --level mid --posting posting.txt
"""

import argparse
import asyncio
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy import any_, delete, literal, select

from config import get_settings
from db import QuestionRow, init_db, session_scope
from question_bank.bank import Question, load_bank

ROLE_LABELS = {"ai-engineer": "AI / ML engineer", "backend": "backend software engineer"}
LEVEL_TARGET_DIFFICULTY = {"junior": 1.5, "mid": 2.75, "senior": 3.75}
CANDIDATES_PER_QUERY = 30


# ───────────────────────── embeddings ─────────────────────────


@lru_cache
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embedding_model)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def embed_passages(texts: list[str]) -> np.ndarray:
    return _normalise(np.array(list(_embedder().passage_embed(texts)), dtype=np.float32))


def embed_queries(texts: list[str]) -> np.ndarray:
    # bge models expect an instruction prefix on queries; fastembed's query_embed adds it.
    return _normalise(np.array(list(_embedder().query_embed(texts)), dtype=np.float32))


# ───────────────────────── indexing ─────────────────────────


async def sync_bank() -> int:
    """Embed new or changed YAML questions into Postgres. Returns the number updated."""
    model = get_settings().embedding_model
    questions = load_bank()
    async with session_scope() as db:
        existing = dict((await db.execute(select(QuestionRow.id, QuestionRow.content_hash))).all())
        changed = [q for q in questions if existing.get(q.id) != q.content_hash(model)]
        if changed:
            logger.info(f"Embedding {len(changed)} question(s) with {model}")
            vectors = await asyncio.to_thread(embed_passages, [q.embedding_text() for q in changed])
            for q, vec in zip(changed, vectors, strict=True):
                await db.merge(_to_row(q, vec, model))
        removed = set(existing) - {q.id for q in questions}
        if removed:
            await db.execute(
                delete(QuestionRow).where(QuestionRow.id.in_(removed), QuestionRow.source == "bank")
            )
        await db.commit()
    return len(changed)


def _to_row(q: Question, vec: np.ndarray, model: str) -> QuestionRow:
    return QuestionRow(
        id=q.id,
        roles=list(q.role),
        levels=list(q.level),
        topic=q.topic,
        type=q.type,
        difficulty=q.difficulty,
        source=q.source,
        data=q.model_dump(),
        content_hash=q.content_hash(model),
        embedding=vec.tolist(),
    )


# ───────────────────────── selection ─────────────────────────


def chunk_posting(text: str, max_chars: int = 250) -> list[str]:
    """Split a job posting into requirement-sized chunks (one bullet or short paragraph).

    One vector for a whole posting blurs distinct requirements together, and
    postings often exceed the embedding model's 512-token window. Each bullet
    ("hybrid search, reranking", "Kubernetes, CI/CD") becomes its own query, so the
    plan can cover several requirements instead of the posting's average.
    """
    lines = [re.sub(r"^\s*[-*•\d.)]+\s*", "", ln).strip() for ln in text.splitlines()]
    chunks: list[str] = []
    buffer = ""
    for line in lines:
        if not line:
            continue
        # Headers like "Requirements:" carry no content on their own.
        if line.endswith(":") and len(line) < 40:
            continue
        buffer = f"{buffer} {line}".strip()
        if len(buffer) >= 40 or len(buffer) >= max_chars:
            chunks.append(buffer[:max_chars])
            buffer = ""
    if len(buffer) >= 20:
        chunks.append(buffer)
    return chunks[:20]


def mmr_select(
    relevance: np.ndarray,
    vectors: np.ndarray,
    topics: list[str],
    k: int,
    lambda_: float = 0.7,
    max_per_topic: int = 2,
) -> list[int]:
    """Maximal Marginal Relevance with a per-topic cap.

    score(i) = lambda * relevance(i) - (1 - lambda) * max_sim(i, already selected)
    """
    selected: list[int] = []
    topic_counts: Counter[str] = Counter()
    sims = vectors @ vectors.T
    remaining = set(range(len(relevance)))
    while remaining and len(selected) < k:
        best, best_score = None, -np.inf
        for i in remaining:
            if topic_counts[topics[i]] >= max_per_topic:
                continue
            redundancy = max((sims[i, j] for j in selected), default=0.0)
            score = lambda_ * relevance[i] - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = i, score
        if best is None:  # every remaining candidate hits its topic cap
            max_per_topic += 1
            continue
        selected.append(best)
        topic_counts[topics[best]] += 1
        remaining.discard(best)
    return selected


async def _candidates(
    query_vecs: np.ndarray, role: str, level: str, qtype: str, exclude_ids: set[str]
) -> dict[str, tuple[QuestionRow, float]]:
    """Top-k nearest questions per query vector (pgvector cosine distance), unioned."""
    found: dict[str, tuple[QuestionRow, float]] = {}
    async with session_scope() as db:
        for qv in query_vecs:
            distance = QuestionRow.embedding.cosine_distance(qv.tolist())
            stmt = (
                select(QuestionRow, distance.label("distance"))
                .where(
                    literal(role) == any_(QuestionRow.roles),
                    literal(level) == any_(QuestionRow.levels),
                    QuestionRow.type == qtype,
                )
                .order_by(distance)
                .limit(CANDIDATES_PER_QUERY)
            )
            for row, dist in (await db.execute(stmt)).all():
                if row.id in exclude_ids:
                    continue
                sim = 1.0 - float(dist)
                if row.id not in found or sim > found[row.id][1]:
                    found[row.id] = (row, sim)
    return found


async def build_plan(
    role: str,
    level: str,
    job_posting: str | None = None,
    num_questions: int = 5,
    mode: str = "technical",
    exclude_ids: set[str] | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Pick and order the questions for one interview."""
    rng = random.Random(seed)
    exclude_ids = exclude_ids or set()

    # With a posting, its requirements alone drive relevance (the SQL filter already
    # enforces role and level); a generic role query would otherwise dominate.
    posting_chunks = chunk_posting(job_posting) if job_posting else []
    queries = posting_chunks or [
        f"Interview questions for a {level} {ROLE_LABELS.get(role, role)}."
    ]
    query_vecs = await asyncio.to_thread(embed_queries, queries)

    if mode == "behavioral":
        slots = {"behavioral": num_questions}
    else:
        n_beh = 1 if num_questions >= 4 else 0
        slots = {"technical": num_questions - n_beh, "behavioral": n_beh}

    plan: list[dict] = []
    for qtype, k in slots.items():
        if k == 0:
            continue
        found = await _candidates(query_vecs, role, level, qtype, exclude_ids)
        if len(found) < k:  # not enough unseen questions: allow repeats
            found = await _candidates(query_vecs, role, level, qtype, set())
        rows = list(found.values())
        target = LEVEL_TARGET_DIFFICULTY.get(level, 2.5)
        # Without a posting, relevance to a generic role query is nearly flat, so a
        # little noise makes each practice session different.
        noise = 0.0 if posting_chunks else 0.04
        relevance = np.array(
            [sim - 0.03 * abs(row.difficulty - target) + rng.gauss(0, noise) for row, sim in rows]
        )
        # Cosine scores from small embedding models are compressed into a narrow band
        # (~0.55-0.75); rescale so MMR's relevance and diversity terms are comparable.
        span = relevance.max() - relevance.min()
        relevance = (relevance - relevance.min()) / span if span > 0 else relevance * 0
        vectors = _normalise(np.array([np.asarray(row.embedding) for row, _ in rows]))
        picks = mmr_select(relevance, vectors, [row.topic for row, _ in rows], k)
        chosen = [rows[i][0] for i in picks]
        chosen.sort(key=lambda r: r.difficulty)  # warm up with easier questions
        plan.extend({**r.data, "relevance": round(float(found[r.id][1]), 3)} for r in chosen)
    return plan


# ───────────────────────── CLI ─────────────────────────


async def _run(cmd: str, role: str, level: str, posting: str | None, n: int) -> None:
    await init_db()
    print(f"indexed {await sync_bank()} changed question(s)")
    if cmd == "plan":
        for q in await build_plan(role, level, posting, n):
            print(
                f"[{q['topic']:<18}] d{q['difficulty']} rel={q['relevance']:.2f}  {q['question']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    p = sub.add_parser("plan")
    p.add_argument("--role", default="ai-engineer")
    p.add_argument("--level", default="mid")
    p.add_argument("--posting", help="path to a job posting text file")
    p.add_argument("-n", type=int, default=5)
    args = parser.parse_args()
    posting = (
        Path(args.posting).read_text(encoding="utf-8") if getattr(args, "posting", None) else None
    )
    asyncio.run(
        _run(
            args.cmd,
            getattr(args, "role", ""),
            getattr(args, "level", ""),
            posting,
            getattr(args, "n", 0),
        )
    )


if __name__ == "__main__":
    main()

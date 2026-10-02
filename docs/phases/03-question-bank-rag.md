# Phase 3: Question bank, embeddings and RAG question selection

**Goal:** given a role, a level and optionally a job posting, choose about 5 relevant, *diverse* questions, each with a rubric the interviewer and judge can use.

## Files

| File | Purpose |
|---|---|
| `server/question_bank/questions.yaml` | **132 questions** across 12 topics, for 2 roles and 3 levels. Each has `key_points` (the rubric), `follow_ups` and a `difficulty` |
| `server/question_bank/bank.py` | Pydantic schema plus loader. Rejects malformed or duplicate questions at load time |
| `server/question_bank/retriever.py` | Embedding, incremental indexing into pgvector, posting chunking, MMR selection, and a CLI |
| `server/db.py` | SQLAlchemy models (`questions`, `sessions`, `turns`), the async engine, and an embedded-Postgres fallback |
| `docker-compose.yml` | `pgvector/pgvector:pg17` for people who use Docker |
| `examples/job_posting_ai_engineer.txt` | A sample posting used by the tests and the demo |
| `server/tests/test_question_bank.py` | Bank coverage per role and level, chunking, and MMR behaviour |

Try it:
```bash
cd server
uv run python -m question_bank.retriever plan --role ai-engineer --level senior --posting ../examples/job_posting_ai_engineer.txt -n 6
```

## Selection algorithm

```
posting ─► split into bullet-sized chunks ─► embed each (bge query mode)
                                                │
          SQL filter: role ∈ roles, level ∈ levels, type = technical|behavioral
                                                │
 pgvector: top-30 nearest questions per chunk (cosine) ─► union, relevance = best chunk match
                                                │
 relevance -= 0.03 × |difficulty − level target|   (seniors get harder questions)
 min-max normalise ─► MMR (λ = 0.7, ≤ 2 per topic) ─► sort by difficulty ─► +1 behavioral at the end
```
Questions asked in your recent sessions are excluded (Phase 5), so repeated practice sessions vary.

### What I measured and changed
The first version embedded the **whole posting as one query** and added a generic "interview for a senior AI engineer" query. For a posting about RAG, evaluation and agents it picked LoRA and multi-agent design, and **no RAG question at all**. Two fixes:
1. **Bullet-level chunks.** Each requirement ("hybrid search, reranking") becomes its own query, so the plan covers several requirements instead of the posting's average. The generic query is only used when there's no posting.
2. **Normalising relevance before MMR.** bge-small cosine scores sit in a narrow 0.55–0.75 band, so MMR's diversity penalty was overpowering relevance. After min-max scaling, both terms are comparable.

Result for the same posting: reranking, query transformation, LLM latency and cost, observability, semantic search design, and one behavioral question.

## Technology choices

### Vector store: **PostgreSQL + pgvector** rather than Chroma, Qdrant or Pinecone
- We need a relational DB anyway (sessions, turns, reports, progress). One database means one backup, one connection pool and transactional consistency between questions and sessions.
- At 132 rows (or even 10k), a dedicated vector DB adds operational cost with no benefit. I don't even build an HNSW index: an exact scan is faster at this size and gives perfect recall.
- The `vector` column has **no fixed dimension**, so changing `EMBEDDING_MODEL` just re-embeds. A content hash includes the model name, so only changed questions are re-embedded.

### Embedded Postgres fallback (**pgserver**)
The dev machine has no Docker. `pgserver` is a pip package that bundles real Postgres binaries **with pgvector**. When `DATABASE_URL` is empty, `db.py` starts it under `server/data/pg`. It runs the same SQL as production, unlike swapping in SQLite (no arrays, no vectors).

### Embeddings: **bge-small-en-v1.5 via fastembed** (spec suggested bge-m3)
| | bge-m3 | bge-small-en-v1.5 (chosen) |
|---|---|---|
| Size | 2.2 GB, needs PyTorch (+2 GB) | 67 MB, ONNX |
| Languages | 100+ | English |
| Max input | 8192 tokens | 512 tokens |

bge-m3's strengths are multilingual support and long inputs. The interviews are English-only by spec, and **chunking the posting** removes the need for long inputs (it also gives better results, see above). The whole voice stack (Silero, Smart Turn, Whisper, Kokoro) runs on ONNX or CTranslate2 **without PyTorch**, and keeping it that way keeps the Docker image small and the startup fast. Swap it with `EMBEDDING_MODEL=...`.

### MMR rather than plain top-k
Top-k for a RAG-heavy posting returns five near-identical RAG questions. MMR trades a little relevance for coverage. The per-topic cap is a hard guarantee on top of that soft penalty.

### Rubrics in YAML rather than in the DB
Questions are content, and content belongs in git: reviewable diffs, PRs that add questions, and tests that validate them. The DB is a derived index, rebuilt automatically from the YAML at startup.

### SQLAlchemy 2.0 async + asyncpg
The API and the bot share one asyncio event loop with real-time audio, so blocking DB drivers (psycopg2) would stall audio frames. asyncpg is the fastest Postgres driver for asyncio, and SQLAlchemy gives typed models without hand-written SQL.

# Phase 0: Repo scaffolding

**Goal:** a reproducible repo that anyone can clone and run with one command per side (server and client).

## Files created

| File | Purpose |
|---|---|
| `.gitignore` | Keeps secrets (`.env`), virtualenvs, `node_modules`, recordings and model caches out of git |
| `.env.example` | Documents every setting. You copy it to `.env`, and the defaults run everything locally except the LLM |
| `server/pyproject.toml` | Python dependencies, optional extras (`embedded-db`, `tracing`, `eval`), plus pytest and ruff config |
| `server/uv.lock` | Exact pinned versions, so CI and your machine install the same thing |
| `server/.python-version` | Pins Python 3.12 |
| `server/config.py` | A single typed `Settings` object, loaded from `.env` with pydantic-settings |
| `docs/SPEC.md` | The original project spec |

## Technology choices

### Python package manager: **uv**
| Option | Verdict |
|---|---|
| pip + requirements.txt | No lockfile by default and slow resolution. It also can't install the Python version |
| Poetry | Good lockfile, but slow, and it doesn't manage Python versions |
| **uv** ✅ | 10–100× faster installs, a cross-platform lockfile, and **it installs Python itself** |

The last point decided it. This machine has Python 3.14, but Pipecat's native dependencies (onnxruntime, ctranslate2 and kokoro-onnx) only ship wheels up to 3.12/3.13. `uv` downloads 3.12 automatically when it sees `.python-version`, so nobody has to manage Python versions by hand.

### Settings: **pydantic-settings**
The alternative is reading `os.getenv` everywhere. With one typed `Settings` class, a typo like `TTS_PROVIDER=kokro` fails at startup (`Literal["kokoro","piper","deepgram"]`) instead of failing mid-interview. The class is cached with `lru_cache`, so every module shares one instance.

### Optional extras instead of one large install
- `embedded-db`: [`pgserver`](https://github.com/orm011/pgserver) bundles a real Postgres binary **with pgvector**. You can run the whole app without Docker (useful on Windows).
- `tracing`: OpenTelemetry exporter, only if you use Langfuse.
- `eval`: scipy, scikit-learn and matplotlib, only needed for the benchmark scripts.

The runtime server stays lean, which also keeps the Docker image smaller.

### Monorepo layout (`server/` + `client/` + `eval/`)
Follows the spec. The Python and TypeScript sides have separate toolchains, so they live in sibling folders with their own lockfiles. A single CI workflow tests both.

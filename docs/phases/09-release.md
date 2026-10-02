# Phase 9: Docker, CI and release

**Goal:** anyone can clone, run and verify the project, and every push is checked automatically.

## Files

| File | Purpose |
|---|---|
| `server/Dockerfile` | `python:3.12-slim` + uv. The dependency layer is cached separately from code. Models and recordings live on one `/data` volume |
| `client/Dockerfile` | Multi-stage Node 22 build: `next build`, then a slim `next start` image |
| `docker-compose.yml` | `db` (pgvector) by default. `--profile app` adds the API and client with **host networking** |
| `.dockerignore`, `client/.dockerignore` | Keep the multi-GB `.venv`, `node_modules` and recordings out of the build context |
| `.github/workflows/ci.yml` | Server: ruff lint/format + pytest. Client: eslint + route typegen + tsc + `next build` |
| `README.md` | Project overview, results, quick start, architecture, and links to these phase notes |
| `LICENSE` | MIT |

## Technology choices

### Host networking for the voice server, not port mapping
WebRTC media flows over UDP on ports negotiated at runtime (ICE). aiortc, under SmallWebRTC, picks random ephemeral ports, and Docker's `-p` mapping needs fixed ones. The honest options are:
| Option | Trade-off |
|---|---|
| `network_mode: host` ✅ | Simple, and works on Linux servers. Docker Desktop on Mac/Windows doesn't support it fully |
| TURN server (coturn) | Relays media through one known port. More infrastructure, more latency |
| Switch transport to Daily | Media goes through Daily's SFU, no UDP from your server. Needs an account, and is the right call for public deployment (spec, Risks table) |

So Compose runs the full stack with host networking on Linux, and the README tells Mac/Windows users to run Postgres in Docker and the app natively (or use the embedded DB and no Docker at all).

### uv in Docker
`uv sync --locked --no-install-project` installs exactly the lockfile versions in one cached layer. `UV_COMPILE_BYTECODE=1` speeds up cold starts. The same lockfile drives CI, so "works on my machine" and "works in CI" use identical dependency versions.

### What CI runs, and what it doesn't
CI runs everything that is **deterministic and model-free**: 44 unit tests (flow state machine, latency arithmetic, turn reconstruction, wall-clock audio placement, judge parsing and retries, filler rules, MMR, question bank validation), plus lint, type-check and production build of the client.

The end-to-end WebRTC harness and the evals are **not** in CI. They download about 1 GB of models, take minutes, and their latency numbers depend on the hardware, so a shared CI runner would produce noise rather than signal. They are one command each locally (see `eval/README.md`), and their outputs are committed under `eval/results/`.

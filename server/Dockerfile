# Interview Voice Coach: API + voice bot (CPU image).
# Build from the repo root:  docker build -f server/Dockerfile -t voice-coach-server .
FROM python:3.12-slim

# libgomp: OpenMP runtime for onnxruntime / CTranslate2 (Whisper, Silero, Smart Turn, Piper)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app/server
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first (cached layer), then the code.
COPY server/pyproject.toml server/uv.lock server/.python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY server/ ./

# All downloaded models and recordings live on one volume.
ENV PATH="/app/server/.venv/bin:$PATH" \
    DATA_DIR=/data \
    HF_HOME=/data/hf \
    FASTEMBED_CACHE_PATH=/data/fastembed
VOLUME /data
EXPOSE 7860

CMD ["python", "api.py"]

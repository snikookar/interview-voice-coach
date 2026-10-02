"""Pre-download every local model so the first interview doesn't stall.

    uv run python download_models.py            # models for the configured providers
    uv run python download_models.py --all      # also Piper + the analysis Whisper model

Large files are downloaded with HTTP range-resume and a size check, because a
dropped connection otherwise leaves a truncated model that later fails to load.
"""

import argparse
import time
from pathlib import Path

import requests
from loguru import logger

from config import get_settings

KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def kokoro_dir() -> Path:
    return get_settings().data_dir / "models" / "kokoro"


def _remote_size(url: str, attempts: int = 5) -> int | None:
    """Size from a HEAD request. Hugging Face sends x-linked-size for LFS files."""
    for attempt in range(1, attempts + 1):
        try:
            return _head_size(url)
        except requests.RequestException as e:
            logger.warning(f"HEAD {url} failed ({e}); retry {attempt}/{attempts}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not reach {url}")


def _head_size(url: str) -> int | None:
    headers = requests.head(url, allow_redirects=False, timeout=30).headers
    size = headers.get("x-linked-size") or headers.get("Content-Length")
    if headers.get("Location") and not headers.get("x-linked-size"):
        size = requests.head(url, allow_redirects=True, timeout=30).headers.get("Content-Length")
    return int(size) if size else None


def download_resumable(url: str, dest: Path, attempts: int = 5) -> None:
    # The final name only appears after a size-verified rename, so existing == complete.
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() and total else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        logger.info(
            f"{dest.name}: {have / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB (attempt {attempt})"
        )
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                mode = "ab" if resp.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if total is None:  # size unknown (small files): a clean finish is complete
                part.replace(dest)
                return
        except requests.RequestException as e:
            logger.warning(f"{dest.name}: {e}")
        if total and part.exists() and part.stat().st_size == total:
            part.replace(dest)
            logger.info(f"{dest.name}: done")
            return
    raise RuntimeError(f"Could not download {url}")


def ensure_kokoro() -> tuple[Path, Path]:
    d = kokoro_dir()
    for name, url in KOKORO_FILES.items():
        download_resumable(url, d / name)
    return d / "kokoro-v1.0.onnx", d / "voices-v1.0.bin"


def ensure_whisper(model: str) -> None:
    from faster_whisper.utils import download_model

    logger.info(f"Whisper: {model}")
    download_model(model)


def ensure_embeddings(model: str) -> None:
    from fastembed import TextEmbedding

    logger.info(f"Embeddings: {model}")
    TextEmbedding(model_name=model)


def piper_dir() -> Path:
    return get_settings().data_dir / "models" / "piper"


def ensure_piper(voice: str) -> Path:
    """Download a Piper voice (model + config), e.g. en_US-ryan-high."""
    lang, name, quality = voice.split("-")
    base = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{lang.split('_')[0]}/{lang}/{name}/{quality}/{voice}"
    )
    for suffix in (".onnx", ".onnx.json"):
        download_resumable(base + suffix, piper_dir() / f"{voice}{suffix}")
    return piper_dir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="download every local model")
    args = parser.parse_args()
    s = get_settings()

    if args.all or s.tts_provider == "kokoro":
        ensure_kokoro()
    if args.all or s.tts_provider == "piper":
        ensure_piper(s.piper_voice)
    if args.all or s.stt_provider == "local":
        ensure_whisper(s.whisper_model)
    ensure_whisper(s.analysis_whisper_model)
    ensure_embeddings(s.embedding_model)


if __name__ == "__main__":
    main()

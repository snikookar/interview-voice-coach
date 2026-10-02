"""Central settings, loaded from environment variables and the repo-root `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", SERVER_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenAI-compatible)
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    judge_model: str = ""

    # STT
    stt_provider: Literal["local", "cloud"] = "local"
    whisper_model: str = "base.en"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    analysis_whisper_model: str = "small.en"

    # TTS
    tts_provider: Literal["kokoro", "piper", "deepgram"] = "piper"
    kokoro_voice: str = "af_heart"
    piper_voice: str = "en_US-ryan-medium"
    deepgram_api_key: str = ""
    deepgram_tts_voice: str = "aura-2-thalia-en"

    # Storage
    database_url: str = ""
    data_dir: Path = SERVER_DIR / "data"

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Turn-taking: minimum silence before a "finished" verdict from Smart Turn ends the
    # turn. Interview answers pause up to ~0.7 s between sentences (measured), so the
    # usual chatbot value of 0.2 s cuts candidates off mid-answer.
    turn_stop_secs: float = 0.8
    # When Smart Turn judges the answer unfinished, wait at most this long in silence.
    # In the 100-turn benchmark, 11% of turns hit this 3 s fallback (the p95 tail).
    turn_max_wait_secs: float = 3.0

    # WebRTC
    ice_servers: str = "stun:stun.l.google.com:19302"

    # Interview defaults
    default_num_questions: int = 5
    default_max_minutes: int = 12

    # Tracing
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.llm_model

    @property
    def ice_server_urls(self) -> list[str]:
        return [u.strip() for u in self.ice_servers.split(",") if u.strip()]

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


@lru_cache
def get_settings() -> Settings:
    return Settings()

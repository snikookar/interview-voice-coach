"""STT / LLM / TTS factory.

Every provider is chosen from settings, so switching between local and cloud mode
(for the latency comparison in the README) only needs an environment change.
"""

from loguru import logger
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from config import Settings

# Whisper drops disfluencies unless the decoder is primed with text that contains
# them. The live STT only needs to be accurate enough for the LLM, but keeping the
# fillers lets the interviewer react naturally ("take your time").
FILLER_PRIMING_PROMPT = "Um, so, uh, I think, like, you know, basically we, um, used it."


def create_stt(settings: Settings) -> STTService:
    if settings.stt_provider == "cloud":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        if not settings.deepgram_api_key:
            raise RuntimeError("STT_PROVIDER=cloud requires DEEPGRAM_API_KEY")
        logger.info("STT: Deepgram (cloud)")
        return DeepgramSTTService(
            api_key=settings.deepgram_api_key,
            settings=DeepgramSTTService.Settings(
                model="nova-3", language="en", extra={"filler_words": True}
            ),
        )

    from pipecat.services.whisper.stt import WhisperSTTService

    logger.info(f"STT: faster-whisper ({settings.whisper_model}, {settings.whisper_compute_type})")
    return WhisperSTTService(
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        settings=WhisperSTTService.Settings(
            model=settings.whisper_model,
            initial_prompt=FILLER_PRIMING_PROMPT,
        ),
    )


def create_llm(settings: Settings, system_instruction: str | None = None) -> LLMService:
    from pipecat.services.openai.llm import OpenAILLMService

    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
    logger.info(f"LLM: {settings.llm_model} @ {settings.llm_base_url}")
    return OpenAILLMService(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        settings=OpenAILLMService.Settings(
            model=settings.llm_model,
            temperature=0.6,
            # Interviewer turns are 2-3 sentences; a hard cap protects latency and cost.
            max_tokens=220,
            system_instruction=system_instruction,
        ),
    )


def create_tts(settings: Settings) -> TTSService:
    match settings.tts_provider:
        case "deepgram":
            from pipecat.services.deepgram.tts import DeepgramTTSService

            if not settings.deepgram_api_key:
                raise RuntimeError("TTS_PROVIDER=deepgram requires DEEPGRAM_API_KEY")
            logger.info(f"TTS: Deepgram Aura ({settings.deepgram_tts_voice})")
            return DeepgramTTSService(
                api_key=settings.deepgram_api_key, voice=settings.deepgram_tts_voice
            )
        case "piper":
            from pipecat.services.piper.tts import PiperTTSService

            logger.info(f"TTS: Piper ({settings.piper_voice})")
            return PiperTTSService(settings=PiperTTSService.Settings(voice=settings.piper_voice))
        case _:
            from pipecat.services.kokoro.tts import KokoroTTSService

            from download_models import ensure_kokoro

            model_path, voices_path = ensure_kokoro()
            logger.info(f"TTS: Kokoro ({settings.kokoro_voice})")
            return KokoroTTSService(
                model_path=str(model_path),
                voices_path=str(voices_path),
                settings=KokoroTTSService.Settings(voice=settings.kokoro_voice),
            )

"""Optional Langfuse tracing through Pipecat's built-in OpenTelemetry spans.

Langfuse accepts OTLP over HTTP, so no Langfuse SDK is needed: each conversation
turn becomes a trace with STT / LLM / TTS child spans and their timings.
"""

import base64
from functools import lru_cache

from loguru import logger

from config import get_settings


@lru_cache
def setup_tracing() -> bool:
    settings = get_settings()
    if not settings.tracing_enabled:
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from pipecat.utils.tracing.setup import setup_tracing as pipecat_setup_tracing
    except ImportError:
        logger.warning("Langfuse keys set but tracing extra missing: uv sync --extra tracing")
        return False

    token = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {token}"},
    )
    ok = pipecat_setup_tracing(service_name="interview-voice-coach", exporter=exporter)
    logger.info(f"Langfuse tracing {'enabled' if ok else 'failed to initialise'}")
    return ok

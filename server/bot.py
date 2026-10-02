"""Pipecat voice pipeline: mic -> VAD/turn -> STT -> LLM -> TTS -> speaker."""

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from config import get_settings
from metrics_observer import BargeInObserver, LatencyRecorder, TurnLatency
from services import create_llm, create_stt, create_tts
from tracing import setup_tracing

INTERVIEWER_PROMPT = (
    "You are Alex, a friendly but rigorous technical interviewer. This is a spoken "
    "conversation: your words are converted to audio, so never use lists, markdown, "
    "code or emojis. Keep every turn to at most two or three short sentences."
)


async def run_bot(connection: SmallWebRTCConnection) -> None:
    settings = get_settings()

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    stt = create_stt(settings)
    llm = create_llm(settings, system_instruction=INTERVIEWER_PROMPT)
    tts = create_tts(settings)

    context = LLMContext()
    # Silero decides "is someone speaking"; with a short stop_secs it reacts fast.
    # The default stop strategy (Smart Turn v3) then decides "is the answer finished",
    # so a thinking pause mid-answer does not hand the turn to the bot.
    # The default start strategy is VAD-based, so user speech interrupts the bot (barge-in).
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.2)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    async def on_turn_latency(turn: TurnLatency):
        # Live latency readout in the browser.
        await worker.queue_frame(
            RTVIServerMessageFrame(
                data={"type": "latency", "total_ms": turn.total_ms, **turn.stages_ms}
            )
        )

    latency = LatencyRecorder(on_turn=on_turn_latency)
    barge_in = BargeInObserver()

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency.observer, barge_in],
        enable_tracing=setup_tracing(),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": "Greet the candidate and ask them to introduce themselves.",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()

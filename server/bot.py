"""Pipecat voice pipeline: mic -> VAD/turn -> STT -> LLM (interview flow) -> TTS -> speaker."""

import time
from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.flows import FlowManager
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
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from config import get_settings
from db import SessionRow, session_scope
from flows import tools
from flows.interview_flow import create_intro_node
from flows.state import InterviewState
from metrics_observer import BargeInObserver, LatencyRecorder, TurnLatency
from services import create_llm, create_stt, create_tts
from session_recorder import SessionRecorder, mark_started
from tracing import setup_tracing
from turn_strategy import MinSilenceTurnStopStrategy

SessionEndCallback = Callable[[str], Awaitable[None]]


async def run_bot(
    connection: SmallWebRTCConnection,
    state: InterviewState,
    on_session_end: SessionEndCallback | None = None,
) -> None:
    settings = get_settings()

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    stt = create_stt(settings)
    llm = create_llm(settings)
    tts = create_tts(settings)

    context = LLMContext()
    # Silero (0.2 s) segments speech so Whisper transcribes sentence by sentence while
    # the candidate talks. Smart Turn v3 judges "is the answer finished", and the turn
    # ends only after TURN_STOP_SECS of silence (see turn_strategy.py).
    # The default start strategy is VAD-based, so user speech interrupts the bot (barge-in).
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.2)),
            user_turn_strategies=UserTurnStrategies(
                # VAD only: the default also starts a turn from a transcript, and the
                # final transcript (which ends the turn) then opened a phantom new
                # turn that cancelled the LLM request: +0.8 s on every turn.
                start=[VADUserTurnStartStrategy()],
                stop=[
                    MinSilenceTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(),
                        min_silence_secs=settings.turn_stop_secs,
                    )
                ],
            ),
        ),
    )

    recorder = SessionRecorder(state, settings.sessions_dir / state.session_id)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            recorder.audio,  # after output: records what was actually played
            context_aggregator.assistant(),
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
        observers=[latency, barge_in, recorder],
        enable_tracing=setup_tracing(),
        conversation_id=state.session_id,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=[tools.end_interview],
    )
    flow_manager.state["interview"] = state

    async def send_progress():
        await worker.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "progress",
                    "question": state.index + 1,
                    "total": len(state.plan),
                    "phase": state.phase,
                    "topic": state.current["topic"] if state.current else None,
                }
            )
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected, session={state.session_id}")
        state.started_at = time.monotonic()
        await recorder.start()
        async with session_scope() as db:
            row = await db.get(SessionRow, state.session_id)
            if row:
                mark_started(row)
                await db.commit()
        await flow_manager.initialize(create_intro_node(state))

    @context_aggregator.assistant().event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(_aggregator):
        await send_progress()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected, session={state.session_id}")
        await runner.cancel()

    try:
        await runner.run()
    finally:
        await recorder.save(latency.as_dicts(), barge_in.stop_times_ms)
        if on_session_end:
            await on_session_end(state.session_id)

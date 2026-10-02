"""FastAPI app: WebRTC signalling for the voice bot."""

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot
from config import get_settings

settings = get_settings()

webrtc_handler = SmallWebRTCRequestHandler(
    ice_servers=[IceServer(urls=url) for url in settings.ice_server_urls] or None
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await webrtc_handler.close()


app = FastAPI(title="Interview Voice Coach", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """SDP offer/answer exchange. The bot starts once the peer connection exists."""

    async def on_connection(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_bot, connection)

    return await webrtc_handler.handle_web_request(
        request=request, webrtc_connection_callback=on_connection
    )


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Trickle ICE: candidates the browser discovers after the initial offer."""
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "stt": settings.stt_provider, "tts": settings.tts_provider}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=7860)

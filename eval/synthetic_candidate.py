"""A synthetic candidate that takes a real interview over WebRTC, with no browser or mic.

It creates a session through the API, connects with aiortc exactly like the browser
does (POST /api/offer), speaks pre-synthesised answers (Kokoro TTS, a different voice
from the interviewer), and measures from the client side:

* voice-to-voice latency: end of the candidate's audio -> first audible bot audio
* barge-in: candidate starts talking over the bot -> bot audio goes silent

    uv run --project server python eval/synthetic_candidate.py --turns 6 --barge-in-every 3
"""

import argparse
import asyncio
import fractions
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import av
import httpx
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

SAMPLE_RATE = 48_000
FRAME = 960  # 20 ms
VOICE_RMS = 400  # int16 RMS above which the bot counts as speaking
BOT_DONE_SILENCE_S = 1.6  # CPU TTS can pause between sentences

INTRO = (
    "Hi, nice to meet you. I'm a software engineer with about four years of experience, "
    "mostly building Python back ends and, um, more recently retrieval augmented generation systems."
)
ANSWERS = [
    "Um, so I would start by building a small labeled dataset of real queries. Then I'd measure "
    "recall at k and precision at k, and, uh, also look at ranking metrics like MRR.",
    "I think the main trade-off is latency against quality. I'd stream the output, cache "
    "frequent requests, and use a smaller model where the evaluation shows it's good enough.",
    "So, basically, I would add monitoring first: latency, error rates and cost per request. "
    "Then I'd sample some outputs every day and have them reviewed.",
    "In my last project we had exactly that problem. I added tracing to every step, found the "
    "slow database query, and, you know, fixed the index. Latency dropped by half.",
    "I'm not completely sure, but I believe it would be something like, um, keeping the old "
    "version running and switching traffic over gradually.",
]
INTERRUPTION = "Sorry, could you repeat that please?"


def synthesise(texts: list[str], voice: str = "am_michael") -> list[np.ndarray]:
    """Kokoro -> int16 mono 48 kHz."""
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly

    from download_models import ensure_kokoro

    model, voices = ensure_kokoro()
    kokoro = Kokoro(str(model), str(voices))
    clips = []
    for text in texts:
        samples, sr = kokoro.create(text, voice=voice, speed=1.05, lang="en-us")
        audio = resample_poly(samples, SAMPLE_RATE, sr)
        clips.append((np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return clips


class ClipTrack(MediaStreamTrack):
    """Outgoing mic: silence, or a queued clip, paced in real time."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._buffer = np.zeros(0, dtype=np.int16)
        self._pts = 0
        self._start: float | None = None
        self.clip_ends_at = 0.0

    def play(self, clip: np.ndarray) -> float:
        """Queue a clip; returns the wall-clock time its last sample will be sent."""
        self._buffer = clip.copy()
        self.clip_ends_at = time.monotonic() + len(clip) / SAMPLE_RATE
        return self.clip_ends_at

    async def recv(self) -> av.AudioFrame:
        if self._start is None:
            self._start = time.monotonic()
        self._pts += FRAME
        delay = self._start + self._pts / SAMPLE_RATE - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        chunk, self._buffer = self._buffer[:FRAME], self._buffer[FRAME:]
        if len(chunk) < FRAME:
            chunk = np.concatenate([chunk, np.zeros(FRAME - len(chunk), dtype=np.int16)])
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        return frame


class BotListener:
    """Watches the incoming bot audio and keeps voice-activity timestamps."""

    def __init__(self):
        self.last_voice = 0.0
        self.voice_onsets: list[float] = []
        self._speaking = False
        self.recording: list[np.ndarray] = []
        self.timeline: list[tuple[float, bool]] = []  # (time, voiced) per frame
        self.closed = asyncio.Event()

    async def consume(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame = await track.recv()
                pcm = frame.to_ndarray().astype(np.float32)
                self.recording.append(pcm.flatten().astype(np.int16))
                rms = float(np.sqrt(np.mean(pcm**2))) if pcm.size else 0.0
                now = time.monotonic()
                self.timeline.append((now, rms > VOICE_RMS))
                if rms > VOICE_RMS:
                    if not self._speaking:
                        self.voice_onsets.append(now)
                    self._speaking = True
                    self.last_voice = now
                elif self._speaking and now - self.last_voice > 0.25:
                    self._speaking = False
        except Exception:
            self.closed.set()

    def silent_from(self, t: float, hold: float = 0.6) -> float | None:
        """First time >= t after which the bot stays silent for `hold` seconds.

        A pause between sentences is shorter than `hold`, so it doesn't count.
        """
        frames = [(ft, v) for ft, v in self.timeline if ft >= t]
        candidate = None
        for ft, voiced in frames:
            if voiced:
                candidate = None
            elif candidate is None:
                candidate = ft
            elif ft - candidate >= hold:
                return candidate
        return None

    def onset_after(self, t: float) -> float | None:
        return next((o for o in self.voice_onsets if o >= t), None)

    @property
    def speaking(self) -> bool:
        return self._speaking


@dataclass
class TurnResult:
    turn: int
    kind: str
    voice_to_voice_ms: float | None = None
    barge_in_stop_ms: float | None = None


@dataclass
class RunResult:
    session_id: str
    turns: list[TurnResult] = field(default_factory=list)
    ended_by_bot: bool = False


async def wait_bot_done(bot: BotListener, limit_secs: float = 40.0) -> bool:
    """Wait until the bot has spoken and then been silent for a while."""
    start = time.monotonic()
    while time.monotonic() - start < limit_secs:
        if bot.closed.is_set():
            return False
        if bot.last_voice and time.monotonic() - bot.last_voice > BOT_DONE_SILENCE_S:
            return True
        await asyncio.sleep(0.05)
    return False


async def run_interview(
    api: str,
    turns: int = 6,
    barge_in_every: int = 0,
    role: str = "ai-engineer",
    level: str = "mid",
    num_questions: int = 3,
) -> RunResult:
    clips = synthesise([INTRO, *ANSWERS, INTERRUPTION])
    intro, answers, interruption = clips[0], clips[1:-1], clips[-1]

    async with httpx.AsyncClient(base_url=api, timeout=60) as http:
        r = await http.post(
            "/api/sessions",
            json={"role": role, "level": level, "num_questions": num_questions, "max_minutes": 15},
        )
        r.raise_for_status()
        session_id = r.json()["id"]
        result = RunResult(session_id=session_id)

        pc = RTCPeerConnection()
        mic = ClipTrack()
        bot = BotListener()
        pc.addTrack(mic)
        pc.createDataChannel("chat")

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                asyncio.ensure_future(bot.consume(track))

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("closed", "failed"):
                bot.closed.set()

        await pc.setLocalDescription(await pc.createOffer())
        r = await http.post(
            "/api/offer",
            json={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "requestData": {"session_id": session_id},
            },
        )
        r.raise_for_status()
        answer = r.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        print(f"connected, session {session_id}")

        if not await wait_bot_done(bot):
            raise RuntimeError("bot never finished the greeting")

        for turn in range(turns):
            clip = intro if turn == 0 else answers[(turn - 1) % len(answers)]
            ended_at = mic.play(clip)
            await asyncio.sleep(max(0.0, ended_at - time.monotonic()))
            onset = None
            while time.monotonic() - ended_at < 20 and not bot.closed.is_set():
                onset = bot.onset_after(ended_at)
                if onset:
                    break
                await asyncio.sleep(0.02)
            tr = TurnResult(turn=turn, kind="intro" if turn == 0 else "answer")
            if onset:
                tr.voice_to_voice_ms = round((onset - ended_at) * 1000)
            print(f"turn {turn}: voice-to-voice {tr.voice_to_voice_ms} ms")

            if onset and barge_in_every and turn and turn % barge_in_every == 0:
                await asyncio.sleep(1.0)
                if bot.speaking:
                    started = time.monotonic()
                    mic.play(interruption)
                    await asyncio.sleep(2.0)
                    silent_at = bot.silent_from(started)
                    stop_ms = round((silent_at - started) * 1000) if silent_at else None
                    result.turns.append(
                        TurnResult(turn=turn, kind="barge_in", barge_in_stop_ms=stop_ms)
                    )
                    print(
                        f"turn {turn}: barge-in, bot silent {stop_ms} ms after interruption began"
                    )
            result.turns.append(tr)

            if not await wait_bot_done(bot):
                result.ended_by_bot = True
                break
            if bot.closed.is_set():
                result.ended_by_bot = True
                break

        await asyncio.sleep(2)
        await pc.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:7860")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--barge-in-every", type=int, default=0)
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "synthetic_run.json"))
    args = parser.parse_args()
    result = asyncio.run(
        run_interview(args.api, args.turns, args.barge_in_every, num_questions=args.questions)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

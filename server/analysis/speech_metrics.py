"""Speech delivery metrics, computed in code from word timestamps.

The live STT is tuned for latency and returns no word timings, so after the call the
candidate-only track (``user.wav``) is re-transcribed offline with a larger Whisper
model, ``word_timestamps=True`` and an ``initial_prompt`` full of disfluencies:
Whisper silently drops "um"/"uh" unless primed to keep them (spec section 6).

Everything below the transcription is pure functions over (word, start, end) lists,
so it is unit-tested without audio.
"""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

LONG_PAUSE_SECS = 3.0
FILLER_PRIMING_PROMPT = (
    "Um, so, uh, I think, like, you know, basically we, um, used it. Uh, I mean, like, yeah."
)
SINGLE_FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "basically"}
PHRASE_FILLERS = {("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")}


@dataclass
class Word:
    text: str
    start: float  # seconds since recording start
    end: float

    @property
    def norm(self) -> str:
        return re.sub(r"[^a-z']", "", self.text.lower())


@dataclass
class TurnWindow:
    """A user turn from the session transcript (ms since call start)."""

    start_ms: int
    end_ms: int
    question_id: str | None
    prev_bot_end_ms: int | None = None


@dataclass
class SpeechMetrics:
    words: int = 0
    speaking_secs: float = 0.0
    wpm: float | None = None
    filler_count: int = 0
    fillers_per_min: float | None = None
    filler_counts: dict[str, int] = field(default_factory=dict)
    long_pauses: int = 0
    longest_pause_secs: float = 0.0
    response_delays_secs: list[float] = field(default_factory=list)
    avg_response_delay_secs: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def count_fillers(words: list[Word]) -> dict[str, int]:
    """Count disfluencies.

    "like" is only a filler when it is set off by a pause or punctuation
    ("it was, like, slow") and not when it compares ("metrics like MRR"), so it
    is counted when the word itself carries a comma or follows a comma.
    """
    counts: dict[str, int] = {}
    norms = [w.norm for w in words]
    i = 0
    while i < len(words):
        pair = (norms[i], norms[i + 1]) if i + 1 < len(words) else None
        # "kind of"/"sort of" are only fillers when hedging ("it's, kind of, slow").
        hedge = pair in {("kind", "of"), ("sort", "of")}
        if pair in PHRASE_FILLERS and (not hedge or _set_off(words, i + 1)):
            key = " ".join(pair)
            counts[key] = counts.get(key, 0) + 1
            i += 2
            continue
        n = norms[i]
        if n in SINGLE_FILLERS or (n == "like" and _like_is_filler(words, i)):
            counts[n] = counts.get(n, 0) + 1
        i += 1
    return counts


def _set_off(words: list[Word], i: int) -> bool:
    return words[i].text.rstrip().endswith((",", "..."))


def _like_is_filler(words: list[Word], i: int) -> bool:
    prev_comma = i > 0 and words[i - 1].text.rstrip().endswith(",")
    return _set_off(words, i) or prev_comma


def words_in_window(words: list[Word], window: TurnWindow, slack_ms: int = 600) -> list[Word]:
    lo = (window.start_ms - slack_ms) / 1000
    hi = (window.end_ms + slack_ms) / 1000
    return [w for w in words if lo <= w.start <= hi]


def compute_metrics(words: list[Word], windows: list[TurnWindow]) -> SpeechMetrics:
    """Metrics over the given user turns (one answer, or the whole session)."""
    m = SpeechMetrics()
    pauses: list[float] = []
    all_words: list[Word] = []
    for win in windows:
        ws = words_in_window(words, win)
        if not ws:
            continue
        all_words.extend(ws)
        m.speaking_secs += max(0.0, ws[-1].end - ws[0].start)
        pauses.extend(b.start - a.end for a, b in zip(ws, ws[1:], strict=False))
        if win.prev_bot_end_ms is not None:
            m.response_delays_secs.append(
                round(max(0.0, ws[0].start - win.prev_bot_end_ms / 1000), 2)
            )

    m.words = len(all_words)
    m.filler_counts = count_fillers(all_words)
    m.filler_count = sum(m.filler_counts.values())
    long = [p for p in pauses if p > LONG_PAUSE_SECS]
    m.long_pauses = len(long)
    m.longest_pause_secs = round(max(pauses, default=0.0), 2)
    # Pauses longer than 3 s are thinking time, not speaking time.
    m.speaking_secs = round(m.speaking_secs - sum(long), 2)
    if m.speaking_secs >= 5:
        minutes = m.speaking_secs / 60
        m.wpm = round(m.words / minutes, 1)
        m.fillers_per_min = round(m.filler_count / minutes, 2)
    if m.response_delays_secs:
        m.avg_response_delay_secs = round(
            sum(m.response_delays_secs) / len(m.response_delays_secs), 2
        )
    return m


def talk_ratio(user_windows: list[TurnWindow], bot_spans_ms: list[tuple[int, int]]) -> float | None:
    """Candidate talk time / interviewer talk time (higher is better in an interview)."""
    user = sum(max(0, w.end_ms - w.start_ms) for w in user_windows)
    bot = sum(max(0, e - s) for s, e in bot_spans_ms)
    return round(user / bot, 2) if bot else None


def words_from_text(text: str, start_ms: int, end_ms: int) -> list[Word]:
    """Fallback without audio: spread the live transcript evenly over the turn."""
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, end_ms - start_ms) / 1000 / len(tokens)
    t0 = start_ms / 1000
    return [Word(tok, t0 + i * step, t0 + (i + 1) * step) for i, tok in enumerate(tokens)]


def transcribe_words(audio_path: Path, model_name: str) -> list[Word]:
    """Offline word-level transcription of the candidate track (CPU-heavy: run in a thread)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        initial_prompt=FILLER_PRIMING_PROMPT,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    return [Word(w.word.strip(), w.start, w.end) for seg in segments for w in seg.words or []]

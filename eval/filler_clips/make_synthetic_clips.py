"""Generate 20 synthetic clips with a known number of filler words.

    uv run --project server --extra eval python eval/filler_clips/make_synthetic_clips.py

Writes eval/filler_clips/synthetic_XX.wav and synthetic_labels.csv. These give a
reproducible *baseline* (TTS voices pronounce "um" cleanly and consistently).
The spec's real target is 20 clips of YOUR voice: record them, list them in
labels.csv with the true counts, and run filler_accuracy.py --labels labels.csv.
"""

import csv
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "server"))

from download_models import ensure_kokoro  # noqa: E402

VOICES = ["am_michael", "af_bella", "bm_george", "af_nicole"]
FILLERS = ("um", "uh", "like", "you know", "basically")

# (text, counts) - counts are what a careful human would mark.
SCRIPTS = [
    ("Um, so I would start with a baseline model.", {"um": 1}),
    ("I think, uh, the main issue was latency.", {"uh": 1}),
    ("We used, like, a really simple cache.", {"like": 1}),
    ("It's basically a key value store, you know, with a TTL.", {"basically": 1, "you know": 1}),
    ("Um, uh, let me think about that for a second.", {"um": 1, "uh": 1}),
    ("I'd measure recall at k and metrics like MRR.", {}),
    ("So, um, we split the data by time, and, um, trained on the past.", {"um": 2}),
    ("The service was, like, really slow, uh, under load.", {"like": 1, "uh": 1}),
    ("Basically, you know, it depends on the traffic.", {"basically": 1, "you know": 1}),
    ("I would add tracing and look at the slowest spans.", {}),
    ("Uh, I'm not sure, but I think it's eventual consistency.", {"uh": 1}),
    ("Um, so, basically we cached the embeddings.", {"um": 1, "basically": 1}),
    ("We had, uh, three replicas and, uh, one primary.", {"uh": 2}),
    ("It was, like, you know, a classic race condition.", {"like": 1, "you know": 1}),
    ("Things like retries and timeouts matter a lot.", {}),
    ("Um, I'd use a queue, and, uh, idempotent consumers.", {"um": 1, "uh": 1}),
    ("You know, the hardest part was the data quality.", {"you know": 1}),
    ("So basically, um, we rolled it out with a canary.", {"basically": 1, "um": 1}),
    ("I'd start with BM25 and add vector search later.", {}),
    ("Uh, um, sorry, could you repeat the question?", {"uh": 1, "um": 1}),
]


def main() -> None:
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly

    model, voices = ensure_kokoro()
    kokoro = Kokoro(str(model), str(voices))
    rows = []
    for i, (text, counts) in enumerate(SCRIPTS):
        samples, sr = kokoro.create(text, voice=VOICES[i % len(VOICES)], speed=1.0, lang="en-us")
        audio = (np.clip(resample_poly(samples, 16000, sr), -1, 1) * 32767).astype(np.int16)
        name = f"synthetic_{i + 1:02d}.wav"
        with wave.open(str(HERE / name), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())
        rows.append({"file": name, "text": text, **{f: counts.get(f, 0) for f in FILLERS}})
        print(name, counts)
    with (HERE / "synthetic_labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "text", *FILLERS])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

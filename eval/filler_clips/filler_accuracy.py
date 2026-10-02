"""Filler-word counting accuracy on labelled clips.

    uv run --project server --extra eval python eval/filler_clips/filler_accuracy.py \
        --labels eval/filler_clips/synthetic_labels.csv [--model small.en] [--no-prompt]

Each clip goes through the same offline path as a real session (Whisper with word
timestamps + the filler-priming prompt, then count_fillers). Reported per filler:
true count, detected count, and matched = sum over clips of min(true, detected),
from which precision = matched/detected and recall = matched/true.

--no-prompt runs the same clips WITHOUT the priming prompt, to show how much the
prompt matters (Whisper drops disfluencies by default).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "server"))

from analysis import speech_metrics  # noqa: E402

FILLERS = ("um", "uh", "like", "you know", "basically")
RESULTS = HERE.parent / "results"


def evaluate(labels: Path, model: str, prompt: bool) -> dict:
    if not prompt:
        speech_metrics.FILLER_PRIMING_PROMPT = None  # type: ignore[assignment]
    rows = list(csv.DictReader(labels.open(encoding="utf-8")))
    totals = {f: {"true": 0, "detected": 0, "matched": 0} for f in FILLERS}
    clips = []
    for row in rows:
        words = speech_metrics.transcribe_words(labels.parent / row["file"], model)
        detected = speech_metrics.count_fillers(words)
        clip = {"file": row["file"], "transcript": " ".join(w.text for w in words), "errors": {}}
        for f in FILLERS:
            t, d = int(row[f]), detected.get(f, 0)
            totals[f]["true"] += t
            totals[f]["detected"] += d
            totals[f]["matched"] += min(t, d)
            if t != d:
                clip["errors"][f] = {"true": t, "detected": d}
        clips.append(clip)

    def prf(t: dict) -> dict:
        p = t["matched"] / t["detected"] if t["detected"] else None
        r = t["matched"] / t["true"] if t["true"] else None
        return {
            **t,
            "precision": None if p is None else round(p, 3),
            "recall": None if r is None else round(r, 3),
        }

    overall = {k: sum(v[k] for v in totals.values()) for k in ("true", "detected", "matched")}
    return {
        "labels": labels.name,
        "model": model,
        "priming_prompt": prompt,
        "clips": len(rows),
        "per_filler": {f: prf(t) for f, t in totals.items()},
        "overall": prf(overall),
        "clips_detail": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=HERE / "labels.csv")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--no-prompt", action="store_true")
    args = parser.parse_args()

    result = evaluate(args.labels, args.model, prompt=not args.no_prompt)
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if not args.no_prompt else "_noprompt"
    out = RESULTS / f"filler_accuracy_{args.labels.stem}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\n{args.labels.name} · {args.model} · priming prompt: {not args.no_prompt}")
    print(f"{'filler':<12}{'true':>6}{'found':>7}{'precision':>11}{'recall':>8}")
    for f, v in {**result["per_filler"], "ALL": result["overall"]}.items():
        print(
            f"{f:<12}{v['true']:>6}{v['detected']:>7}{str(v['precision']):>11}{str(v['recall']):>8}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

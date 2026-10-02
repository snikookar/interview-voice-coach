# Filler-word counting accuracy

Whisper drops "um" and "uh" unless primed. These scripts measure how well the offline pipeline (Whisper `small.en` + word timestamps + priming prompt + `count_fillers`) counts them.

## Synthetic baseline (reproducible, no recording needed)
```bash
uv run --project server --extra eval python eval/filler_clips/make_synthetic_clips.py
uv run --project server --extra eval python eval/filler_clips/filler_accuracy.py --labels eval/filler_clips/synthetic_labels.csv
uv run --project server --extra eval python eval/filler_clips/filler_accuracy.py --labels eval/filler_clips/synthetic_labels.csv --no-prompt
```
20 Kokoro clips (4 voices) with known fillers, including **negative cases** ("metrics like MRR" must *not* count as a filler). The `--no-prompt` run shows what the priming prompt buys you.

## The real test: your voice (spec: 20 labelled clips)
1. Record 20 short clips (5–15 s each) answering interview questions naturally, as 16 kHz mono WAV, into this folder.
2. Listen back and write the true counts in `labels.csv` (same columns as `synthetic_labels.csv`: `file,text,um,uh,like,you know,basically`).
3. Run `filler_accuracy.py --labels eval/filler_clips/labels.csv` and report the result honestly, including misses.

WAV files are git-ignored. Commit `labels.csv` and the results JSON, not your voice.

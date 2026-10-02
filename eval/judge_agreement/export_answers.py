"""Export answers from finished sessions into a labelling file for human scoring.

    uv run --project server python eval/judge_agreement/export_answers.py

Appends new answers to eval/judge_agreement/labels.jsonl with empty "human" fields.
The judge's own scores are deliberately NOT exported, so labelling is blind.
Fill in each row's "human" object (scores 1-5 and the key points you think were
covered, copied from "key_points"), then run judge_agreement.py.
"""

import argparse
import json
from pathlib import Path

import httpx

LABELS = Path(__file__).with_name("labels.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:7860")
    args = parser.parse_args()

    existing = set()
    if LABELS.exists():
        existing = {
            json.loads(line)["id"]
            for line in LABELS.read_text(encoding="utf-8").splitlines()
            if line
        }

    sessions = httpx.get(f"{args.api}/api/sessions", timeout=30).json()
    added = 0
    with LABELS.open("a", encoding="utf-8") as f:
        for s in sessions:
            if s["status"] != "done":
                continue
            detail = httpx.get(f"{args.api}/api/sessions/{s['id']}", timeout=30).json()
            for a in (detail.get("report") or {}).get("answers", []):
                row_id = f"{s['id']}:{a['question_id']}"
                if row_id in existing or not a["answer_text"].strip():
                    continue
                row = {
                    "id": row_id,
                    "question_id": a["question_id"],
                    "type": a["type"],
                    "question": a["question"],
                    "key_points": a["key_points"],
                    "answer_text": a["answer_text"],
                    "follow_up_question": a["follow_up_question"],
                    "follow_up_answer": a["follow_up_answer"],
                    "human": {
                        "rater": "",
                        "correctness": None,
                        "depth": None,
                        "structure": None,
                        "communication": None,
                        "covered_points": [],
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                added += 1
    print(f"added {added} answers to {LABELS}")


if __name__ == "__main__":
    main()

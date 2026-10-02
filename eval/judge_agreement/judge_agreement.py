"""How well does the LLM judge agree with human ratings?

    uv run --project server --extra eval python eval/judge_agreement/judge_agreement.py
    uv run ... judge_agreement.py --labels eval/judge_agreement/labels.example.jsonl   # demo

For every labelled answer the judge scores it fresh (blind to the human labels), then:
* Spearman rho between judge and human, per dimension and for the mean score
  (target in the spec: >= 0.7)
* quadratic-weighted Cohen's kappa per dimension (agreement on the 1-5 scale itself)
* key-point detection: precision / recall / F1 of the judge's covered points
  against the human's, micro-averaged over all (answer, key point) pairs (target F1 >= 0.8)

Writes eval/results/judge_agreement.json (plot_results.py draws the scatter).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from analysis.judge import Judge  # noqa: E402

DIMS = ("correctness", "depth", "structure", "communication")
DEFAULT_LABELS = Path(__file__).with_name("labels.jsonl")
RESULTS = ROOT / "eval" / "results"


def load_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    labelled = [r for r in rows if all(r["human"].get(d) is not None for d in DIMS)]
    if len(labelled) < len(rows):
        print(f"skipping {len(rows) - len(labelled)} unlabelled rows")
    return labelled


async def score_all(rows: list[dict], judge: Judge) -> list[dict]:
    sem = asyncio.Semaphore(4)

    async def one(r: dict) -> dict:
        question = {k: r[k] for k in ("question_id", "type", "question", "key_points")}
        question["id"] = r["question_id"]
        text = r["answer_text"] + (
            f"\n{r['follow_up_answer']}" if r.get("follow_up_answer") else ""
        )
        async with sem:
            s = await judge.score(question, text, r.get("follow_up_question"))
        return s.model_dump()

    return await asyncio.gather(*(one(r) for r in rows))


def keypoint_prf(rows: list[dict], judged: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for r, j in zip(rows, judged, strict=True):
        human, model = set(r["human"]["covered_points"]), set(j["covered_points"])
        for kp in r["key_points"]:
            h, m = kp in human, kp in model
            tp += h and m
            fp += m and not h
            fn += h and not m
            tn += not h and not m
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
    }


def agreement(rows: list[dict], judged: list[dict]) -> dict:
    out: dict = {"n": len(rows), "dimensions": {}}
    for d in DIMS:
        h = [r["human"][d] for r in rows]
        m = [j[d] for j in judged]
        rho = spearmanr(h, m).statistic if len(set(h)) > 1 and len(set(m)) > 1 else float("nan")
        kappa = cohen_kappa_score(h, m, weights="quadratic", labels=[1, 2, 3, 4, 5])
        exact = sum(a == b for a, b in zip(h, m, strict=True)) / len(h)
        within1 = sum(abs(a - b) <= 1 for a, b in zip(h, m, strict=True)) / len(h)
        out["dimensions"][d] = {
            "spearman": round(float(rho), 3),
            "kappa_quadratic": round(float(kappa), 3),
            "exact": round(exact, 3),
            "within_1": round(within1, 3),
        }
    h_mean = [sum(r["human"][d] for d in DIMS) / 4 for r in rows]
    m_mean = [sum(j[d] for d in DIMS) / 4 for j in judged]
    out["mean_score_spearman"] = round(float(spearmanr(h_mean, m_mean).statistic), 3)
    out["key_points"] = keypoint_prf(rows, judged)
    out["points"] = [{"human": h, "judge": m} for h, m in zip(h_mean, m_mean, strict=True)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model", help="judge model override (default: JUDGE_MODEL / LLM_MODEL)")
    args = parser.parse_args()

    rows = load_rows(args.labels)
    if len(rows) < 3:
        sys.exit("Need at least 3 labelled answers (see eval/judge_agreement/README.md).")
    judge = Judge(model=args.model)
    judged = asyncio.run(score_all(rows, judge))
    result = {
        "judge_model": judge.model,
        "labels": str(args.labels.name),
        **agreement(rows, judged),
    }

    RESULTS.mkdir(exist_ok=True)
    name = (
        "judge_agreement_example.json" if "example" in args.labels.name else "judge_agreement.json"
    )
    (RESULTS / name).write_text(json.dumps(result, indent=2))

    print(f"\nJudge {judge.model} vs human on {result['n']} answers")
    print(f"{'dimension':<15}{'spearman':>10}{'kappa(q)':>10}{'exact':>8}{'±1':>8}")
    for d, v in result["dimensions"].items():
        print(
            f"{d:<15}{v['spearman']:>10}{v['kappa_quadratic']:>10}{v['exact']:>8}{v['within_1']:>8}"
        )
    print(f"mean score Spearman: {result['mean_score_spearman']}  (target >= 0.7)")
    kp = result["key_points"]
    print(f"key points: P {kp['precision']}  R {kp['recall']}  F1 {kp['f1']}  (target F1 >= 0.8)")


if __name__ == "__main__":
    main()

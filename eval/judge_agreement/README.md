# Judge agreement (LLM-as-judge vs human)

The spec target is **Spearman ≥ 0.7** between the judge's scores and human scores on 30 answers, and **key-point F1 ≥ 0.8**. This needs *your* labels: a judge validated against labels it wrote itself proves nothing.

## Workflow (about 1–2 hours)

1. **Collect answers.** Do 6–8 practice interviews in the app (5 questions each), or have a colleague do some. Mix good, mediocre and bad answers on purpose; agreement on a set of uniformly good answers is meaningless.
2. **Export them for blind labelling** (the judge's scores are *not* exported):
   ```bash
   uv run --project server python eval/judge_agreement/export_answers.py
   ```
   This appends rows to `labels.jsonl`.
3. **Label each row.** Fill the `human` object: `correctness`, `depth`, `structure`, `communication` (1–5, using the anchors in `server/analysis/judge.py`), and `covered_points` (copy the rubric strings you think the answer covered). Set `rater`. Ideally two people label, and you report human–human agreement too.
4. **Run the comparison:**
   ```bash
   uv run --project server --extra eval python eval/judge_agreement/judge_agreement.py
   uv run --project server --extra eval python eval/plot_results.py    # → docs/img/judge_vs_human.png
   ```

`labels.example.jsonl` holds **5 illustrative rows with made-up labels**, only so the pipeline can be demoed (`--labels eval/judge_agreement/labels.example.jsonl`). Never report numbers from it.

## What the script reports

| Metric | Why |
|---|---|
| Spearman ρ per dimension and for the mean | Rank agreement: does the judge order answers the way a human does? |
| Quadratic-weighted Cohen's κ | Agreement on the scale itself. Penalises a judge that is consistently 1 point harsher, which ρ ignores |
| exact / within ±1 | An intuitive sanity check |
| Key-point precision / recall / F1 | Is the "missed points" feedback trustworthy? |

If agreement is low, the levers in order of effectiveness are: sharper rubric key points, more few-shot examples (especially low-scoring ones), a stronger `JUDGE_MODEL`, and only then prompt wording.

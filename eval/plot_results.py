"""Charts for the README, from eval/results/*.json -> docs/img/*.png.

    uv run --project server --extra eval python eval/plot_results.py

* latency_by_stage.png: p50 per pipeline stage, one stacked bar per configuration
  (local vs cloud). Stages use a colour order validated for colour-vision deficiency;
  every segment carries its value as a label, so colour is never the only cue.
* judge_vs_human.png: judge mean score vs human mean score per answer, with y = x.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "img"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
STAGES = [
    ("turn", "Turn detection", "#2a78d6"),
    ("llm", "LLM", "#eb6834"),
    ("tts", "TTS", "#1baf7a"),
    ("output", "Output", "#eda100"),
]


def _style(ax, fig) -> None:
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def latency_chart() -> Path | None:
    files = sorted(RESULTS.glob("latency_*.json"))
    if not files:
        return None
    runs = [json.loads(f.read_text()) for f in files]
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.7 * len(runs)), dpi=150)
    _style(ax, fig)
    first = runs[0]["summary"]["server_stages"]
    for row, run in enumerate(runs):
        stages = run["summary"]["server_stages"]
        left = 0.0
        for key, label, color in STAGES:
            ms = (stages.get(key) or {}).get("p50_ms") or 0
            # Legend entries carry the values too, so thin segments are never colour-only.
            first_ms = (first.get(key) or {}).get("p50_ms") or 0
            ax.barh(
                row,
                ms,
                left=left,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                height=0.5,
                label=f"{label} {first_ms:.0f} ms" if row == 0 else None,
            )
            if ms >= 150:
                ax.text(
                    left + ms / 2, row, f"{ms:.0f}", ha="center", va="center", fontsize=8, color=INK
                )
            left += ms
        client = run["summary"]["client_voice_to_voice"]
        ax.text(
            left + 40,
            row,
            f"{left:.0f} ms\nclient p50 {client['p50_ms']:.0f} · p95 {client['p95_ms']:.0f}",
            va="center",
            fontsize=8,
            color=INK2,
        )
    labels = [r.get("display", r["label"]) for r in runs]
    ax.set_yticks(range(len(runs)), labels, color=INK, fontsize=9)
    ax.set_ylim(len(runs) - 0.5, -0.65)
    ax.set_xlabel("milliseconds (median per stage, server-side)", color=MUTED, fontsize=9)
    ax.set_xlim(0, max(2600, ax.get_xlim()[1] * 1.35))
    ax.axvline(1500, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1515, -0.45, "1.5 s target", color=MUTED, fontsize=8, va="center")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=4,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
    )
    ax.set_title(
        "Voice-to-voice latency by stage", loc="left", color=INK, fontsize=11, fontweight="bold"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "latency_by_stage.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def judge_chart() -> Path | None:
    f = RESULTS / "judge_agreement.json"
    if not f.exists():
        f = RESULTS / "judge_agreement_example.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    xs = [p["human"] for p in data["points"]]
    ys = [p["judge"] for p in data["points"]]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    _style(ax, fig)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot([1, 5], [1, 5], color=MUTED, linestyle="--", linewidth=1)
    ax.scatter(xs, ys, s=36, color="#2a78d6", edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xlabel("human mean score", color=MUTED, fontsize=9)
    ax.set_ylabel("judge mean score", color=MUTED, fontsize=9)
    example = " (EXAMPLE labels)" if "example" in f.name else ""
    ax.set_title(
        f"Judge vs human{example}\nSpearman {data['mean_score_spearman']} · n={data['n']}",
        loc="left",
        color=INK,
        fontsize=10,
        fontweight="bold",
    )
    path = OUT / ("judge_vs_human_example.png" if example else "judge_vs_human.png")
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (latency_chart(), judge_chart()):
        if p:
            print(f"wrote {p}")

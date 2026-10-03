"""Charts for the README, from eval/results/*.json -> docs/img/*.png.

    uv run --project server --extra eval python eval/plot_results.py

* latency_by_stage.png: p50 per pipeline stage, one stacked bar per configuration
  (local vs cloud). Stages use a colour order validated for colour-vision deficiency;
  every segment carries its value as a label, so colour is never the only cue.
* latency_distribution.png: histogram of client-measured voice-to-voice latency.
* latency_journey.png: latency after each fix found by the Phase 5 end-to-end harness.
* filler_accuracy.png: detected vs labelled filler words, per filler.
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


def latency_distribution_chart() -> Path | None:
    f = RESULTS / "latency_local-cpu.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    ms = data["raw"]["client_ms"]
    summary = data["summary"]["client_voice_to_voice"]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    _style(ax, fig)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    bins = range(1400, 5800, 100)
    ax.hist(ms, bins=bins, color="#2a78d6", edgecolor=SURFACE, linewidth=1)
    for value, label in ((summary["p50_ms"], "p50"), (summary["p95_ms"], "p95")):
        ax.axvline(value, color=INK2, linestyle="--", linewidth=1)
        ax.text(
            value + 40, ax.get_ylim()[1] * 0.92, f"{label} {value:.0f} ms", color=INK2, fontsize=8
        )
    slow = sum(1 for m in ms if m > 3000)
    ax.text(
        4750,
        ax.get_ylim()[1] * 0.45,
        f"{slow} of {len(ms)} turns > 3 s:\nSmart Turn waits up to 3 s\nwhen it thinks you're mid-thought",
        color=INK2,
        fontsize=8,
        ha="center",
    )
    ax.set_xlabel("client-measured voice-to-voice latency (ms)", color=MUTED, fontsize=9)
    ax.set_ylabel("turns", color=MUTED, fontsize=9)
    ax.set_title(
        f"Latency distribution · {len(ms)} turns, 17 interviews",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    path = OUT / "latency_distribution.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# Voice-to-voice latency after each fix found by the end-to-end harness.
# Source: the table in docs/phases/05-sessions-and-e2e.md; the last bar is the Phase 8 p50.
JOURNEY = [
    ("First\nE2E run", 8400),
    ("Turn stop\ngate", 3900),
    ("Piper TTS", 2500),
    ("VAD-only\nturn start", 1600),
    ("103-turn\nbench p50", 1810),
]


def latency_journey_chart() -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    _style(ax, fig)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    labels = [label for label, _ in JOURNEY]
    values = [ms / 1000 for _, ms in JOURNEY]
    colors = ["#c3c2b7"] * (len(JOURNEY) - 1) + ["#2a78d6"]
    bars = ax.bar(labels, values, color=colors, edgecolor=SURFACE, linewidth=2, width=0.6)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.15,
            f"{v:.1f} s",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.axhline(1.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(-0.5, len(JOURNEY) - 0.5 + 0.75)
    ax.text(len(JOURNEY) - 0.45, 1.5, "1.5 s\ntarget", color=MUTED, fontsize=8, va="center")
    ax.set_ylim(0, 9.5)
    ax.set_ylabel("seconds", color=MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=INK, labelsize=8)
    ax.set_title(
        "From 8.4 s to 1.8 s: what the end-to-end harness found",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    path = OUT / "latency_journey.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def filler_chart() -> Path | None:
    f = RESULTS / "filler_accuracy_synthetic_labels.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    per = data["per_filler"]
    names = list(per)
    fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
    _style(ax, fig)
    true = [per[n]["true"] for n in names]
    matched = [per[n]["matched"] for n in names]
    rows = range(len(names))
    ax.barh(rows, true, color="#e1e0d9", height=0.55, label="in the audio")
    ax.barh(rows, matched, color="#1baf7a", height=0.55, label="detected")
    for r, (t, m) in enumerate(zip(true, matched, strict=True)):
        ax.text(t + 0.15, r, f"{m}/{t}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(rows), [f'"{n}"' for n in names], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(true) + 1.5)
    o = data["overall"]
    ax.set_title(
        f"Filler-word detection · precision {o['precision']:.2f} · recall {o['recall']:.2f}",
        loc="left",
        color=INK,
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    path = OUT / "filler_accuracy.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    charts = (
        latency_chart(),
        latency_distribution_chart(),
        latency_journey_chart(),
        filler_chart(),
        judge_chart(),
    )
    for p in charts:
        if p:
            print(f"wrote {p}")

"""Figures for the interpretability analysis (Matplotlib, vector output)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6E6E6E"
CMAP = LinearSegmentedColormap.from_list("python_lean", [ORANGE, "#FFFFFF", BLUE])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False})


def export(fig, output_dir: Path, name: str, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_top_words(words: list[dict[str, Any]], output_dir: Path, name: str = "router_top_words",
                   k: int = 15, min_count: int = 10, min_examples: int = 5, width: float = 3.1,
                   height: float = 3.4) -> list[Path]:
    """Words with the most positive (Lean) and most negative (Python) mean attribution."""
    eligible = [w for w in words if w["count"] >= min_count and w["examples"] >= min_examples]
    lean = sorted(eligible, key=lambda w: -w["mean_attribution"])[:k]
    python = sorted(eligible, key=lambda w: w["mean_attribution"])[:k]
    fig, axes = plt.subplots(1, 2, figsize=(width * 2, height), sharex=False)
    for ax, rows, color, title in ((axes[0], python, ORANGE, "Towards Python"), (axes[1], lean, BLUE, "Towards Lean")):
        rows = list(reversed(rows))
        y = np.arange(len(rows))
        ax.barh(y, [100 * w["mean_attribution"] for w in rows], xerr=[100 * w["sem"] for w in rows], color=color,
                height=0.7, error_kw=dict(lw=0.6, capsize=1.5, ecolor=GREY))
        ax.set_yticks(y)
        ax.set_yticklabels([f"{w['word']}  ({w['count']})" for w in rows], fontsize=7)
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(axis="x", color="#E5E5E5", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("Mean attribution to the routing margin (×10$^{-2}$ logits)", fontsize=8)
    fig.tight_layout()
    return export(fig, output_dir, name)


def plot_categories(categories: list[dict[str, Any]], output_dir: Path, name: str = "router_categories",
                    width: float = 3.1, height: float = 2.0) -> list[Path]:
    rows = sorted(categories, key=lambda c: c["mean_attribution"])
    fig, ax = plt.subplots(figsize=(width, height))
    y = np.arange(len(rows))
    values = [100 * c["mean_attribution"] for c in rows]
    ax.barh(y, values, xerr=[100 * c["sem"] for c in rows], color=[BLUE if v >= 0 else ORANGE for v in values],
            height=0.65, error_kw=dict(lw=0.6, capsize=1.5, ecolor=GREY))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c['category']}  ({100 * c['share_of_tokens']:.0f}%)" for c in rows], fontsize=7)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlabel("Mean attribution (×10$^{-2}$ logits)", fontsize=8)
    ax.grid(axis="x", color="#E5E5E5", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return export(fig, output_dir, name)


def _layout_words(ax, words: Sequence[dict[str, Any]], norm, y: float, fontsize: float = 7.5,
                  x_start: float = 0.0, max_width: float = 1.0, line_height: float | None = None) -> float:
    """Render colour-boxed words with wrapping in axes coordinates; return the final y."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_width, ax_height = ax.get_window_extent(renderer).size
    gap = fontsize * fig.dpi / 72 * 0.45 / ax_width
    line = line_height or fontsize * fig.dpi / 72 * 1.9 / ax_height
    x = x_start
    for w in words:
        text = ax.text(x, y, w["text"], fontsize=fontsize, va="top", transform=ax.transAxes)
        tw = text.get_window_extent(renderer).width / ax_width
        if x + tw > max_width and x > x_start:
            x = x_start
            y -= line
            text.set_position((x, y))
        color = CMAP(norm(w["attribution"]))
        text.set_bbox(dict(facecolor=color, edgecolor="none", pad=1.4))
        text.set_color("white" if np.mean(color[:3]) < 0.45 else "#151515")
        x += tw + gap
    return y - line


def plot_word_heatmaps(cases: Sequence[dict[str, Any]], output_dir: Path, name: str = "router_heatmaps",
                       width: float = 6.3, height_per_case: float = 0.95, limit: float | None = None) -> list[Path]:
    """Signed word attributions of a few steps on one shared colour scale.

    Each case has ``words`` (``text``, ``attribution``), ``title`` and ``subtitle``.
    """
    values = [abs(w["attribution"]) for c in cases for w in c["words"]]
    limit = limit or float(np.percentile(values, 98))
    norm = Normalize(-limit, limit, clip=True)
    fig = plt.figure(figsize=(width, height_per_case * len(cases) + 0.5))
    ax = fig.add_axes([0.01, 0.12, 0.98, 0.86])
    ax.axis("off")
    y = 1.0
    line = 7.5 * fig.dpi / 72 * 1.9 / ax.get_window_extent(fig.canvas.get_renderer()).size[1]
    for case in cases:
        ax.text(0, y, case["title"], fontsize=8, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(1, y, case["subtitle"], fontsize=8, va="top", ha="right", transform=ax.transAxes)
        y -= line
        y = _layout_words(ax, case["words"], norm, y, line_height=line) - 0.45 * line
    cax = fig.add_axes([0.3, 0.04, 0.4, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-limit, 0, limit])
    cb.set_ticklabels([f"−{limit:.2f}", "0", f"+{limit:.2f}"])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label("Attribution to the routing margin (logits): orange → Python · blue → Lean", fontsize=7.5)
    return export(fig, output_dir, name)

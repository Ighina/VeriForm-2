#!/usr/bin/env python3
"""Figures and LaTeX table for the Integrated Gradients analysis of the BERT router."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _paths  # noqa: F401
from veriform2.interpretability.corpus import load_results
from veriform2.interpretability.plots import plot_categories, plot_top_words, plot_word_heatmaps
from veriform2.interpretability.words import CATEGORIES


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in ("word", "category"):
                continue
            try:
                row[key] = int(value)
            except ValueError:
                row[key] = float(value)
    return rows


def excerpt(result: dict, max_words: int = 42) -> list[dict]:
    """Keep the window of ``max_words`` words with the largest absolute attribution mass."""
    words = result["words"]
    if len(words) <= max_words:
        return words
    best, best_mass = 0, -1.0
    for start in range(0, len(words) - max_words + 1):
        mass = sum(abs(w["attribution"]) for w in words[start:start + max_words])
        if mass > best_mass:
            best, best_mass = start, mass
    chosen = words[best:best + max_words]
    prefix = [dict(text="…", attribution=0.0)] if best > 0 else []
    suffix = [dict(text="…", attribution=0.0)] if best + max_words < len(words) else []
    return prefix + chosen + suffix


def category_table(categories: list[dict]) -> str:
    lines = [r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\hline",
             r"\textbf{Category} & \textbf{Tokens} & \textbf{Mean attr.} & \textbf{Share $|$attr.$|$} \\", r"\hline"]
    for c in sorted(categories, key=lambda c: -c["mean_attribution"]):
        lines.append(f"{c['category']} & {100 * c['share_of_tokens']:.1f}\\% & "
                     f"{100 * c['mean_attribution']:+.2f} & {100 * c['share_of_absolute_attribution']:.1f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{Mean Integrated Gradients attribution to the routing margin (in $10^{-2}$ logits; positive values push towards Lean) and share of the total absolute attribution, by lexical category of the input words over the whole test split.}",
              r"\label{tab:ig_categories}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/interpretability"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-examples", type=int, default=5)
    args = parser.parse_args()

    words = read_csv(args.input_dir / "word_attributions.csv")
    categories = read_csv(args.input_dir / "category_attributions.csv")
    results = load_results(args.input_dir / "attributions.jsonl")
    paths = plot_top_words(words, args.output_dir, k=args.k, min_count=args.min_count, min_examples=args.min_examples)
    paths += plot_categories(categories, args.output_dir)

    # Heatmaps: the step with the largest margin (most Lean-like) and the smallest (most Python-like),
    # plus the Lean-labelled step with the largest margin among those the router sent to Lean.
    by_margin = sorted(results, key=lambda r: r["margin"])
    cases = [by_margin[-1], by_margin[0]]
    lean_hits = [r for r in by_margin if r.get("label") == 1 and r["margin"] >= 0]
    if lean_hits and lean_hits[-1] is not cases[0]:
        cases.insert(1, lean_hits[-1])
    labels = {0: "Python", 1: "Lean"}
    heat = []
    for i, r in enumerate(cases, 1):
        heat.append(dict(words=excerpt(r), title=f"E{i}  {r['example_id']} / step {r['step_index']}",
                         subtitle=f"gold: {labels.get(r.get('label'), '?')}   P(Lean) = {100 * r['p_lean']:.1f}%   "
                                  f"margin = {r['margin']:+.2f}"))
    paths += plot_word_heatmaps(heat, args.output_dir)
    (args.output_dir.parent / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir.parent / "tables" / "ig_categories.tex").write_text(category_table(categories), encoding="utf-8")
    manifest = dict(cases=[dict(example_id=r["example_id"], step_index=r["step_index"], label=r.get("label"),
                                p_lean=r["p_lean"], margin=r["margin"]) for r in cases],
                    categories={c["category"]: c for c in categories if c["category"] in CATEGORIES})
    (args.output_dir / "interpretability_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

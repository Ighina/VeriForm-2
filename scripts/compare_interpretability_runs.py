#!/usr/bin/env python3
"""Stability of the corpus-level Integrated Gradients analysis across retrained routers.

Compares two ``interpret_corpus.py`` output directories: Spearman correlation of the
per-word mean attributions (words frequent in both runs), overlap of the top-k word
lists in each direction, and the category means side by side.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _paths  # noqa: F401
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


def top(words: list[dict], k: int, lean: bool, min_count: int, min_examples: int) -> list[str]:
    eligible = [w for w in words if w["count"] >= min_count and w["examples"] >= min_examples]
    eligible.sort(key=lambda w: -w["mean_attribution"] if lean else w["mean_attribution"])
    return [w["word"] for w in eligible[:k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-examples", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON summary")
    args = parser.parse_args()
    from scipy.stats import spearmanr

    words_a = {w["word"]: w for w in read_csv(args.run_a / "word_attributions.csv")}
    words_b = {w["word"]: w for w in read_csv(args.run_b / "word_attributions.csv")}
    common = [w for w in words_a if w in words_b and words_a[w]["count"] >= args.min_count
              and words_a[w]["examples"] >= args.min_examples]
    rho = spearmanr([words_a[w]["mean_attribution"] for w in common],
                    [words_b[w]["mean_attribution"] for w in common]).correlation
    summary = {"common_words": len(common), "spearman_mean_attribution": rho, "top_overlap": {}, "categories": {}}
    for lean in (True, False):
        a = top(list(words_a.values()), args.k, lean, args.min_count, args.min_examples)
        b = top(list(words_b.values()), args.k, lean, args.min_count, args.min_examples)
        name = "towards_lean" if lean else "towards_python"
        summary["top_overlap"][name] = {"overlap": len(set(a) & set(b)), "k": args.k, "run_a": a, "run_b": b}
    cats_a = {c["category"]: c for c in read_csv(args.run_a / "category_attributions.csv")}
    cats_b = {c["category"]: c for c in read_csv(args.run_b / "category_attributions.csv")}
    for c in CATEGORIES:
        summary["categories"][c] = {"run_a": cats_a[c]["mean_attribution"], "run_b": cats_b[c]["mean_attribution"]}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

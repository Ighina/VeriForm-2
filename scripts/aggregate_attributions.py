#!/usr/bin/env python3
"""Recompute the word- and category-level aggregates of an ``interpret_corpus.py`` run.

Re-derives each word's key and lexical category from the saved text (so that
changes to ``veriform2.interpretability.words`` apply without re-running the
attribution) and rewrites ``word_attributions.csv``, ``category_attributions.csv``
and the ``categories`` entry of ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
from veriform2.interpretability.corpus import aggregate_categories, aggregate_words, load_results, write_csv
from veriform2.interpretability.words import categorize, normalize_word


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="+")
    args = parser.parse_args()
    for run_dir in args.run_dir:
        results = load_results(run_dir / "attributions.jsonl")
        for r in results:
            for w in r["words"]:
                w["key"] = normalize_word(w["text"])
                w["category"] = categorize(w["key"])
        with (run_dir / "attributions.jsonl").open("w", encoding="utf-8") as handle:
            for r in results:
                handle.write(json.dumps(r) + "\n")
        categories = aggregate_categories(results)
        write_csv(run_dir / "word_attributions.csv", aggregate_words(results))
        write_csv(run_dir / "category_attributions.csv", categories)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            report["categories"] = categories
            summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(run_dir)
        for c in categories:
            print(f"  {c['category']:20s} n={c['count']:6d} mean={100 * c['mean_attribution']:+.3f}e-2 "
                  f"share|attr|={100 * c['share_of_absolute_attribution']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

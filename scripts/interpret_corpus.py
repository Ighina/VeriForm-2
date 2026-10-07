#!/usr/bin/env python3
"""Attribute the BERT router's routing margin on every test step with Integrated Gradients.

Writes ``attributions.jsonl`` (per step: tokens, raw attributions, merged words),
``word_attributions.csv`` and ``category_attributions.csv`` (corpus aggregates)
and ``summary.json``.  Figures are produced separately by
``scripts/make_interpretability_figures.py``.
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import _paths  # noqa: F401
from veriform2.evaluation.loaders import read_rows
from veriform2.interpretability.corpus import (aggregate_categories, aggregate_words, attribute_rows, summary,
                                                write_csv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Fine-tuned BERT router directory")
    parser.add_argument("--rows", type=Path, required=True, help="JSONL rows with text (e.g. test_probabilities.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpretability"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--internal-batch-size", type=int, default=25)
    parser.add_argument("--delta-tolerance", type=float, default=0.02)
    parser.add_argument("--limit", type=int, help="Only the first N rows (smoke tests)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory")

    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification

    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = BertForSequenceClassification.from_pretrained(str(args.model)).to(args.device).eval()
    rows = read_rows(args.rows)[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output_dir / "attributions.jsonl").open("w", encoding="utf-8") as handle:
        for result in attribute_rows(model, tokenizer, rows, n_steps=args.n_steps,
                                     internal_batch_size=args.internal_batch_size,
                                     delta_tolerance=args.delta_tolerance, max_steps=args.max_steps):
            handle.write(json.dumps(result) + "\n")
            results.append(result)
    words = aggregate_words(results)
    categories = aggregate_categories(results)
    write_csv(args.output_dir / "word_attributions.csv", words)
    write_csv(args.output_dir / "category_attributions.csv", categories)
    report = summary(results, categories)
    report["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report["versions"] = {p: version(p) for p in ("torch", "transformers", "captum", "numpy")}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "categories"}, indent=2))
    for c in categories:
        print(f"{c['category']:20s} n={c['count']:6d} mean={100 * c['mean_attribution']:+.3f}e-2 "
              f"share|attr|={100 * c['share_of_absolute_attribution']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Explain individual routing decisions with Integrated Gradients (HTML token heatmaps)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
from veriform2.evaluation.loaders import read_rows
from veriform2.interpretability.integrated_gradients import TARGETS, explain, render_html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--step", help="A single step text")
    source.add_argument("--input-jsonl", type=Path, help="Rows with a text field (and optional metadata)")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, default="margin")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/interpretations"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory")
    from transformers import AutoTokenizer, BertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = BertForSequenceClassification.from_pretrained(str(args.model)).to(args.device).eval()
    rows = [{"text": args.step}] if args.step is not None else read_rows(args.input_jsonl)[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output_dir / "attributions.jsonl").open("w") as handle:
        for row in rows:
            result = explain(model, tokenizer, row["text"], target=args.target, n_steps=args.n_steps,
                             max_steps=args.max_steps)
            result.update({k: row[k] for k in ("example_id", "step_index", "dataset") if k in row})
            if "label" in row:
                result["gold_label"] = {0: "Python", 1: "Lean"}[int(row["label"])]
            handle.write(json.dumps(result) + "\n")
            results.append(result)
            print(f"{len(results)}/{len(rows)} P(Lean)={result['p_lean']:.4f} residual={result['convergence_delta']:.3g}")
    (args.output_dir / "report.html").write_text(render_html(results), encoding="utf-8")
    print(args.output_dir / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

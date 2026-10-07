#!/usr/bin/env python3
"""Local word interventions on the test steps the BERT router is most inclined to send to Lean."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
from veriform2.evaluation.loaders import read_rows
from veriform2.interpretability.integrated_gradients import render_html
from veriform2.interpretability.lean_triggers import analyze, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="test_probabilities.jsonl")
    parser.add_argument("--min-probability", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lean_triggers"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory")
    from transformers import AutoTokenizer, BertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = BertForSequenceClassification.from_pretrained(str(args.model)).to(args.device).eval()
    rows = sorted(read_rows(args.predictions), key=lambda r: -float(r["p_lean"]))
    rows = [r for r in rows if float(r["p_lean"]) >= args.min_probability]
    if not rows:
        parser.error("No predictions meet the probability cutoff")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        results.append(analyze(model, tokenizer, row, n_steps=args.n_steps, max_steps=args.max_steps))
        print(f"{len(results)}/{len(rows)} {row['example_id']} residual={results[-1]['convergence_delta']:.3g}")
    (args.output_dir / "attributions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (args.output_dir / "report.html").write_text(render_html(results), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(summarize(results), encoding="utf-8")
    print(args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

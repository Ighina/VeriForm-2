#!/usr/bin/env python3
"""Score verifiers and routers on the BERT test split and export the paper tables.

Example (from the repository root, using the compact result files in results/):

    python scripts/evaluate_routers.py \
        --python-results results/verifiers/python_step_outcomes.csv \
        --lean-results results/verifiers/lean_step_outcomes.csv \
        --bert-probabilities results/bert_router/test_probabilities.jsonl \
        --threshold-file results/bert_router/threshold.json \
        --llm "Qwen3.5-9B=results/llm_router/qwen3.5-9b_choices.csv" \
        --llm "GPT-5.6 Luna=results/llm_router/gpt-5.6-luna_choices.csv" \
        --output-dir results/router_comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
from veriform2.data import DEFAULT_DATASET, load_step_labels
from veriform2.evaluation.compare import (BERT_DEFAULT, BERT_TUNED, LEAN_ONLY, ORACLE, PYTHON_ONLY,
                                          build_comparison, latex_table, print_summary, write_report)
from veriform2.evaluation.loaders import (load_bert_probabilities, load_lean_predictions,
                                          load_python_predictions, load_router_choices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--python-results", type=Path, required=True)
    parser.add_argument("--lean-results", type=Path, required=True)
    parser.add_argument("--bert-probabilities", type=Path, required=True,
                        help="test_probabilities.jsonl written by the threshold tuning script")
    parser.add_argument("--threshold-file", type=Path, help="threshold.json (adds the tuned-threshold row)")
    parser.add_argument("--llm", action="append", default=[], metavar="NAME=PATH",
                        help="LLM router classifications (jsonl or csv); repeatable")
    parser.add_argument("--extra-bert", action="append", default=[], metavar="LABEL=PROBABILITIES[=THRESHOLD_FILE]",
                        help="Additional BERT router(s) on the same test split; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("results/router_comparison"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm = {}
    for item in args.llm:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--llm expects NAME=PATH, got {item!r}")
        llm[name] = load_router_choices(Path(path))
    tuned = None
    if args.threshold_file:
        tuned = float(json.loads(args.threshold_file.read_text())["threshold"])
    extra = {}
    for item in args.extra_bert:
        parts = item.split("=")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--extra-bert expects LABEL=PROBABILITIES[=THRESHOLD_FILE], got {item!r}")
        threshold = float(json.loads(Path(parts[2]).read_text())["threshold"]) if len(parts) == 3 else None
        extra[parts[0]] = (load_bert_probabilities(Path(parts[1])), threshold)
    report = build_comparison(
        truth=load_step_labels(args.dataset),
        python=load_python_predictions(args.python_results),
        lean=load_lean_predictions(args.lean_results),
        bert_rows=load_bert_probabilities(args.bert_probabilities),
        llm_choices=llm, tuned_threshold=tuned, extra_bert=extra,
    )
    write_report(report, args.output_dir / "metrics.json")
    names = {PYTHON_ONLY: "Python only", LEAN_ONLY: "Lean only", ORACLE: "Oracle routing (upper bound)"}
    for name in llm:
        names[name] = rf"$\mathcal{{R}}_{{LLM}}$, {name}"
    names[BERT_DEFAULT] = r"$\mathcal{R}_{BERT}$"
    names[BERT_TUNED] = r"$\mathcal{R}_{BERT}$, tuned threshold"
    suffix = BERT_TUNED[len(BERT_DEFAULT):]
    for label in extra:
        names[label] = rf"$\mathcal{{R}}_{{BERT}}$ ({label})"
        if label + suffix in report["matched_test"]:
            names[label + suffix] = rf"$\mathcal{{R}}_{{BERT}}$ ({label}), tuned threshold"
    bold = [n for n in names if n not in (ORACLE,)]
    n = report["matched_steps"]
    tables = {
        "router_test_split_balanced_accuracy.tex": latex_table(
            report, "balanced_accuracy", names,
            caption=(f"Balanced step-verification accuracy (\\%) on the {n:,} held-out test steps shared by all "
                     "methods. Average is the unweighted mean over the four datasets; best non-oracle value per "
                     "column in bold."),
            label="tab:test_split_results", bold_rows=bold),
        "router_test_split_accuracy.tex": latex_table(
            report, "accuracy", names,
            caption=(f"Step-verification accuracy (\\%) on the same {n:,} held-out test steps as "
                     "Table~\\ref{tab:test_split_results}."),
            label="tab:test_split_accuracy", bold_rows=bold),
    }
    for filename, text in tables.items():
        (args.output_dir / filename).write_text(text, encoding="utf-8")
    print_summary(report)
    print(f"\nWrote {args.output_dir / 'metrics.json'} and {', '.join(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

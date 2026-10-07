#!/usr/bin/env python3
"""Export compact, repository-sized copies of the raw experiment outputs.

The raw pipeline outputs (prompts, generated code, LLM rationales, Lean
diagnostics) amount to hundreds of megabytes.  This script keeps only what the
evaluation and interpretability scripts need:

results/verifiers/python_step_outcomes.csv   example_id, step_index, outcome
results/verifiers/lean_step_outcomes.csv     example_id, step_index, formalization_valid,
                                             alignment_status, proof_status, semantic_outcome
results/llm_router/<name>_choices.csv        example_id, step_index, choice, error
results/bert_router/                         data report, test metrics, threshold, probabilities
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import _paths  # noqa: F401
from veriform2.evaluation.loaders import read_rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({f: row.get(f, "") for f in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-results", type=Path, required=True, help="step_results.jsonl")
    parser.add_argument("--lean-results", type=Path, required=True, help="Lean step_outcomes.csv")
    parser.add_argument("--llm", action="append", default=[], metavar="NAME=PATH", help="classifications.jsonl")
    parser.add_argument("--bert-run", type=Path, help="BERT training directory (with threshold_tuning/)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = [r for r in read_rows(args.python_results) if r.get("representation", "linear") == "linear"]
    write_csv(args.output_dir / "verifiers/python_step_outcomes.csv",
              [dict(example_id=r["example_id"], step_index=r["step_index"], outcome=r["outcome"]) for r in rows],
              ["example_id", "step_index", "outcome"])
    print(f"Python verifier: {len(rows)} steps")

    lean = []
    for r in read_rows(args.lean_results):
        lean.append(dict(example_id=r["example_id"], step_index=int(r["step_id"].rsplit("_", 1)[1]),
                         formalization_valid=r["formalization_valid"], alignment_status=r["alignment_status"],
                         proof_status=r["proof_status"], semantic_outcome=r["semantic_outcome"]))
    write_csv(args.output_dir / "verifiers/lean_step_outcomes.csv", lean,
              ["example_id", "step_index", "formalization_valid", "alignment_status", "proof_status", "semantic_outcome"])
    print(f"Lean verifier: {len(lean)} steps")

    for item in args.llm:
        name, _, path = item.partition("=")
        choices = [dict(example_id=r["example_id"], step_index=r["step_index"], choice=r.get("choice") or "",
                        error="" if r.get("error") is None else "request_failed") for r in read_rows(Path(path))]
        write_csv(args.output_dir / f"llm_router/{name}_choices.csv", choices,
                  ["example_id", "step_index", "choice", "error"])
        print(f"LLM router {name}: {len(choices)} steps")

    if args.bert_run:
        target = args.output_dir / "bert_router"
        target.mkdir(parents=True, exist_ok=True)
        for source, dest in (("data_report.json", "data_report.json"), ("test_metrics.json", "test_metrics.json"),
                             ("threshold_tuning/threshold.json", "threshold.json"),
                             ("threshold_tuning/metrics.json", "threshold_tuning_metrics.json"),
                             ("threshold_tuning/validation_threshold_curve.json", "validation_threshold_curve.json"),
                             ("threshold_tuning/test_probabilities.jsonl", "test_probabilities.jsonl"),
                             ("threshold_tuning/validation_probabilities.jsonl", "validation_probabilities.jsonl")):
            if (args.bert_run / source).is_file():
                shutil.copyfile(args.bert_run / source, target / dest)
        print(f"BERT router artefacts copied to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

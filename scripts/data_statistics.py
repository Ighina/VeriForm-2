#!/usr/bin/env python3
"""Dataset, verifier-outcome and router-choice statistics for the appendix (JSON + LaTeX)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _paths  # noqa: F401
from veriform2 import DATASET_NAMES, DATASETS
from veriform2.data import DEFAULT_DATASET, dataset_of, load_examples, step_labels_for
from veriform2.evaluation.loaders import read_rows

PY_OUTCOMES = ("True", "False", "Autoformalisation failure", "Prover failure")
LEAN_OUTCOMES = ("True", "Autoformalisation failure", "Prover failure")
CHOICES = ("python", "lean", "tie", "neither", "inconclusive", "unresolved")


def outcome_rows(path: Path, field_candidates: tuple[str, ...]) -> dict[tuple[str, int], str]:
    result = {}
    for row in read_rows(path):
        value = next((row[f] for f in field_candidates if row.get(f) not in (None, "")), "")
        result[(str(row["example_id"]), int(row["step_index"]))] = value
    return result


def pct(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}" if total else "--"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--python-results", type=Path, default=Path("results/verifiers/python_step_outcomes.csv"))
    parser.add_argument("--lean-results", type=Path, default=Path("results/verifiers/lean_step_outcomes.csv"))
    parser.add_argument("--llm", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--data-report", type=Path, default=Path("results/bert_router/data_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.dataset)
    labels: dict[tuple[str, int], bool] = {}
    per_dataset = {d: Counter() for d in DATASETS}
    for eid, row in examples.items():
        d = dataset_of(eid)
        steps = row.get("steps") or []
        per_dataset[d]["solutions"] += 1
        per_dataset[d]["steps"] += len(steps)
        per_dataset[d]["correct_solutions"] += int(row["label"]) == -1
        expanded = step_labels_for(eid, int(row["label"]), len(steps))
        labels.update(expanded)
        per_dataset[d]["annotated_steps"] += len(expanded)
        per_dataset[d]["annotated_correct"] += sum(expanded.values())

    python = outcome_rows(args.python_results, ("outcome", "sandbox_outcome"))
    lean = outcome_rows(args.lean_results, ("semantic_outcome", "result"))
    verifier = {d: {"python": Counter(), "lean": Counter(), "both": 0} for d in DATASETS}
    for key, _ in labels.items():
        d = dataset_of(key[0])
        if key in python:
            verifier[d]["python"][python[key]] += 1
        if key in lean:
            verifier[d]["lean"][lean[key]] += 1
        verifier[d]["both"] += key in python and key in lean

    routers = {}
    for item in args.llm:
        name, _, path = item.partition("=")
        choices = outcome_rows(Path(path), ("choice",))
        counts = {d: Counter() for d in DATASETS}
        for key in labels:
            if key in choices:
                counts[dataset_of(key[0])][choices[key] or "unresolved"] += 1
        routers[name] = counts

    report = json.loads(args.data_report.read_text()) if args.data_report.is_file() else None
    summary = {
        "processbench": {d: dict(per_dataset[d]) for d in DATASETS},
        "verifier_outcomes": {d: {"python": dict(verifier[d]["python"]), "lean": dict(verifier[d]["lean"]),
                                  "steps_with_both": verifier[d]["both"]} for d in DATASETS},
        "router_choices": {n: {d: dict(c[d]) for d in DATASETS} for n, c in routers.items()},
    }
    (args.output_dir / "data_statistics.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Table 1: ProcessBench statistics and router splits.
    lines = [r"\begin{table*}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{lrrrrrrrr}", r"\hline",
             r"\textbf{Dataset} & \textbf{Solutions} & \textbf{Steps} & \textbf{Annotated} & \textbf{Correct (\%)} & "
             r"\textbf{Routable} & \textbf{Train} & \textbf{Val.} & \textbf{Test} \\", r"\hline"]
    totals = Counter()
    for d in DATASETS:
        c = per_dataset[d]
        splits = {s: report["splits"][s].get(d, {}).get("steps", 0) for s in ("train", "validation", "test")} if report else {}
        routable = sum(splits.values())
        totals.update(c)
        totals.update({f"split_{s}": v for s, v in splits.items()})
        totals["routable"] += routable
        lines.append(f"{DATASET_NAMES[d]} & {c['solutions']:,} & {c['steps']:,} & {c['annotated_steps']:,} & "
                     f"{pct(c['annotated_correct'], c['annotated_steps'])} & {routable:,} & "
                     f"{splits.get('train', 0):,} & {splits.get('validation', 0):,} & {splits.get('test', 0):,} \\\\")
    lines.append(r"\hline")
    lines.append(f"Total & {totals['solutions']:,} & {totals['steps']:,} & {totals['annotated_steps']:,} & "
                 f"{pct(totals['annotated_correct'], totals['annotated_steps'])} & {totals['routable']:,} & "
                 f"{totals['split_train']:,} & {totals['split_validation']:,} & {totals['split_test']:,} \\\\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{ProcessBench statistics. \emph{Annotated} counts the steps up to and including the first annotated error (steps after it carry no independent label); \emph{Correct} is the share of annotated steps that are correct. \emph{Routable} steps are the annotated steps with a Boolean Python verdict used to train and evaluate the BERT router, split by problem into training, validation and test.}",
              r"\label{tab:data_stats}", r"\end{table*}"]
    (args.output_dir / "data_statistics.tex").write_text("\n".join(lines) + "\n")

    # Table 2: verifier outcome distributions on annotated steps.
    lines = [r"\begin{table*}[ht]", r"\centering", r"\small", r"\begin{tabular}{lrrrrrrr}", r"\hline",
             r"& \multicolumn{4}{c}{\textbf{Python verifier (\%)}} & \multicolumn{3}{c}{\textbf{Lean verifier (\%)}} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             r"\textbf{Dataset} & True & False & Synth.\ fail & Exec.\ fail & True & Autoform.\ fail & Prover fail \\", r"\hline"]
    for d in DATASETS:
        p, l = verifier[d]["python"], verifier[d]["lean"]
        np_, nl = sum(p.values()), sum(l.values())
        lines.append(f"{DATASET_NAMES[d]} & " + " & ".join(pct(p[o], np_) for o in PY_OUTCOMES) + " & "
                     + " & ".join(pct(l[o], nl) for o in LEAN_OUTCOMES) + r" \\")
    p_all = sum((verifier[d]["python"] for d in DATASETS), Counter())
    l_all = sum((verifier[d]["lean"] for d in DATASETS), Counter())
    lines += [r"\hline", "All & " + " & ".join(pct(p_all[o], sum(p_all.values())) for o in PY_OUTCOMES) + " & "
              + " & ".join(pct(l_all[o], sum(l_all.values())) for o in LEAN_OUTCOMES) + r" \\", r"\hline", r"\end{tabular}",
              r"\caption{Distribution of verifier outcomes on the annotated ProcessBench steps. For Python, \emph{Synth.\ fail} means no executable checker was produced (including programs rejected by the sandbox) and \emph{Exec.\ fail} means the checker did not return a Boolean (runtime error, time-out). For Lean, \emph{Autoform.\ fail} means no well-formed theorem was produced and \emph{Prover fail} that no candidate proof compiled; the Lean pipeline never returns an explicit False, so every non-True outcome counts as an incorrect-step verdict.}",
              r"\label{tab:verifier_outcomes}", r"\end{table*}"]
    (args.output_dir / "verifier_outcomes.tex").write_text("\n".join(lines) + "\n")

    # Table 3: LLM router choice distributions.
    if routers:
        lines = [r"\begin{table*}[ht]", r"\centering", r"\small",
                 r"\begin{tabular}{ll" + "r" * len(CHOICES) + "}", r"\hline",
                 r"\textbf{Router} & \textbf{Dataset} & " + " & ".join(rf"\textit{{{c}}}" for c in CHOICES) + r" \\", r"\hline"]
        for name, counts in routers.items():
            for i, d in enumerate(DATASETS):
                n = sum(counts[d].values())
                first = name if i == 0 else ""
                lines.append(f"{first} & {DATASET_NAMES[d]} & " + " & ".join(pct(counts[d][c], n) for c in CHOICES) + r" \\")
            lines.append(r"\hline")
        lines += [r"\end{tabular}",
                  r"\caption{Distribution (\%) of the LLM router choices over the annotated steps. \emph{unresolved} are steps for which the model produced no valid answer within the token budget.}",
                  r"\label{tab:router_choices}", r"\end{table*}"]
        (args.output_dir / "router_choices.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["processbench"], indent=1))
    for d in DATASETS:
        print(d, "python", dict(verifier[d]["python"]), "lean", dict(verifier[d]["lean"]))
    for name, counts in routers.items():
        print(name, {d: dict(counts[d]) for d in DATASETS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Like-for-like comparison of verifiers and routers on the BERT test split.

Every method is scored on exactly the same held-out steps: the steps of the
BERT router's test split that have both a Python and a Lean verdict and for
which every LLM router produced a resolved choice.  Metrics are computed per
ProcessBench sub-dataset, pooled over the subset, and macro-averaged over the
four sub-datasets (the ``Average`` column of the paper tables).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import DATASETS, DATASET_NAMES
from ..data import dataset_of
from .loaders import routed_step_prediction
from .metrics import ConfusionMatrix, exact_mcnemar_p, paired_counts

StepKey = tuple[str, int]

PYTHON_ONLY = "Python only"
LEAN_ONLY = "Lean only"
ORACLE = "Oracle routing"
BERT_DEFAULT = "BERT router"
BERT_TUNED = "BERT router (tuned threshold)"


def metrics_by_dataset(items: Sequence[tuple[str, bool, bool]]) -> dict[str, Any]:
    """``items`` are ``(dataset, expected, predicted)`` triples."""
    result: dict[str, Any] = {}
    for dataset in DATASETS:
        matrix = ConfusionMatrix()
        for name, expected, predicted in items:
            if name == dataset:
                matrix.add(expected, predicted)
        result[dataset] = matrix.as_dict()
    pooled = ConfusionMatrix()
    for _, expected, predicted in items:
        pooled.add(expected, predicted)
    result["pooled"] = pooled.as_dict()
    result["macro_average"] = {
        metric: sum(result[d][metric] for d in DATASETS) / len(DATASETS)
        for metric in ("accuracy", "balanced_accuracy", "correct_step_recall", "incorrect_step_recall")
    }
    return result


def build_comparison(
    truth: Mapping[StepKey, bool],
    python: Mapping[StepKey, bool],
    lean: Mapping[StepKey, bool],
    bert_rows: Sequence[Mapping[str, Any]],
    llm_choices: Mapping[str, Mapping[StepKey, str | None]],
    tuned_threshold: float | None = None,
    default_threshold: float = 0.5,
    extra_bert: Mapping[str, tuple[Sequence[Mapping[str, Any]], float | None]] | None = None,
) -> dict[str, Any]:
    """Score every method on the matched test subset.

    ``bert_rows`` are the test rows with ``p_lean`` of the main BERT router and define
    the test set; ``llm_choices`` maps a router name to its saved choices.
    ``extra_bert`` maps a label to ``(rows, tuned_threshold)`` of additional BERT
    routers trained on the same split (e.g. ablations).  Returns a JSON-serialisable
    report.
    """
    bert_variants: dict[str, tuple[dict[StepKey, Mapping[str, Any]], float | None]] = {
        BERT_DEFAULT: ({(str(r["example_id"]), int(r["step_index"])): r for r in bert_rows}, tuned_threshold)}
    for label, (rows, threshold) in (extra_bert or {}).items():
        index = {(str(r["example_id"]), int(r["step_index"])): r for r in rows}
        if set(index) != set(bert_variants[BERT_DEFAULT][0]):
            raise ValueError(f"{label}: test rows differ from the main BERT router")
        bert_variants[label] = (index, threshold)
    bert_methods: list[tuple[str, str, float]] = []  # (method name, variant, threshold)
    for label, (_, threshold) in bert_variants.items():
        bert_methods.append((label, label, default_threshold))
        if threshold is not None:
            suffix = BERT_TUNED[len(BERT_DEFAULT):]
            bert_methods.append((label + suffix, label, threshold))
    methods = [PYTHON_ONLY, LEAN_ONLY, ORACLE, *llm_choices, *(m for m, _, _ in bert_methods)]
    matched: dict[str, list[tuple[str, bool, bool]]] = {m: [] for m in methods}
    bert_full: dict[str, list[tuple[str, bool, bool]]] = {m: [] for m, _, _ in bert_methods}
    routing: dict[str, list[tuple[str, bool, bool]]] = {m: [] for m, _, _ in bert_methods}
    excluded: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    matched_keys: list[StepKey] = []

    for row in bert_rows:
        key = (str(row["example_id"]), int(row["step_index"]))
        dataset = dataset_of(key[0])
        gold_route = bool(int(row["label"]))
        if key not in truth:
            excluded["unannotated"] += 1
            continue
        if key not in python or key not in lean:
            excluded["missing_verifier"] += 1
            continue
        expected = truth[key]
        predictions: dict[str, bool | None] = {
            PYTHON_ONLY: python[key],
            LEAN_ONLY: lean[key],
            ORACLE: expected if (python[key] == expected or lean[key] == expected) else (not expected),
        }
        for name, variant, threshold in bert_methods:
            p_lean = float(bert_variants[variant][0][key]["p_lean"])
            route_lean = p_lean >= threshold
            routing[name].append((dataset, gold_route, route_lean))
            routes[f"{name}:lean"] += int(route_lean)
            predictions[name] = lean[key] if route_lean else python[key]
            bert_full[name].append((dataset, expected, predictions[name]))
        for name, choices in llm_choices.items():
            predictions[name] = routed_step_prediction(choices.get(key), python[key], lean[key])
        if any(value is None for value in predictions.values()):
            excluded["unresolved_llm_router"] += 1
            continue
        matched_keys.append(key)
        for name in methods:
            matched[name].append((dataset, expected, predictions[name]))

    # Paired significance of each method against Python only on the matched subset.
    expected_list = [e for _, e, _ in matched[PYTHON_ONLY]]
    base = [p for _, _, p in matched[PYTHON_ONLY]]
    significance = {}
    for name in methods:
        if name == PYTHON_ONLY:
            continue
        preds = [p for _, _, p in matched[name]]
        only_method, only_python = paired_counts(expected_list, preds, base)
        significance[name] = {
            "only_method_correct": only_method,
            "only_python_correct": only_python,
            "mcnemar_exact_p": exact_mcnemar_p(only_method, only_python),
        }

    report = {
        "matched_steps": len(matched_keys),
        "matched_problems": len({key[0] for key in matched_keys}),
        "matched_steps_by_dataset": dict(Counter(dataset_of(k[0]) for k in matched_keys)),
        "excluded_test_steps": dict(excluded),
        "thresholds": {"default": default_threshold, "tuned": tuned_threshold,
                       **{f"tuned:{label}": t for label, (_, t) in bert_variants.items() if label != BERT_DEFAULT}},
        "lean_routes_on_bert_test": dict(routes),
        "matched_test": {m: metrics_by_dataset(v) for m, v in matched.items()},
        "bert_full_test": {m: metrics_by_dataset(v) for m, v in bert_full.items() if v},
        "routing_label_metrics": {m: metrics_by_dataset(v) for m, v in routing.items() if v},
        "significance_vs_python_only": significance,
        "policy": (
            "Step verdicts: Python/Lean True -> correct step, anything else -> incorrect. "
            "LLM router choices: python/lean follow that verifier; neither -> incorrect; "
            "inconclusive -> correct; tie resolved only when verifiers agree. Steps with an "
            "unresolved LLM choice (request error or discordant tie) are excluded for every method."
        ),
    }
    return report


def _format(value: float, best: bool) -> str:
    text = f"{100 * value:.2f}"
    return rf"\textbf{{{text}}}" if best else text


def latex_table(
    report: Mapping[str, Any],
    metric: str,
    display_names: Mapping[str, str],
    caption: str,
    label: str,
    section: str = "matched_test",
    bold_rows: Sequence[str] | None = None,
) -> str:
    """Render one metric of the comparison as a LaTeX ``table*``.

    ``display_names`` maps method keys to the LaTeX used in the first column and
    fixes the row order.  Bold marks the best value per column among
    ``bold_rows`` (default: every listed method).
    """
    groups = report[section]
    rows = [name for name in display_names if name in groups]
    candidates = [r for r in rows if bold_rows is None or r in bold_rows]
    columns = [*DATASETS, "macro_average"]
    best = {
        column: max(groups[r][column][metric] for r in candidates) for column in columns
    }
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Model / setting} & " + " & ".join(rf"\textbf{{{DATASET_NAMES[d]}}}" for d in DATASETS)
        + r" & \textbf{Average} \\",
        r"\hline",
    ]
    for name in rows:
        cells = [
            _format(groups[name][column][metric], name in candidates and groups[name][column][metric] == best[column])
            for column in columns
        ]
        lines.append(display_names[name] + " & " + " & ".join(cells) + r" \\")
        if name in (ORACLE,) or name == rows[-1]:
            lines.append(r"\hline" if name == rows[-1] else r"\midrule")
    lines += [r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_summary(report: Mapping[str, Any]) -> None:
    print(f"Matched test steps: {report['matched_steps']} "
          f"({report['matched_problems']} problems); excluded: {report['excluded_test_steps']}")
    for metric in ("accuracy", "balanced_accuracy"):
        print(f"\n{metric} (%) on the matched test subset")
        header = f"{'method':34s}" + "".join(f"{DATASET_NAMES[d]:>15s}" for d in DATASETS) + f"{'Average':>10s}{'Pooled':>10s}"
        print(header)
        for name, groups in report["matched_test"].items():
            values = [groups[d][metric] for d in DATASETS] + [groups["macro_average"][metric], groups["pooled"][metric]]
            print(f"{name:34s}" + "".join(f"{100 * v:15.2f}" for v in values[:4]) + f"{100 * values[4]:10.2f}{100 * values[5]:10.2f}")
    print("\nExact McNemar test versus Python only (matched subset):")
    for name, stats in report["significance_vs_python_only"].items():
        print(f"  {name:34s} only-method={stats['only_method_correct']:4d} only-python={stats['only_python_correct']:4d} p={stats['mcnemar_exact_p']:.3g}")

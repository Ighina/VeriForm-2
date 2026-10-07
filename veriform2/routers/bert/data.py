"""Training data for the BERT router.

Routing labels are derived from ProcessBench annotations and the Python
verifier's verdicts with the following truth table:

=================  ======================  ======
Step annotation    Python verdict          Label
=================  ======================  ======
correct            True                    Python
correct            False                   Lean
incorrect          False                   Python
incorrect          True                    Lean
=================  ======================  ======

i.e. a step is labelled *Python* when the Python verifier already agrees with
the annotation and *Lean* otherwise, which prioritises the cheaper verifier.
Steps after the first annotated error are excluded (no independent label), and
so are steps whose Python verdict is not Boolean unless ``non_boolean="false"``.

Splits are allocated per problem (all solutions of the same problem statement
stay together) and per sub-dataset, with 75/5/20 train/validation/test
proportions, using largest-remainder rounding and a fixed seed.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

LABELS = {0: "Python", 1: "Lean"}
RATIOS = {"train": 0.75, "validation": 0.05, "test": 0.20}


def routing_label(correct: bool, python_result: bool) -> int:
    if type(correct) is not bool or type(python_result) is not bool:
        raise ValueError("Routing labels require two Boolean values")
    return int(correct != python_result)


def build_examples(annotations: Iterable[dict[str, Any]], python_results: Iterable[dict[str, Any]],
                   non_boolean: str = "exclude") -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join annotated steps with Python verifier results and derive routing labels."""
    if non_boolean not in ("exclude", "false"):
        raise ValueError("Unknown non-Boolean policy")
    truth: dict[str, dict[str, Any]] = {}
    for row in annotations:
        eid = str(row["id"])
        if eid in truth:
            raise ValueError(f"Duplicate annotation ID: {eid}")
        steps, label = row["steps"], int(row["label"])
        if label < -1 or label >= len(steps):
            raise ValueError(f"Invalid first-error label: {eid}")
        truth[eid] = row
    examples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, int]] = set()
    for row in python_results:
        if row.get("representation", "linear") != "linear":
            counts["non_linear"] += 1
            continue
        eid, index = str(row["example_id"]), int(row["step_index"])
        key = (eid, index)
        if key in seen:
            raise ValueError(f"Duplicate Python result: {key}")
        seen.add(key)
        if eid not in truth:
            raise ValueError(f"Missing annotation: {eid}")
        annotation = truth[eid]
        steps, first_error = annotation["steps"], int(annotation["label"])
        if not 0 <= index < len(steps):
            raise ValueError(f"Invalid step index: {key}")
        if row["target_step"] != steps[index]:
            raise ValueError(f"Step text does not match annotation: {key}")
        if first_error != -1 and index > first_error:
            counts["after_first_error"] += 1
            continue
        outcome = str(row.get("outcome", row.get("sandbox_outcome", ""))).strip().lower()
        if outcome not in ("true", "false"):
            counts["non_boolean"] += 1
            if non_boolean == "exclude":
                continue
        correct = first_error == -1 or index < first_error
        text = steps[index]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty step: {key}")
        problem = " ".join(annotation["problem"].split())
        examples.append(dict(
            text=text, label=routing_label(correct, outcome == "true"), example_id=eid, step_index=index,
            dataset=annotation.get("split") or eid.partition("-")[0],
            problem_id=hashlib.sha256(problem.encode()).hexdigest(),
        ))
    counts["included_steps"] = len(examples)
    counts["included_problems"] = len({e["problem_id"] for e in examples})
    return examples, dict(counts)


def split_examples(examples: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Largest-remainder 75/5/20 allocation of problems, stratified by sub-dataset."""
    groups: dict[str, set[str]] = defaultdict(set)
    source_by_group: dict[str, str] = {}
    for e in examples:
        group, source = e["problem_id"], e["dataset"]
        if group in source_by_group and source_by_group[group] != source:
            raise ValueError("Identical problem appears in multiple sources")
        source_by_group[group] = source
        groups[source].add(group)
    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    for source, ids in sorted(groups.items()):
        ordered = sorted(ids)
        rng.shuffle(ordered)
        sizes = {s: int(len(ordered) * r) for s, r in RATIOS.items()}
        remainder = len(ordered) - sum(sizes.values())
        order = sorted(RATIOS, key=lambda s: -(len(ordered) * RATIOS[s] - sizes[s]))
        for s in order[:remainder]:
            sizes[s] += 1
        if any(n == 0 for n in sizes.values()):
            raise ValueError(f"Too few eligible problems to stratify {source}: {len(ordered)}")
        start = 0
        for s, n in sizes.items():
            for group in ordered[start:start + n]:
                assignment[group] = s
            start += n
    splits: dict[str, list[dict[str, Any]]] = {s: [] for s in RATIOS}
    for e in examples:
        splits[assignment[e["problem_id"]]].append(e)
    return splits


def summarize_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            source: {
                "problems": len({e["problem_id"] for e in rows if e["dataset"] == source}),
                "steps": sum(e["dataset"] == source for e in rows),
                "labels": dict(Counter(LABELS[e["label"]] for e in rows if e["dataset"] == source)),
            }
            for source in sorted({e["dataset"] for e in rows})
        }
        for split, rows in splits.items()
    }

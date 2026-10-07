"""ProcessBench loading and step-level correctness labels.

ProcessBench (Zheng et al., 2025) annotates each step-by-step solution with the
zero-based index of the *first* erroneous step, or ``-1`` when the whole
solution is correct.  Consequently every step before the labelled one is
correct, the labelled step is incorrect, and later steps carry no independent
annotation and are excluded from step-level evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import DATASETS

StepKey = tuple[str, int]

DEFAULT_DATASET = "Qwen/ProcessBench"


def dataset_of(example_id: str) -> str:
    """Return the ProcessBench sub-dataset encoded in an example id such as ``math-12``."""
    return example_id.partition("-")[0]


def iter_processbench_rows(source: str | Path = DEFAULT_DATASET) -> Iterable[dict[str, Any]]:
    """Yield ProcessBench rows from a Hugging Face dataset id or a local JSON file.

    Every yielded row has at least ``id``, ``problem``, ``steps``, ``label`` and a
    ``split`` field naming the sub-dataset.
    """
    path = Path(source)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            rows = data["data"]
        elif isinstance(data, dict):
            # Column-oriented export (``{"id": {"0": ...}, "steps": {"0": ...}}``).
            keys = sorted(data.get("id", {}), key=lambda v: int(v) if str(v).isdigit() else v)
            rows = [
                {field: values.get(key) for field, values in data.items() if isinstance(values, dict)}
                for key in keys
            ]
        else:
            raise ValueError(f"Unsupported dataset JSON structure: {path}")
        for row in rows:
            row = dict(row)
            row.setdefault("split", dataset_of(str(row["id"])))
            yield row
        return

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "Loading Qwen/ProcessBench requires the Hugging Face `datasets` package"
        ) from error
    dataset = load_dataset(str(source))
    for split_name, split in dataset.items():
        for row in split:
            row = dict(row)
            row.setdefault("split", split_name)
            yield row


def load_examples(source: str | Path = DEFAULT_DATASET) -> dict[str, dict[str, Any]]:
    """Return ProcessBench rows indexed by example id."""
    examples: dict[str, dict[str, Any]] = {}
    for row in iter_processbench_rows(source):
        example_id = str(row["id"])
        if example_id in examples:
            raise ValueError(f"Duplicate ProcessBench id: {example_id}")
        examples[example_id] = row
    return examples


def validate_label(example_id: str, label: int, step_count: int) -> None:
    if label < -1 or label >= step_count:
        raise ValueError(f"{example_id}: label {label} is invalid for {step_count} steps")


def step_labels_for(example_id: str, label: int, step_count: int) -> dict[StepKey, bool]:
    """Expand a first-error label into per-step correctness for the annotated prefix."""
    validate_label(example_id, label, step_count)
    last_labelled = step_count - 1 if label == -1 else label
    return {(example_id, index): label == -1 or index < label for index in range(last_labelled + 1)}


def load_step_labels(source: str | Path = DEFAULT_DATASET) -> dict[StepKey, bool]:
    """Return ``{(example_id, step_index): is_correct}`` for independently labelled steps."""
    labels: dict[StepKey, bool] = {}
    for row in iter_processbench_rows(source):
        labels.update(step_labels_for(str(row["id"]), int(row["label"]), len(row.get("steps") or [])))
    return labels


def filter_dataset(keys: Iterable[StepKey], dataset: str) -> list[StepKey]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown ProcessBench dataset: {dataset}")
    return [key for key in keys if dataset_of(key[0]) == dataset]

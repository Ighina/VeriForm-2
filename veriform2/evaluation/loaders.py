"""Loaders for saved verifier and router outputs.

All loaders return dictionaries keyed by ``(example_id, step_index)``.  They
accept both the raw files written by the pipeline scripts and the compact CSV
exports shipped in ``results/`` (see ``scripts/export_compact_results.py``).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

StepKey = tuple[str, int]

CHOICES = ("python", "lean", "tie", "neither", "inconclusive")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    # Lean diagnostics and LLM rationales can exceed csv's default field limit.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return _csv_rows(path)
    return _jsonl_rows(path)


def _is_true(value: Any) -> bool:
    return str(value or "").strip().lower() == "true"


def _step_index(row: dict[str, Any]) -> int:
    """Read the zero-based step index from any of the stored encodings."""
    if row.get("step_index") not in (None, ""):
        return int(row["step_index"])
    step_id = str(row.get("step_id", ""))
    if step_id.startswith("step_"):  # Lean pipeline: ``step_<index>``
        return int(step_id.rsplit("_", 1)[1])
    if step_id.startswith("linear_"):  # Python verifier: ``linear_<index + 1>``
        return int(step_id.rsplit("_", 1)[1]) - 1
    raise ValueError(f"Cannot determine the step index of row {row!r}")


def load_python_predictions(path: Path) -> dict[StepKey, bool]:
    """Python verifier verdicts: ``True`` only when the sandboxed checker returned True.

    Autoformalisation failures, sandbox failures and ``False`` results all map to
    an incorrect-step verdict.
    """
    result: dict[StepKey, bool] = {}
    for row in read_rows(path):
        if row.get("representation", "linear") != "linear":
            continue
        key = (str(row["example_id"]), _step_index(row))
        outcome = row.get("outcome") if "outcome" in row else row.get("sandbox_outcome")
        result[key] = _is_true(outcome)
    return result


def load_lean_predictions(path: Path) -> dict[StepKey, bool]:
    """Lean verifier verdicts: ``True`` only when a proof of the formal statement compiled."""
    result: dict[StepKey, bool] = {}
    for row in read_rows(path):
        key = (str(row["example_id"]), _step_index(row))
        outcome = row.get("semantic_outcome") or row.get("result") or ""
        result[key] = _is_true(outcome)
    return result


def load_verifier_outcomes(path: Path) -> dict[StepKey, str]:
    """Raw verifier outcome per step (``True``, ``False`` or a failure category).

    Works for both the Python results (``outcome``) and the Lean results
    (``semantic_outcome``); used by the router-free fallback baselines, which need
    to know whether a verifier produced a verdict at all.
    """
    result: dict[StepKey, str] = {}
    for row in read_rows(path):
        if row.get("representation", "linear") != "linear":
            continue
        key = (str(row["example_id"]), _step_index(row))
        outcome = row.get("outcome") if "outcome" in row else (
            row.get("semantic_outcome") or row.get("sandbox_outcome") or row.get("result") or "")
        result[key] = str(outcome or "").strip()
    return result


def has_verdict(outcome: str | None) -> bool:
    """Whether a raw verifier outcome is an actual verdict rather than a failure."""
    return str(outcome or "").strip().lower() in ("true", "false")


def load_step_verdicts(path: Path) -> dict[StepKey, bool | None]:
    """Per-step correctness verdicts of a direct critic (``verdict`` column).

    ``True``/``correct`` and ``False``/``incorrect`` map to booleans; empty values
    (failed requests, unparseable answers) are stored as ``None``.
    """
    result: dict[StepKey, bool | None] = {}
    for row in read_rows(path):
        key = (str(row["example_id"]), _step_index(row))
        if key in result:
            raise ValueError(f"Duplicate verdict row: {key}")
        value = str(row.get("verdict") or "").strip().lower()
        if value in ("true", "correct"):
            result[key] = True
        elif value in ("false", "incorrect"):
            result[key] = False
        elif value == "":
            result[key] = None
        else:
            raise ValueError(f"Unexpected verdict {value!r} for {key}")
    return result


def load_router_choices(path: Path) -> dict[StepKey, str | None]:
    """LLM router choices (``python``/``lean``/``tie``/``neither``/``inconclusive``).

    Steps whose request failed are stored with ``None``.
    """
    result: dict[StepKey, str | None] = {}
    for row in read_rows(path):
        key = (str(row["example_id"]), _step_index(row))
        if key in result:
            raise ValueError(f"Duplicate router row: {key}")
        choice = (row.get("choice") or "").strip().lower() or None
        if choice is not None and choice not in CHOICES:
            raise ValueError(f"Unexpected router choice {choice!r} for {key}")
        result[key] = choice
    return result


def load_bert_probabilities(path: Path) -> list[dict[str, Any]]:
    """Rows of the BERT router test split with ``p_lean`` and the routing label."""
    rows = read_rows(path)
    for row in rows:
        row["step_index"] = int(row["step_index"])
        row["label"] = int(row["label"])
        row["p_lean"] = float(row["p_lean"])
    return rows


def routed_step_prediction(choice: str | None, python_result: bool, lean_result: bool) -> bool | None:
    """Step verdict obtained by following a router choice.

    ``neither`` means the router judged both verifications wrong and the step is
    treated as incorrect; ``inconclusive`` means no checkable inference was made
    and the step is treated as correct.  A ``tie`` is resolved only when the two
    verifiers agree; otherwise, as for failed requests, ``None`` is returned.
    """
    if choice == "python":
        return python_result
    if choice == "lean":
        return lean_result
    if choice == "tie" and python_result == lean_result:
        return python_result
    if choice == "neither":
        return False
    if choice == "inconclusive":
        return True
    return None

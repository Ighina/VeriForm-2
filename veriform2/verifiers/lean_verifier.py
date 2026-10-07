"""Lean verifier: interface to the autoformalise-and-prove pipeline.

The Lean pipeline itself (Goedel-Formalizer-V2-8B autoformaliser, Goedel-Prover-
V2-8B prover, Lean 4 + Mathlib compilation) lives in the external ``veriform``
package and is driven by ``scripts/run_lean_verifier.py``.  Each run writes

``<run>/steps/LinearDAGModel/step_outcomes.csv``
    one row per step with ``semantic_outcome`` in ``True`` (a proof of the
    autoformalised statement compiled), ``Autoformalisation failure`` (no
    well-formed theorem was produced) or ``Prover failure`` (no candidate proof
    compiled), plus the heuristic ``alignment_status`` of the statement;
``<run>/steps/LinearDAGModel/<example_id>.lean``
    the accumulated Lean file, one ``-- Step i:`` block per step;
``<run>/steps/chain_<n>/step_<i>_*.txt``
    the raw prompts and model outputs of both models.

This module only reads those artefacts.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

StepKey = tuple[str, int]

OUTCOMES = ("True", "False", "Autoformalisation failure", "Prover failure")


def _raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_step_outcomes(run_dir: Path) -> dict[StepKey, dict[str, str]]:
    """Index the ``step_outcomes.csv`` rows of a Lean run by ``(example_id, step_index)``."""
    path = run_dir / "steps/LinearDAGModel/step_outcomes.csv"
    if not path.is_file():
        raise SystemExit(f"Lean step outcomes not found: {path}")
    _raise_csv_limit()
    result: dict[StepKey, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            _, separator, step = row["result_id"].partition(".")
            if not separator:
                raise ValueError(f"Malformed Lean result_id: {row['result_id']}")
            key = (row["example_id"], int(step))
            if key in result:
                raise ValueError(f"Duplicate Lean row: {key}")
            result[key] = row
    return result


def lean_blocks(path: Path) -> dict[int, str]:
    """Split an accumulated Lean file into its per-step ``-- Step i:`` blocks."""
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^-- Step (\d+):.*$", text))
    blocks: dict[int, str] = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        blocks[int(match.group(1))] = text[match.start():end].strip()
    return blocks


def runtime_artifacts(run_dir: Path, result_id: str) -> dict[str, str]:
    """Raw autoformaliser/prover prompts and outputs recorded for one step."""
    chain, separator, step = result_id.partition(".")
    if not separator:
        return {}
    directory = run_dir / "steps" / f"chain_{int(chain)}"
    prefix = f"step_{int(step):06}_"
    return {
        path.name: path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(directory.glob(f"{prefix}*.txt"))
    }

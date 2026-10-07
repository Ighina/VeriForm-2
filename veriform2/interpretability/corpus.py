"""Dataset-wide Integrated Gradients analysis of the BERT router.

For every step of the test split the routing margin ``logit(Lean) -
logit(Python)`` is attributed to the input tokens.  Token attributions are
merged into words and then aggregated across the whole split, giving for each
word type (and for coarse lexical categories) the mean contribution towards Lean
(positive) or Python (negative) in logit units.  Because the margin is in the
same units for every step, raw attributions are comparable across steps and
no per-example normalisation is applied.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .integrated_gradients import explain
from .words import CATEGORIES, categorize, normalize_word, word_units


def attribute_rows(model, tokenizer, rows: Iterable[dict[str, Any]], n_steps: int = 50,
                   internal_batch_size: int = 25, delta_tolerance: float = 0.02,
                   max_steps: int = 400, log_every: int = 50) -> Iterable[dict[str, Any]]:
    """Yield one ``explain`` result (plus word units and metadata) per input row."""
    max_length = min(512, tokenizer.model_max_length)
    for number, row in enumerate(rows, 1):
        result = explain(model, tokenizer, row["text"], target="margin", n_steps=n_steps,
                         internal_batch_size=internal_batch_size, delta_tolerance=delta_tolerance,
                         max_steps=max_steps)
        offsets = tokenizer(row["text"], truncation=True, max_length=max_length,
                            return_offsets_mapping=True)["offset_mapping"]
        units = word_units(row["text"], offsets, result["attributions"])
        result.update({k: row[k] for k in ("example_id", "step_index", "dataset", "label", "problem_id") if k in row})
        result["words"] = [dict(text=u["text"], start=u["start"], end=u["end"], attribution=u["attribution"],
                                key=normalize_word(u["text"]), category=categorize(normalize_word(u["text"])))
                           for u in units]
        if number % log_every == 0:
            print(f"  attributed {number} steps", flush=True)
        yield result


def aggregate_words(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-word-type statistics over all occurrences in the corpus."""
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: dict(count=0, examples=set(), total=0.0, squares=0.0))
    for r in results:
        identity = (r.get("example_id"), r.get("step_index"))
        for w in r["words"]:
            s = stats[w["key"]]
            s["count"] += 1
            s["examples"].add(identity)
            s["total"] += w["attribution"]
            s["squares"] += w["attribution"] ** 2
            s["category"] = w["category"]
    table = []
    for key, s in stats.items():
        mean = s["total"] / s["count"]
        variance = max(0.0, s["squares"] / s["count"] - mean**2)
        sem = math.sqrt(variance / s["count"]) if s["count"] > 1 else 0.0
        table.append(dict(word=key, category=s["category"], count=s["count"], examples=len(s["examples"]),
                          total_attribution=s["total"], mean_attribution=mean, sem=sem))
    table.sort(key=lambda t: -t["mean_attribution"])
    return table


def aggregate_categories(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean attribution and share of total absolute attribution per lexical category."""
    sums: dict[str, dict[str, float]] = {c: dict(count=0, total=0.0, squares=0.0, absolute=0.0) for c in CATEGORIES}
    for r in results:
        for w in r["words"]:
            s = sums[w["category"]]
            s["count"] += 1
            s["total"] += w["attribution"]
            s["squares"] += w["attribution"] ** 2
            s["absolute"] += abs(w["attribution"])
    grand_absolute = sum(s["absolute"] for s in sums.values()) or 1.0
    grand_count = sum(s["count"] for s in sums.values()) or 1
    table = []
    for category, s in sums.items():
        n = s["count"]
        mean = s["total"] / n if n else 0.0
        variance = max(0.0, s["squares"] / n - mean**2) if n else 0.0
        table.append(dict(category=category, count=n, share_of_tokens=n / grand_count, mean_attribution=mean,
                          sem=math.sqrt(variance / n) if n > 1 else 0.0, total_attribution=s["total"],
                          share_of_absolute_attribution=s["absolute"] / grand_absolute))
    return table


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(results: list[dict[str, Any]], categories: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [r["margin"] for r in results]
    return {
        "steps": len(results),
        "converged": sum(r["converged"] for r in results),
        "mean_margin": sum(margins) / len(margins),
        "steps_routed_to_lean_at_0.5": sum(m >= 0 for m in margins),
        "mean_abs_convergence_delta": sum(abs(r["convergence_delta"]) for r in results) / len(results),
        "categories": categories,
    }


def load_results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

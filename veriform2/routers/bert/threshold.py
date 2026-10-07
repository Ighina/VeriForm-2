"""Validation-selected decision threshold for the BERT router.

The router is strongly biased towards Python (about 9 in 10 training steps are
labelled Python).  This module searches every distinct validation probability
for the threshold maximising routing *balanced accuracy*, freezes it, and only
then evaluates the test split.  Probabilities of both splits are saved so that
downstream comparisons never need to rerun the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .predict import BertRouter


def select_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[float, list[tuple[float, float]]]:
    """Return the balanced-accuracy-maximising threshold and the full validation curve."""
    from sklearn.metrics import balanced_accuracy_score

    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Validation must contain both classes")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    candidates = np.unique(np.r_[0.0, 0.5, p, np.nextafter(1.0, 2.0)])
    scores = [(float(t), float(balanced_accuracy_score(y, p >= t))) for t in candidates]
    # Deterministic ties: closest to the default threshold, then the larger one.
    best = max(scores, key=lambda s: (s[1], -abs(s[0] - 0.5), s[0]))
    return best[0], scores


def report(rows: Sequence[dict[str, Any]], probabilities: Sequence[float], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

    labels = [int(r["label"]) for r in rows]
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    return dict(steps=len(rows), accuracy=accuracy_score(labels, pred),
                balanced_accuracy=balanced_accuracy_score(labels, pred),
                classification_report=classification_report(
                    labels, pred, labels=[0, 1], target_names=["Python", "Lean"], output_dict=True, zero_division=0))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/bert_router"),
                        help="Training output directory (contains model/, validation.jsonl, test.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run-dir>/threshold_tuning")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    args.output_dir = args.output_dir or args.run_dir / "threshold_tuning"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    router = BertRouter(args.run_dir / "model", device=args.device, batch_size=args.batch_size)

    def read(name: str) -> list[dict[str, Any]]:
        return [json.loads(s) for s in (args.run_dir / f"{name}.jsonl").read_text().splitlines() if s.strip()]

    validation = read("validation")
    vp = router.lean_probabilities([r["text"] for r in validation])
    threshold, curve = select_threshold([r["label"] for r in validation], vp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(threshold=threshold, comparison="p_lean >= threshold", model=str((args.run_dir / "model").resolve()),
                  selected_on="validation", objective="balanced_accuracy", device=args.device)

    def save(name: str, value: Any) -> None:
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")

    save("threshold.json", config)  # Persist the decision before reading test labels.
    save("validation_threshold_curve.json", curve)
    print("Validation-selected threshold:", threshold, flush=True)
    test = read("test")
    if {r["problem_id"] for r in validation} & {r["problem_id"] for r in test}:
        raise ValueError("Problem leakage between validation and test")
    tp = router.lean_probabilities([r["text"] for r in test])
    results = dict(config=config,
                   validation={"default": report(validation, vp, 0.5), "tuned": report(validation, vp, threshold)},
                   test={"default": report(test, tp, 0.5), "tuned": report(test, tp, threshold)}, per_dataset={})
    for source in sorted({r["dataset"] for r in test}):
        rows, ps = zip(*[(r, p) for r, p in zip(test, tp) if r["dataset"] == source])
        results["per_dataset"][source] = {"default": report(rows, ps, 0.5), "tuned": report(rows, ps, threshold)}
    for name, rows, ps in (("validation", validation, vp), ("test", test, tp)):
        with (args.output_dir / f"{name}_probabilities.jsonl").open("w") as handle:
            for row, p in zip(rows, ps):
                handle.write(json.dumps({**row, "p_lean": p, "prediction": "Lean" if p >= threshold else "Python"}) + "\n")
    save("metrics.json", results)
    print(json.dumps({k: {m: results["test"][k][m] for m in ("accuracy", "balanced_accuracy")}
                      for k in ("default", "tuned")}, indent=2))
    return 0

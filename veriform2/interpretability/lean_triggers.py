"""Local word-level interventions on the strongest Lean predictions.

For each selected step the Lean logit is attributed with Integrated Gradients,
WordPiece scores are merged into words, and the highest-attribution words are
tested causally by deleting them or replacing them with ``[MASK]`` and
re-evaluating ``P(Lean)``.  Low-attribution words serve as descriptive controls.
"""
from __future__ import annotations

import re
from typing import Any

from .integrated_gradients import explain


def word_candidates(text: str, tokenizer, attributions: list[float]) -> list[dict[str, Any]]:
    """Aggregate WordPieces into word occurrences (with character spans) using offsets."""
    encoded = tokenizer(text, truncation=True, max_length=min(512, tokenizer.model_max_length),
                        return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(attributions):
        raise ValueError("Attribution/token offset mismatch")
    words = []
    for match in re.finditer(r"\b\w+(?:['’]\w+)*\b", text):
        start, end = match.span()
        indices = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
        if not indices or offsets[indices[-1]][1] < end:
            continue  # word (partly) outside the truncated model input
        words.append(dict(word=match.group(), start=start, end=end,
                          attribution=sum(attributions[i] for i in indices)))
    return words


def perturb(text: str, words: list[dict[str, Any]], replacement: str) -> str:
    for w in sorted(words, key=lambda w: -w["start"]):
        text = text[: w["start"]] + replacement + text[w["end"]:]
    return text


def lean_probabilities(model, tokenizer, texts: list[str], batch_size: int = 4) -> list[float]:
    import torch

    device = next(model.parameters()).device
    result: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=min(512, tokenizer.model_max_length), return_tensors="pt").to(device)
            result.extend(model(**encoded).logits.softmax(-1)[:, 1].tolist())
    return result


def analyze(model, tokenizer, row: dict[str, Any], n_steps: int = 100, max_steps: int = 1600,
            candidates: int = 8, controls: int = 3) -> dict[str, Any]:
    """IG on the Lean logit followed by deletion/masking interventions."""
    result = explain(model, tokenizer, row["text"], target="Lean", n_steps=n_steps, max_steps=max_steps,
                     internal_batch_size=4)
    result.update({k: row[k] for k in ("example_id", "step_index", "dataset") if k in row})
    if "label" in row:
        result["gold_label"] = {0: "Python", 1: "Lean"}[int(row["label"])]
    words = word_candidates(row["text"], tokenizer, result["attributions"])
    positive = sorted([w for w in words if w["attribution"] > 0], key=lambda w: -w["attribution"])[:candidates]
    used = {w["start"] for w in positive}
    control_words = sorted([w for w in words if w["start"] not in used], key=lambda w: abs(w["attribution"]))[:controls]
    tests, variants = [], []
    for kind, selected in [("candidate", [w]) for w in positive] + [("control", [w]) for w in control_words] + [
            ("joint_top5", positive[:5])]:
        if not selected:
            continue
        tests.append(dict(kind=kind, words=selected))
        variants += [perturb(row["text"], selected, ""), perturb(row["text"], selected, tokenizer.mask_token)]
    probabilities = lean_probabilities(model, tokenizer, variants)
    original = result["p_lean"]
    for i, test in enumerate(tests):
        test["p_lean_deleted"], test["p_lean_masked"] = probabilities[2 * i:2 * i + 2]
        test["deletion_drop"] = original - test["p_lean_deleted"]
        test["mask_drop"] = original - test["p_lean_masked"]
        test["both_reduce_lean"] = test["deletion_drop"] > 0 and test["mask_drop"] > 0
    result["word_tests"] = tests
    return result


def summarize(results: list[dict[str, Any]], min_drop: float = 0.005) -> str:
    """Markdown summary: strongest confirmed word per example and recurring words."""
    lines = ["| Example / step | P(Lean) | Gold route | Strongest confirmed word | Deletion drop (pp) | Mask drop (pp) |",
             "|---|---:|---|---|---:|---:|"]
    repeated: dict[str, dict[tuple, float]] = {}
    for r in results:
        confirmed = [t for t in r["word_tests"] if t["kind"] == "candidate" and t["both_reduce_lean"]]
        best = max(confirmed, key=lambda t: min(t["deletion_drop"], t["mask_drop"]), default=None)
        word = best["words"][0]["word"] if best else "none confirmed"
        d = f"{100 * best['deletion_drop']:.2f}" if best else "—"
        m = f"{100 * best['mask_drop']:.2f}" if best else "—"
        lines.append(f"| {r['example_id']} / {r['step_index']} | {100 * r['p_lean']:.2f}% | "
                     f"{r.get('gold_label', '?')} | {word} | {d} | {m} |")
        for t in r["word_tests"]:
            if t["kind"] == "candidate" and min(t["deletion_drop"], t["mask_drop"]) >= min_drop:
                key = t["words"][0]["word"].lower()
                identity = (r["example_id"], r["step_index"])
                effect = min(t["deletion_drop"], t["mask_drop"])
                repeated.setdefault(key, {})[identity] = max(effect, repeated.get(key, {}).get(identity, 0))
    recurring = sorted(((w, e) for w, e in repeated.items() if len(e) >= 2), key=lambda i: (-len(i[1]), -sum(i[1].values())))
    lines += ["", "## Recurring words (≥0.5 pp drop under both interventions in ≥2 examples)", "",
              "| Word | Examples | Smaller drop range (pp) |", "|---|---:|---:|"]
    for word, effects in recurring:
        lines.append(f"| {word} | {len(effects)} | {100 * min(effects.values()):.2f}–{100 * max(effects.values()):.2f} |")
    return "\n".join(lines) + "\n"

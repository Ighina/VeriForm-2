"""Aggregating WordPiece attributions into words and lexical categories."""
from __future__ import annotations

import re
from typing import Any, Sequence

NEGATION = {
    "not", "no", "never", "cannot", "neither", "nor", "without", "none", "nothing",
    "n't", "doesn't", "isn't", "can't", "don't", "won't", "wouldn't", "couldn't", "shouldn't",
    "didn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "mustn't", "ain't",
}
HEDGE_CONTRAST = {
    "however", "but", "although", "though", "yet", "nevertheless", "whereas", "unless", "otherwise",
    "seems", "seem", "appears", "appear", "might", "may", "could", "perhaps", "maybe", "possibly",
    "probably", "likely", "unlikely", "unclear", "ambiguous", "assume", "assuming", "suppose",
    "supposing", "suppose,", "if", "whether", "alternatively", "instead", "actually", "wait",
    "hmm", "recheck", "re-check", "re-evaluate", "reconsider", "revisit", "mistake", "error",
    "wrong", "incorrect", "correct", "verify", "check", "double-check",
}
MATH_CHARS = set("=+*/^_<>{}()[]|$\\~≤≥≠×÷√∑∏∫∞±→←")


def word_units(text: str, offsets: Sequence[tuple[int, int]], attributions: Sequence[float]) -> list[dict[str, Any]]:
    """Sum token attributions into whitespace-delimited units of the original text.

    A WordPiece spanning two units (rare, e.g. around punctuation) is split
    proportionally to the overlap so the total attribution is preserved.
    """
    if len(offsets) != len(attributions):
        raise ValueError("Attribution/offset mismatch")
    units = [dict(text=m.group(), start=m.start(), end=m.end(), attribution=0.0)
             for m in re.finditer(r"\S+", text)]
    for (a, b), score in zip(offsets, attributions):
        if a == b:  # special tokens
            continue
        overlaps = [(u, max(0, min(b, u["end"]) - max(a, u["start"]))) for u in units]
        total = sum(n for _, n in overlaps)
        if total == 0:
            raise ValueError(f"Unmapped token span {(a, b)}")
        for unit, n in overlaps:
            if n:
                unit["attribution"] += score * n / total
    return units


def normalize_word(unit: str) -> str:
    """Lower-case and strip surrounding punctuation; keep LaTeX and math tokens intact.

    Brackets are stripped as well when the unit contains no LaTeX, so that
    ``(not`` and ``not`` are the same word while ``\\(x\\)`` is left alone.
    """
    key = unit.lower().strip(".,;:!?\"'“”‘’`")
    if "\\" not in key:
        key = key.strip("()[]{}.,;:!?\"'“”‘’`")
    return key or unit.lower()


def categorize(word: str) -> str:
    """Coarse lexical category used for the aggregate analysis."""
    core = word.strip("()[]{}.,;:!?")
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?%?", core):
        return "number"
    if "\\" in word or any(ch in MATH_CHARS for ch in word):
        return "math notation"
    if any(ch.isdigit() for ch in word):
        return "number"
    stripped = word.strip("(),.;:")
    if stripped in NEGATION or stripped.endswith("n't"):
        return "negation"
    if stripped in HEDGE_CONTRAST:
        return "hedging / contrast"
    return "other word"


CATEGORIES = ("number", "math notation", "negation", "hedging / contrast", "other word")

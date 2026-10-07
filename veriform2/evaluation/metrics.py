"""Step-level and solution-level metrics.

Step-level metrics treat *a correct step* as the positive class.  Balanced
accuracy is the mean of the recall on correct steps and the recall on incorrect
steps, which matters because roughly nine in ten annotated ProcessBench steps are
correct.

The solution-level ``OfficialMetric`` reproduces the ProcessBench protocol: the
harmonic mean between the accuracy on erroneous solutions (exact first-error
localisation) and the accuracy on fully correct solutions.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

StepKey = tuple[str, int]


class ConfusionMatrix:
    """Binary confusion matrix whose positive label is a correct step."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, expected: bool, predicted: bool) -> None:
        if expected and predicted:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif expected:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        """Recall on correct steps (sensitivity)."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def specificity(self) -> float:
        """Recall on incorrect steps."""
        return self.tn / (self.tn + self.fp) if self.tn + self.fp else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int | list[list[int]]]:
        return {
            "n": self.total,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "correct_step_recall": self.recall,
            "incorrect_step_recall": self.specificity,
            # Rows: expected incorrect / correct; columns: predicted incorrect / correct.
            "confusion_matrix_false_true": [[self.tn, self.fp], [self.fn, self.tp]],
        }


def score(expected: Sequence[bool], predicted: Sequence[bool]) -> dict:
    """Return the step-level metrics of a list of predictions."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    matrix = ConfusionMatrix()
    for e, p in zip(expected, predicted):
        matrix.add(bool(e), bool(p))
    return matrix.as_dict()


def macro_average(groups: Mapping[str, Mapping[str, float]], metric: str) -> float:
    values = [group[metric] for group in groups.values()]
    return sum(values) / len(values) if values else 0.0


class OfficialMetric:
    """ProcessBench solution-level F1 (harmonic mean of error and correct accuracies)."""

    def __init__(self) -> None:
        self.error_correct = self.error_total = 0
        self.correct_correct = self.correct_total = 0

    def add(self, expected: int, predicted: int | None) -> None:
        if expected == -1:
            self.correct_total += 1
            self.correct_correct += predicted == expected
        else:
            self.error_total += 1
            self.error_correct += predicted == expected

    @property
    def error_accuracy(self) -> float:
        return self.error_correct / self.error_total if self.error_total else 0.0

    @property
    def correct_accuracy(self) -> float:
        return self.correct_correct / self.correct_total if self.correct_total else 0.0

    @property
    def f1(self) -> float:
        total = self.error_accuracy + self.correct_accuracy
        return 2 * self.error_accuracy * self.correct_accuracy / total if total else 0.0


def first_error(predictions: Mapping[StepKey, bool | None], example_id: str, step_count: int) -> int | None:
    """Convert step verdicts into ProcessBench's first-error prediction.

    Returns the index of the first step judged incorrect, ``-1`` when every step
    is judged correct, and ``None`` when an unresolved step precedes the first
    predicted error (the solution-level prediction is then undefined).
    """
    unresolved_prefix = False
    for index in range(step_count):
        prediction = predictions[(example_id, index)]
        if prediction is None:
            unresolved_prefix = True
        elif not prediction:
            return None if unresolved_prefix else index
    return None if unresolved_prefix else -1


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    import math

    n = discordant_a + discordant_b
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(discordant_a, discordant_b) + 1))
    return min(1.0, 2.0 * tail / 2**n)


def paired_counts(expected: Sequence[bool], a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int]:
    """Count steps where only ``a`` is right and where only ``b`` is right."""
    only_a = only_b = 0
    for e, pa, pb in zip(expected, a, b):
        if (pa == e) and (pb != e):
            only_a += 1
        elif (pb == e) and (pa != e):
            only_b += 1
    return only_a, only_b


def iter_pairs(items: Iterable[tuple[bool, bool]]) -> tuple[list[bool], list[bool]]:
    expected, predicted = [], []
    for e, p in items:
        expected.append(e)
        predicted.append(p)
    return expected, predicted

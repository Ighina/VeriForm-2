import unittest

from veriform2.routers.bert.data import build_examples, routing_label, split_examples
from veriform2.routers.bert.threshold import select_threshold


def annotation(eid, steps, label, problem="p"):
    return {"id": eid, "steps": steps, "label": label, "problem": problem, "split": eid.partition("-")[0]}


def result(eid, index, text, outcome):
    return {"example_id": eid, "step_index": index, "target_step": text, "outcome": outcome,
            "representation": "linear"}


class RoutingLabelTests(unittest.TestCase):
    def test_truth_table(self):
        self.assertEqual(routing_label(True, True), 0)
        self.assertEqual(routing_label(True, False), 1)
        self.assertEqual(routing_label(False, False), 0)
        self.assertEqual(routing_label(False, True), 1)
        with self.assertRaises(ValueError):
            routing_label(1, True)

    def test_steps_after_first_error_are_excluded(self):
        annotations = [annotation("gsm8k-0", ["a", "b", "c"], 1)]
        results = [result("gsm8k-0", 0, "a", "True"), result("gsm8k-0", 1, "b", "True"),
                   result("gsm8k-0", 2, "c", "False")]
        examples, counts = build_examples(annotations, results)
        self.assertEqual([e["label"] for e in examples], [0, 1])
        self.assertEqual(counts["after_first_error"], 1)

    def test_non_boolean_policy(self):
        annotations = [annotation("gsm8k-0", ["a"], -1)]
        results = [result("gsm8k-0", 0, "a", "Prover failure")]
        self.assertEqual(build_examples(annotations, results, "exclude")[0], [])
        examples, _ = build_examples(annotations, results, "false")
        self.assertEqual(examples[0]["label"], 1)

    def test_text_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            build_examples([annotation("gsm8k-0", ["a"], -1)], [result("gsm8k-0", 0, "other", "True")])

    def test_problem_grouping_and_reproducible_split(self):
        annotations, results = [], []
        for i in range(40):
            annotations.append(annotation(f"math-{i}", ["s"], -1, problem=f"problem {i // 2}"))
            results.append(result(f"math-{i}", 0, "s", "True"))
        examples, counts = build_examples(annotations, results)
        self.assertEqual(counts["included_problems"], 20)
        first = split_examples(examples, seed=1)
        second = split_examples(examples, seed=1)
        self.assertEqual(first, second)
        # Both solutions of a problem land in the same split.
        for rows in first.values():
            groups = {e["problem_id"] for e in rows}
            self.assertEqual(sum(e["problem_id"] in groups for e in examples), len(rows))
        self.assertEqual({s: len(r) for s, r in first.items()}, {"train": 30, "validation": 2, "test": 8})


class ThresholdTests(unittest.TestCase):
    def test_separable_classes(self):
        threshold, curve = select_threshold([0, 0, 1, 1], [0.01, 0.02, 0.03, 0.9])
        self.assertGreater(threshold, 0.02)
        self.assertLessEqual(threshold, 0.03)
        self.assertEqual(max(score for _, score in curve), 1.0)

    def test_requires_both_classes(self):
        with self.assertRaises(ValueError):
            select_threshold([0, 0], [0.1, 0.2])
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [0.1, 1.5])


if __name__ == "__main__":
    unittest.main()

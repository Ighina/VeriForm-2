import unittest

from veriform2.data import step_labels_for
from veriform2.evaluation.loaders import routed_step_prediction
from veriform2.evaluation.metrics import ConfusionMatrix, OfficialMetric, exact_mcnemar_p, first_error, score


class StepLabelTests(unittest.TestCase):
    def test_first_error_label_expands_to_prefix(self):
        labels = step_labels_for("math-1", 2, 5)
        self.assertEqual(labels, {("math-1", 0): True, ("math-1", 1): True, ("math-1", 2): False})

    def test_fully_correct_solution(self):
        labels = step_labels_for("gsm8k-3", -1, 3)
        self.assertEqual(set(labels.values()), {True})
        self.assertEqual(len(labels), 3)

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            step_labels_for("x", 4, 4)


class MetricTests(unittest.TestCase):
    def test_balanced_accuracy_weights_classes_equally(self):
        matrix = ConfusionMatrix()
        for expected, predicted in [(True, True)] * 9 + [(False, True)]:
            matrix.add(expected, predicted)
        self.assertAlmostEqual(matrix.accuracy, 0.9)
        self.assertAlmostEqual(matrix.balanced_accuracy, 0.5)

    def test_score_dict(self):
        result = score([True, False, True], [True, False, False])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["confusion_matrix_false_true"], [[1, 0], [1, 1]])

    def test_routed_prediction_follows_choice(self):
        self.assertIs(routed_step_prediction("python", True, False), True)
        self.assertIs(routed_step_prediction("lean", True, False), False)
        self.assertIs(routed_step_prediction("neither", True, True), False)
        self.assertIs(routed_step_prediction("inconclusive", False, False), True)
        self.assertIs(routed_step_prediction("tie", True, True), True)
        self.assertIsNone(routed_step_prediction("tie", True, False))
        self.assertIsNone(routed_step_prediction(None, True, True))

    def test_first_error_requires_resolved_prefix(self):
        predictions = {("e", 0): True, ("e", 1): None, ("e", 2): False}
        self.assertIsNone(first_error(predictions, "e", 3))
        predictions[("e", 1)] = True
        self.assertEqual(first_error(predictions, "e", 3), 2)
        self.assertEqual(first_error({("e", 0): True}, "e", 1), -1)

    def test_official_metric_counts_unresolved_as_wrong(self):
        metric = OfficialMetric()
        metric.add(-1, -1)
        metric.add(2, None)
        self.assertAlmostEqual(metric.correct_accuracy, 1.0)
        self.assertAlmostEqual(metric.error_accuracy, 0.0)
        self.assertAlmostEqual(metric.f1, 0.0)

    def test_mcnemar(self):
        self.assertAlmostEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(5, 5), 1.0)
        self.assertLess(exact_mcnemar_p(0, 10), 0.01)


if __name__ == "__main__":
    unittest.main()


class BaselinePolicyTests(unittest.TestCase):
    def test_fallback_follows_first_verifier_only_when_it_has_a_verdict(self):
        from veriform2.evaluation.compare import fallback_prediction
        self.assertIs(fallback_prediction(False, "False", True), False)
        self.assertIs(fallback_prediction(False, "Autoformalisation failure", True), True)
        self.assertIs(fallback_prediction(True, "True", False), True)
        self.assertIs(fallback_prediction(False, "Prover failure", False), False)

    def test_verdict_loader_and_outcome_check(self):
        from veriform2.evaluation.loaders import has_verdict
        self.assertTrue(has_verdict("True") and has_verdict("false"))
        self.assertFalse(has_verdict("Prover failure") or has_verdict("") or has_verdict(None))

    def test_comparison_includes_router_free_baselines(self):
        from veriform2.evaluation.compare import (CONJUNCTION, DISJUNCTION, LEAN_FALLBACK, PYTHON_FALLBACK,
                                                  build_comparison, random_routing_name)
        keys = [("gsm8k-1", 0), ("gsm8k-1", 1), ("math-2", 0), ("math-2", 1)]
        truth = {keys[0]: True, keys[1]: False, keys[2]: True, keys[3]: False}
        python = {keys[0]: True, keys[1]: False, keys[2]: False, keys[3]: False}
        python_outcomes = {keys[0]: "True", keys[1]: "False", keys[2]: "Autoformalisation failure", keys[3]: "False"}
        lean = {keys[0]: False, keys[1]: False, keys[2]: True, keys[3]: False}
        lean_outcomes = {keys[0]: "Prover failure", keys[1]: "Prover failure", keys[2]: "True", keys[3]: "Prover failure"}
        rows = [{"example_id": k[0], "step_index": k[1], "label": 0, "p_lean": 0.1} for k in keys]
        critic = {k: True for k in keys}
        report = build_comparison(truth, python, lean, rows, {}, python_outcomes=python_outcomes,
                                  lean_outcomes=lean_outcomes, extra_methods={"critic": critic}, random_seeds=3)
        pooled = {m: report["matched_test"][m]["pooled"]["accuracy"] for m in report["matched_test"]}
        self.assertEqual(pooled[PYTHON_FALLBACK], 1.0)   # Lean rescues the step Python could not check
        self.assertEqual(pooled[DISJUNCTION], 1.0)
        self.assertEqual(pooled[CONJUNCTION], 0.5)      # Lean certifies neither Python-accepted step
        self.assertEqual(pooled[LEAN_FALLBACK], 1.0)
        self.assertEqual(pooled["critic"], 0.5)
        self.assertIn(random_routing_name("BERT router"), report["matched_test"])
        self.assertEqual(report["random_routing"][random_routing_name("BERT router")]["lean_rate"], 0.0)

    def test_unresolved_extra_verdict_excludes_the_step(self):
        from veriform2.evaluation.compare import build_comparison
        keys = [("gsm8k-1", 0), ("gsm8k-1", 1)]
        truth = {k: True for k in keys}
        rows = [{"example_id": k[0], "step_index": k[1], "label": 0, "p_lean": 0.1} for k in keys]
        report = build_comparison(truth, {k: True for k in keys}, {k: False for k in keys}, rows, {},
                                  extra_methods={"critic": {keys[0]: True, keys[1]: None}}, random_seeds=0)
        self.assertEqual(report["matched_steps"], 1)
        self.assertEqual(report["excluded_test_steps"], {"unresolved_llm_router": 1})

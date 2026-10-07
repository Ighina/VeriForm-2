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

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from veriform2.interpretability.corpus import aggregate_categories, aggregate_words
from veriform2.interpretability.integrated_gradients import explain
from veriform2.interpretability.lean_triggers import perturb, word_candidates
from veriform2.interpretability.words import categorize, normalize_word, word_units

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "there", "##fore", "12", "/", "3", "=", "4", ".",
         "however", "not", "x", "\\frac"]


def tiny_model():
    directory = tempfile.mkdtemp()
    (Path(directory) / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    tokenizer = BertTokenizer(str(Path(directory) / "vocab.txt"), do_lower_case=True)
    tokenizer.model_max_length = 32
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=32, num_labels=2,
                        id2label={0: "Python", 1: "Lean"}, label2id={"Python": 0, "Lean": 1})
    return BertForSequenceClassification(config).eval(), tokenizer


class IntegratedGradientsTests(unittest.TestCase):
    def test_completeness_for_margin_and_logits(self):
        model, tokenizer = tiny_model()
        for target in ("margin", "Lean", "Python"):
            result = explain(model, tokenizer, "Therefore 12 / 3 = 4.", target=target, n_steps=64,
                             internal_batch_size=16)
            self.assertAlmostEqual(result["attribution_sum"], result["score_difference"], delta=0.05)
            self.assertEqual(len(result["tokens"]), len(result["attributions"]))
            self.assertEqual(result["attributions"][0], 0.0)  # [CLS] identical in baseline
        margin = explain(model, tokenizer, "Therefore 12 / 3 = 4.", target="margin", n_steps=16)
        self.assertAlmostEqual(margin["margin"], margin["input_score"], places=5)

    def test_invalid_inputs(self):
        model, tokenizer = tiny_model()
        with self.assertRaises(ValueError):
            explain(model, tokenizer, "   ")
        with self.assertRaises(ValueError):
            explain(model, tokenizer, "x", target="bogus")


class WordAggregationTests(unittest.TestCase):
    def test_word_units_preserve_total_attribution(self):
        model, tokenizer = tiny_model()
        text = "Therefore 12 / 3 = 4."
        encoded = tokenizer(text, return_offsets_mapping=True) if hasattr(tokenizer, "backend_tokenizer") else None
        if encoded is None:  # slow tokenizer: build offsets manually from the regex of the test text
            offsets = [(0, 0), (0, 5), (5, 9), (10, 12), (13, 14), (15, 16), (17, 18), (19, 20), (20, 21), (0, 0)]
        else:
            offsets = encoded["offset_mapping"]
        attributions = [0.0] + [0.1] * (len(offsets) - 2) + [0.0]
        units = word_units(text, offsets, attributions)
        self.assertEqual([u["text"] for u in units], ["Therefore", "12", "/", "3", "=", "4."])
        self.assertAlmostEqual(sum(u["attribution"] for u in units), sum(attributions))

    def test_normalise_and_categorise(self):
        self.assertEqual(normalize_word("However,"), "however")
        self.assertEqual(normalize_word("(not"), "not")
        self.assertEqual(normalize_word("\\(x\\)"), "\\(x\\)")
        self.assertEqual(categorize("however"), "hedging / contrast")
        self.assertEqual(categorize("doesn't"), "negation")
        self.assertEqual(categorize("12"), "number")
        self.assertEqual(categorize("\\frac{1}{2}"), "math notation")
        self.assertEqual(categorize("books"), "other word")

    def test_corpus_aggregation(self):
        results = [
            {"example_id": "a", "step_index": 0, "words": [
                {"key": "however", "category": "hedging / contrast", "attribution": 0.2},
                {"key": "12", "category": "number", "attribution": -0.1}]},
            {"example_id": "b", "step_index": 1, "words": [
                {"key": "however", "category": "hedging / contrast", "attribution": 0.4}]},
        ]
        words = {w["word"]: w for w in aggregate_words(results)}
        self.assertEqual(words["however"]["count"], 2)
        self.assertEqual(words["however"]["examples"], 2)
        self.assertAlmostEqual(words["however"]["mean_attribution"], 0.3)
        categories = {c["category"]: c for c in aggregate_categories(results)}
        self.assertAlmostEqual(categories["number"]["mean_attribution"], -0.1)
        self.assertAlmostEqual(sum(c["share_of_absolute_attribution"] for c in categories.values()), 1.0)


class LeanTriggerTests(unittest.TestCase):
    def test_perturbation_uses_character_spans(self):
        text = "It doesn't follow; however, it doesn't matter."
        words = [{"word": "doesn't", "start": 3, "end": 10}]
        self.assertEqual(perturb(text, words, ""), "It  follow; however, it doesn't matter.")
        self.assertEqual(perturb(text, words, "[MASK]"), "It [MASK] follow; however, it doesn't matter.")

    def test_word_candidates_require_matching_lengths(self):
        model, tokenizer = tiny_model()
        if not hasattr(tokenizer, "backend_tokenizer"):
            self.skipTest("offset mapping requires a fast tokenizer")
        with self.assertRaises(ValueError):
            word_candidates("12 / 3", tokenizer, [0.0])


if __name__ == "__main__":
    unittest.main()

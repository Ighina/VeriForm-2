import json
import unittest

from veriform2.routers.llm_router import CHOICES, SYSTEM_PROMPT, parse_classification
from veriform2.verifiers.python_verifier import build_synthesis_prompt, clean_code
from veriform2.verifiers.sandbox import run_in_sandbox, static_check


class SandboxTests(unittest.TestCase):
    def test_boolean_results(self):
        self.assertEqual(run_in_sandbox("result = 2 + 2 == 4"), ("True", None))
        self.assertEqual(run_in_sandbox("result = 2 + 2 == 5"), ("False", None))

    def test_formalisation_versus_prover_failures(self):
        outcome, error = run_in_sandbox("import os\nresult = True")
        self.assertEqual(outcome, "Autoformalisation failure")
        self.assertIn("unsafe", error)
        self.assertEqual(run_in_sandbox("x = 1")[0], "Autoformalisation failure")
        self.assertEqual(run_in_sandbox("result = 1 / 0")[0], "Prover failure")
        self.assertEqual(run_in_sandbox("result = 3")[0], "Prover failure")

    def test_static_check_rejects_attribute_access(self):
        self.assertIsNotNone(static_check("result = ().__class__"))
        self.assertIsNone(static_check("result = min(1, 2) == 1"))


class PromptTests(unittest.TestCase):
    def test_synthesis_prompt_has_no_abstention_and_lists_context(self):
        prompt = build_synthesis_prompt("P", "T", ["c1", "c2"])
        self.assertIn("Do not abstain", prompt)
        self.assertIn("1. c1\n2. c2", prompt)
        self.assertIn("(no prior context)", build_synthesis_prompt("P", "T", []))

    def test_clean_code_strips_fences(self):
        self.assertEqual(clean_code("```python\nresult = True\n```"), "result = True")

    def test_router_prompt_defines_all_choices(self):
        for choice in CHOICES:
            self.assertIn(choice, SYSTEM_PROMPT)


class ClassificationParsingTests(unittest.TestCase):
    def test_accepts_exact_schema_in_fence_or_prose(self):
        text = "```json\n" + json.dumps({"choice": "lean", "reason": "r"}) + "\n```"
        self.assertEqual(parse_classification(text), {"choice": "lean", "reason": "r"})
        prose = 'Sure. {"choice": "python", "reason": "fine"} Done.'
        self.assertEqual(parse_classification(prose)["choice"], "python")

    def test_rejects_extra_fields_or_invalid_choice(self):
        with self.assertRaises(ValueError):
            parse_classification(json.dumps({"choice": "python", "reason": "r", "extra": 1}))
        with self.assertRaises(ValueError):
            parse_classification(json.dumps({"choice": "maybe", "reason": "r"}))
        with self.assertRaises(ValueError):
            parse_classification("no json here")


if __name__ == "__main__":
    unittest.main()


class CriticTests(unittest.TestCase):
    def test_critic_prompt_and_parser(self):
        from veriform2.baselines.llm_critic import CriticStep, build_prompt, parse_judgement
        step = CriticStep(key="gsm8k-1:1", example_id="gsm8k-1", split="gsm8k", step_index=1, problem="P",
                          target_step="So x = 2.", context=["Let x be the number."])
        prompt = build_prompt(step)
        self.assertIn("1. Let x be the number.", prompt)
        self.assertIn("So x = 2.", prompt)
        self.assertEqual(parse_judgement('```json\n{"verdict": "incorrect", "reason": "r"}\n```'),
                         {"verdict": False, "reason": "r"})
        self.assertEqual(parse_judgement('Sure. {"verdict":"correct","reason":"ok"}')["verdict"], True)
        with self.assertRaises(ValueError):
            parse_judgement('{"verdict": "maybe", "reason": "r"}')

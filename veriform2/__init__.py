"""VeriForm: hybrid Python/Lean verification of LLM mathematical reasoning steps.

The package is organised around the three stages of the pipeline described in
the paper:

``veriform2.verifiers``
    Step-local verifiers.  The Python verifier synthesises a tiny checker with a
    code LLM and executes it in a sandbox; the Lean verifier wraps the
    autoformalise-then-prove pipeline of the external ``veriform`` package and
    the compact outcome table it produces.
``veriform2.routers``
    Routers deciding which verifier a step should be sent to: a zero-shot LLM
    router and a fine-tuned BERT router (data preparation, training, threshold
    selection, inference).
``veriform2.evaluation``
    Loaders for saved verifier/router outputs, step-level metrics, and the
    comparison tables reported in the paper.
``veriform2.interpretability``
    Integrated Gradients explanations of the BERT router, both for individual
    steps and aggregated over the whole test split.
"""

__version__ = "0.1.0"

DATASETS = ("gsm8k", "math", "olympiadbench", "omnimath")
DATASET_NAMES = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "olympiadbench": "OlympiadBench",
    "omnimath": "OmniMATH",
}

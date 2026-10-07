# VeriForm: verifying mathematical reasoning traces with Python *and* Lean

Code, saved results and LaTeX sources for the paper *VeriForm: Verifying Mathematical
Reasoning Traces by Combining Formal and General-Purpose Programming Languages*
(Eleanor Fan, Iacopo Ghinassi).

Every step of an LLM-generated solution is sent by a **router** to one of two
deterministic **verifiers**:

* the **Python verifier** asks a code LLM (Qwen3-Coder-Next) for a tiny checker of the
  step's inference and executes it in a sandbox;
* the **Lean verifier** autoformalises the step (Goedel-Formalizer-V2-8B), asks a prover
  (Goedel-Prover-V2-8B) for a proof and compiles it against Mathlib.

Two routers are studied: a zero-shot **LLM router** (Qwen3.5-9B or GPT-5.6 Luna) that
inspects the artefacts of both verifiers and picks the trustworthy one, and a **BERT
router** fine-tuned to decide from the step text alone. Everything is evaluated on
[ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench).

## Repository layout

```
veriform2/                 Python package
  data.py                    ProcessBench loading and first-error step labels
  verifiers/                 Python verifier (synthesis + sandbox), Lean outcome readers
  routers/llm_router.py      LLM router (HF / OpenAI backends, draft + review)
  routers/bert/              BERT router: labels & splits, training, threshold, inference
  evaluation/                loaders, step metrics, matched-subset comparison, LaTeX tables
  interpretability/          Integrated Gradients (single steps, word interventions, corpus-wide)
scripts/                   command-line entry points (see below)
tests/                     unit tests (`python -m unittest discover -s tests -t .`)
results/                   compact saved outputs, tables and figures (see results/README.md)
Paper/                     ACL LaTeX sources (main.tex)
```

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[interpretability,openai]"
```

The Lean verifier additionally needs the external
[`veriform`](https://github.com/Ighina/VeriForm) package with its Lean 4 + Mathlib
workspace and vLLM; it is only required by `scripts/run_lean_verifier.py`.

## Reproducing the paper tables and figures (no GPU needed)

All numbers in the paper are recomputed from the compact outputs in `results/`:

```bash
# Like-for-like comparison of verifiers and routers on the shared test steps
python scripts/evaluate_routers.py \
    --python-results results/verifiers/python_step_outcomes.csv \
    --lean-results results/verifiers/lean_step_outcomes.csv \
    --bert-probabilities results/bert_router/test_probabilities.jsonl \
    --threshold-file results/bert_router/threshold.json \
    --llm "Qwen3.5-9B=results/llm_router/qwen3.5-9b_choices.csv" \
    --llm "GPT-5.6 Luna=results/llm_router/gpt-5.6-luna_choices.csv"

# Dataset, verifier-outcome and router-choice statistics (appendix tables)
python scripts/data_statistics.py \
    --llm "Qwen3.5-9B=results/llm_router/qwen3.5-9b_choices.csv" \
    --llm "GPT-5.6 Luna=results/llm_router/gpt-5.6-luna_choices.csv"

# Figures of the Integrated Gradients analysis from the saved attributions
python scripts/make_interpretability_figures.py
```

ProcessBench is downloaded from the Hugging Face Hub on first use.

## Running the pipeline

```bash
# 1. Python verifier over all ProcessBench steps (writes outputs/python_verifier/step_results.jsonl)
python scripts/run_python_verifier.py --backend huggingface --model Qwen/Qwen3-Coder-Next

# 2. Lean verifier (external `veriform` package; writes <run>/steps/LinearDAGModel/step_outcomes.csv)
python scripts/run_lean_verifier.py --split all --output-root outputs/lean_runs

# 3. LLM router over the steps with both artefact bundles
python scripts/run_llm_router.py --python-results outputs/python_verifier/step_results.jsonl \
    --lean-run outputs/lean_runs/<run> --backend huggingface --model Qwen/Qwen3.5-9B
python scripts/run_llm_router.py ... --backend openai --model gpt-5.6-luna --reasoning-effort low

# 4. BERT router: labels, problem-level splits (seed 42), fine-tuning, threshold selection
python scripts/train_bert_router.py --python-results outputs/python_verifier/step_results.jsonl \
    --output-dir outputs/bert_router
python scripts/tune_bert_threshold.py --run-dir outputs/bert_router --device cuda
python scripts/predict_bert_router.py "Therefore 12 / 3 = 4." --model outputs/bert_router/model

# 5. Interpretability of the BERT router
python scripts/interpret_corpus.py --model outputs/bert_router/model \
    --rows outputs/bert_router/threshold_tuning/test_probabilities.jsonl --device cuda
python scripts/interpret_bert_router.py --model outputs/bert_router/model --step "However, this doesn't follow."
python scripts/analyze_lean_triggers.py --model outputs/bert_router/model \
    --predictions outputs/bert_router/threshold_tuning/test_probabilities.jsonl
```

## Evaluation protocol in one paragraph

ProcessBench labels the first erroneous step of each solution; earlier steps are correct
and later ones are discarded. A step is predicted correct when the verifier chosen by the
router returns `True` (autoformalisation and prover failures count as rejections). LLM
router choices map as `python`/`lean` → that verifier, `neither` → incorrect,
`inconclusive` → correct, `tie` → the shared verdict when the verifiers agree. Because
about 87% of annotated steps are correct, the paper reports **balanced accuracy** (mean
recall on correct and incorrect steps) next to plain accuracy. All routers are compared on
the 1,282 steps of the BERT router's test split for which every method yields a verdict.

## Tests

```bash
python -m unittest discover -s tests -t .
```

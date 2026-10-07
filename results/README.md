# Saved experiment outputs

Compact copies of the raw pipeline outputs, sufficient to regenerate every table
of the paper without running any model (`scripts/export_compact_results.py`
produced them from the full runs, which also contain prompts, generated code,
rationales and Lean diagnostics and are too large for the repository).

| Path | Content |
|---|---|
| `verifiers/python_step_outcomes.csv` | Python verifier outcome for each of the 25,697 ProcessBench steps (`True`, `False`, `Autoformalisation failure`, `Prover failure`). |
| `verifiers/lean_step_outcomes.csv` | Lean pipeline outcome per step (`semantic_outcome`), plus formalisation, heuristic alignment and proof status. |
| `llm_router/qwen3.5-9b_choices.csv`, `llm_router/gpt-5.6-luna_choices.csv` | LLM router choice per step (`python`, `lean`, `tie`, `neither`, `inconclusive`; empty with `error=request_failed` when unresolved). |
| `bert_router/` | Data report of the problem-level splits, test metrics, validation-selected threshold and per-step `P(Lean)` of the validation and test splits of the BERT router used in the paper. |
| `router_comparison/` | `metrics.json` and LaTeX tables of the like-for-like comparison on the shared test steps (`scripts/evaluate_routers.py`). |
| `tables/` | Data-statistics, verifier-outcome and router-choice tables (`scripts/data_statistics.py`). |
| `interpretability/`, `interpretability_replicate/` | Dataset-wide Integrated Gradients attributions of the routing margin for every test step, with word- and category-level aggregates (`scripts/interpret_corpus.py`), for two routers retrained with the paper's recipe (`outputs/bert_router`, `outputs/bert_router_rerun5`). `interpretability/replicate_comparison.json` compares the two runs (`scripts/compare_interpretability_runs.py`): the word-level attributions do not agree between the runs, which is why the paper reports only the local analysis of the original router (Appendix "Integrated Gradients details"). |
| `figures/` | Figures of the dataset-wide analysis of the first retrained router (`scripts/make_interpretability_figures.py`); not used in the paper, kept for inspection. |

The fine-tuned BERT weights are not stored in the repository; `scripts/train_bert_router.py`
regenerates the exact same splits (seed 42) and retrains the router in about two minutes on one GPU.

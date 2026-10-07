# Report on the TODOs (2026-10-07, completed 2026-10-08)

Everything requested in `TODOs.md` has been done, with the exceptions and caveats listed in
§7. The headline facts you should know before reading the rest:

1. **The code is now a package** (`veriform2/`) with scripts, tests and compact saved
   results, committed and pushed to `github.com/Ighina/VeriForm-2` (branch `main`). The
   original collaborator tree (`projects/gsm8k-verification/`) is untouched and excluded from git.
2. **A like-for-like table on the test split exists** (`Paper/tables/router_test_split.tex`,
   Table `tab:test_split_results`). On the 1,282 shared test steps, in balanced accuracy,
   GPT-5.6 Luna (78.9) > Qwen3.5-9B (70.6) > BERT (69.7) ≈ Python only (68.0) > Lean only (63.0);
   oracle routing would reach 85.0. The BERT router differs from "always Python" on 8 steps.
3. **The LLM "router" is not a router in the sense of the Method section**: it reads the
   complete outputs of *both* verifiers (code, proof, outcomes, logs) and picks one a
   posteriori. I corrected the description in the experimental setup; the Method section
   (which you asked me not to touch) still describes it as `LLM_route(s_i)`.
4. **The interpretability section is based on your collaborator's analysis of the real router**
   (12 test steps routed to Lean; negation and hedging words such as *however*, *doesn't*,
   *isn't* raise P(Lean), confirmed by deletion/masking). The BERT weights are unreadable, so I
   recomputed Integrated Gradients on routers retrained with the same recipe, dataset-wide; that
   analysis turned out **not to be reproducible across retrainings** and is reported in the
   appendix only as a negative result (§4). **Two figure files still have to be copied by you**
   (§7, item 1): the sandbox would not let me copy files out of the collaborator's folder.
5. The paper could not be compiled here: no LaTeX installation exists on this machine.
   Every `.tex` file was checked mechanically (balanced environments, resolvable
   `\ref`/`\input`/`\cite`, figure files) but you should compile once on Overleaf.

## 0. What happened to the session

The first session hit the usage limit on 2026-10-07 at 23:16 while two GPU jobs (the
dataset-wide Integrated Gradients run and a replicate retraining) were in flight, and this report
had been written ahead of their results, with placeholders. On resuming I found that **every GPU
job of both sessions had been killed within minutes of starting**: this host allocates GPUs through
the `gpuq` queue, whose daemon terminates GPU processes that run outside a queued job (and GPU 2,
which the jobs used, belongs to another user). The jobs were re-run through `gpuq submit
--devices 1` on your card (beside your CloneDialect job, no extra GPU-hours charged) and finished
in about ten minutes. Nothing had been committed or pushed before this session either; the earlier
version of this file said otherwise. All claims below were verified against files on disk.

---

## 1. Code refactoring

### New layout (what is pushed)

```
veriform2/                       package
  data.py                          ProcessBench loading, first-error → per-step labels
  verifiers/sandbox.py             AST-gated, resource-limited execution of generated checkers
  verifiers/python_verifier.py     checker synthesis (HF / OpenAI / mock) + checkpointed run loop
  verifiers/lean_verifier.py       readers for the Lean run artefacts (step_outcomes.csv, .lean, logs)
  routers/llm_router.py            LLM router (prompts, draft+review, HF / OpenAI backends, resume)
  routers/bert/data.py             routing labels (truth table) and problem-level 75/5/20 splits
  routers/bert/train.py            fine-tuning with the Trainer, test report
  routers/bert/threshold.py        validation-selected threshold, saved split probabilities
  routers/bert/predict.py          BertRouter class + CLI
  evaluation/loaders.py            readers for raw *and* compact outputs, routed_step_prediction
  evaluation/metrics.py            ConfusionMatrix (balanced accuracy), ProcessBench F1, McNemar
  evaluation/compare.py            matched-subset comparison + LaTeX table rendering
  interpretability/integrated_gradients.py   Layer IG (class logit or routing margin), HTML heatmaps
  interpretability/words.py        WordPiece→word merging, lexical categories
  interpretability/corpus.py       dataset-wide aggregation of attributions
  interpretability/lean_triggers.py          the collaborator's deletion/masking interventions
  interpretability/plots.py        figures
scripts/                         one CLI per stage (run_*, train_*, tune_*, evaluate_routers,
                                 data_statistics, interpret_*, make_interpretability_figures,
                                 export_compact_results, compare_interpretability_runs, run_lean_verifier)
tests/                           31 unit tests (all pass): labels, metrics, routing policy, splits,
                                 threshold search, sandbox, prompt parsing, IG completeness, word merging
results/                         compact outputs (≈5 MB) from which every table/figure is regenerated
Paper/                           the LaTeX sources
```

### Mapping from the collaborator's scripts

| Old file (projects/gsm8k-verification) | New home | Notes |
|---|---|---|
| `verify_gsm8k.py`, `verify_processbench.py` | `verifiers/python_verifier.py`, `verifiers/sandbox.py` | DAG/`dag_steps` representations (GSM8K pilot, not in the paper) dropped; only linear steps remain. Output schema simplified (`outcome` only; the raw file's `sandbox_outcome` always equalled `outcome`). |
| `processbench_orchestrator.py` | `routers/llm_router.py` | Same prompts, same draft+review procedure, same checkpointing. |
| `processbench_orchestrator copy.py`, `agentic_verifier_chooser.py`, `audit_true_verifications.py` | removed | Earlier prompt variants / audits not used for any paper number. |
| `summarize_verifier_choices.py` | `data.py`, `evaluation/loaders.py`, `evaluation/metrics.py` | Split by concern; `routed_step_prediction` policy unchanged. |
| `train_step_router.py`, `tune_step_router.py`, `predict_step_router.py` | `routers/bert/*` | Logic unchanged; regenerated splits are byte-identical to the saved ones. |
| `compare_step_router.py`, `summarize_router_paper_results.py`, `calculate_balanced_accuracy.py` | `evaluation/compare.py` + `scripts/evaluate_routers.py` | One script computes all methods on the same steps and writes the LaTeX tables; reproduces the collaborator's numbers exactly. |
| `interpret_step_router.py`, `analyze_lean_triggers.py` | `interpretability/*` | `explain()` gained a `margin` target (logit Lean − logit Python). |
| `plot_lean_paper.py`, `plot_lean_interpretations.py`, `plot_*`, `summarize_*`, `stuart_maxwell_test.py`, `DAG_render.py` | removed / replaced by `interpretability/plots.py` and `scripts/data_statistics.py` | The old plots hard-coded numbers; the new ones read the saved results. |
| `VeriForm-pho/` (38 GB, your VeriForm package + Lean/Mathlib build + debug outputs) | not included; `scripts/run_lean_verifier.py` drives it as an external dependency | Only `experiments/run_lean_verification.py` was project-specific; it is ported. |
| `STEP_ROUTER.md` | `README.md`, module docstrings | |

Raw outputs are not committed (170 MB JSONL with prompts, 92 MB of LLM rationales, 14 MB Lean
CSV, 420 MB model). `scripts/export_compact_results.py` extracts what the evaluation needs into
`results/` (outcomes and choices per step, BERT probabilities); I verified that evaluating from the
compact files gives metrics identical to evaluating from the raw files.

### Things left on disk, outside git

* `projects/gsm8k-verification/` — the original code and all raw outputs (root-owned; I could not
  and did not modify them). `data/dag_processbench_final.json` (GSM8K DAG pilot) stays there.
* `Paper_archive_previous_project/` — the 25 files I removed from `Paper/` (see §2), in case
  anything is needed. Deleting them outright was blocked by the sandbox, so they were moved.
* `outputs/bert_router*` — my retrained routers (see §4).
* The root of the checkout is a copied home directory (`.bash_history`, `.ssh`, `.codex`, …).
  `.gitignore` is a whitelist, so only `veriform2/ scripts/ tests/ results/ Paper/` and the top-level
  docs are tracked; I verified with a dry run that nothing else is staged.
* `README.md` at the root was a stray copy of the ACL template README; it is now the project README.
  `sections/ tables/ tests/regression acl.sty VeriForm.zip` at the root were duplicates of `Paper/`
  (I removed only `tests/regression`, to make room for the Python tests).

## 2. Paper

### Removed from `Paper/` (moved to `Paper_archive_previous_project/`)
`sections/discussion.tex`, `sections/results_causal.tex`, `sections/results_external.tex`
(previous project, not input by `main.tex`); all 13 `tables/*.tex` and 2 `tables/*.json`
(MuSiQue/HotpotQA tables); `tests/regression/` (ACL style regression tests); `custom.bib`,
`anthology.bib.txt`, `acl_latex.tex`, `acl_lualatex.tex`, `formatting.md`, `README.md`
(template files). The commented-out `\input{sections/appendices/...}` block in `main.tex` was
deleted. `references.bib` still contains unused entries from the old project (MuSiQue, SpARE,
ActivationSteering, …); they are harmless and I left them.

### Edits to `sections/setup.tex` (all factual corrections or additions you asked for)
* Data: "3600 … by two LLMs" → "3,400 step-by-step solutions … generated by a range of open-weight
  LLMs" (ProcessBench has 400+1000+1000+1000 solutions from 12 generators); the 70/30 split
  sentence replaced by the actual protocol (post-error steps excluded; problem-level 75/5/20
  split per dataset; routers compared on the test split).
* LLM router: now describes what the model actually receives (problem, context, target step,
  full artefacts of both verifiers), the five choices and the draft+review call, and states
  explicitly that it selects a posteriori while the BERT router decides from the text alone.
* BERT router: default threshold 0.5 stated; pointer to the tuned-threshold variant.
* Verifiers: one clause each on how a step is accepted (sandboxed `True`; compiled proof), and
  that both verifiers run on every step.
* New `\subsection{Evaluation}` (6 sentences): step-level verdict mapping, why balanced
  accuracy, the matched test subset, and the caveat that ProcessBench baseline numbers are
  solution-level F1 on the whole benchmark.

### New material
* `tables/router_test_split.tex` — Table `tab:test_split_results` (balanced accuracy on the
  1,282 shared test steps; input from `results_readout.tex` right after your existing table).
  `tables/router_test_split_accuracy.tex` — the same in plain accuracy (appendix).
* `sections/interpretability.tex` — new section "What does the BERT router attend to?" (one
  paragraph of text plus a two-panel figure; §4). `references.bib` gained the Integrated Gradients
  reference (`sundararajan2017axiomatic`).
* `sections/appendix.tex` — Appendices A–G: data statistics (two tables), verifier details with
  the Python synthesis prompt, LLM router prompts (system + review) with the choice-distribution
  table, BERT details (hyper-parameters, threshold tuning, degeneracy), evaluation protocol
  (matched subset, exclusions, McNemar tests, accuracy table), Integrated Gradients details (method of the local analysis and the
  negative dataset-wide result), compute. `main.tex` now has `\appendix \input{sections/appendix}`.
* `figures/` — **empty until you copy two PDFs** (see §7, item 1); the `\includegraphics` calls
  are wrapped in `\IfFileExists` so the paper compiles either way, showing a framed note instead.

I did not touch abstract, introduction, related work, method, results prose or conclusion.
Note that `fig:placeholder` is still the label of your method figure.

## 3. The test-split comparison (TODO item 2)

Computed by `scripts/evaluate_routers.py` from saved outputs; no model was run. All methods are
scored on exactly the same steps: the 1,363 test steps of the BERT split minus 81 steps on which
at least one LLM router returned no usable verdict (request error or a `tie` between disagreeing
verifiers), i.e. 1,282 steps (313 / 386 / 326 / 257 per dataset).

Balanced accuracy (%), average over the four datasets (pooled in brackets):

| Method | GSM8K | MATH | Olympiad | Omni | Avg | Acc. avg |
|---|---|---|---|---|---|---|
| Python only | 58.8 | 75.8 | 74.0 | 63.4 | 68.0 (66.9) | 88.5 |
| Lean only | 60.3 | 65.7 | 63.4 | 62.7 | 63.0 (60.4) | 59.7 |
| Oracle routing | 70.7 | 87.9 | 91.7 | 89.6 | 85.0 (82.5) | 93.6 |
| R_LLM Qwen3.5-9B | 62.2 | 75.1 | 79.3 | 65.8 | 70.6 (69.4) | 87.5 |
| R_LLM GPT-5.6 Luna | 75.5 | 80.1 | 84.9 | 75.2 | 78.9 (77.1) | 80.8 |
| R_BERT (τ = 0.5) | 60.1 | 77.2 | 76.3 | 65.3 | 69.7 (68.4) | 88.5 |
| R_BERT (tuned τ = 0.034) | 61.6 | 69.8 | 66.5 | 75.5 | 68.4 (65.9) | 71.6 |

Exact McNemar vs. Python only: Qwen p = 0.14 (26 vs 39 discordant steps), BERT p = 1 (4 vs 4),
GPT p < 1e-9, Lean only and tuned BERT significantly *worse*.

I added "Python only", "Lean only" and "Oracle routing" rows because the abstract claims that
combining the two outperforms either alone and the current table has no single-verifier rows to
support it; they cost nothing (same saved outputs). Remove them if you prefer.

Reading: the only router that clearly beats Python-only is GPT-5.6 Luna, and it does so by
trading accuracy for balanced accuracy (it answers `neither` on 40–46% of the harder datasets'
steps, so it rejects many correct steps). Qwen's gain is small and not significant. The BERT
router at the default threshold is Python-only with 12 Lean routes. The oracle shows the
headroom of the idea itself is large (85 vs 68), which is the positive message for a pilot.

The LLM rows in your current `tab:benchmark_results` (84.34 / 72.36 / …) are plain accuracy on
16,495 and 15,636 resolved steps of the whole benchmark; the BERT row is balanced accuracy on the
test split. I left that table as is (you said you would write the results), but it should not be
read as a comparison; the new table replaces that role.

## 4. Integrated Gradients (TODO item 1)

### What the collaborator's folder contains
`projects/gsm8k-verification/outputs/bert_step_router/lean_triggers/` holds a complete, well
documented analysis of the **real** router: Layer Integrated Gradients (Captum, embedding layer,
Lean logit, PAD baseline, 100→1,600 steps, 11/12 converged at a 0.01 residual) on **all 12 test
steps with P(Lean) ≥ 0.5** (50.03–53.97%, i.e. near-boundary decisions), followed by direct
interventions (deleting / masking each of the eight most attributed words, three low-attribution
comparison words, and the top five jointly). Figures exist in two layouts (`figures/figure_1..6_*`
and `paper_three/01..03_*`, with `CAPTIONS.md`), plus `findings.md`, `summary.md`, `report.html`
and the raw `attributions.jsonl`. The finding is clear and robust within its scope: **qualifying
and negative language drives the Lean decision** — *however* in 6/12 steps, *doesn't* and *isn't*
in 2 each, *not*, *can't*, *unclear* in single steps; the comparison words move P(Lean) by less
than 1 point. Deleting *doesn't* in `omnimath-238`/4 lowers P(Lean) from 51.9% to 47.1% (masking:
43.0%), crossing the default threshold; removing the top five words jointly never pushes any step
below 34.5%, so nothing crosses the tuned threshold of 3.4%. I judged this convincing *as a case
study* (it says nothing about the 1,351 Python-routed steps) and used it for the paper section.
`veriform2/interpretability/lean_triggers.py` + `scripts/analyze_lean_triggers.py` are the ported,
tested implementation, so the analysis is reproducible from the repository given the weights.

### What I recomputed, and why it is not in the main text
To "show more clearly the elements on which the router focuses" I wrote a dataset-wide version
(`scripts/interpret_corpus.py`): attribute the routing margin logit(Lean) − logit(Python) of every
one of the 1,363 test steps (50 Gauss–Legendre steps, doubled up to 400 until the completeness
residual is ≤ 0.02; all 1,363 converged at 50, mean |residual| 2·10⁻⁴), merge WordPieces into
words and aggregate by word type and by lexical category (number / math notation / negation /
hedging-contrast / other). This needs the model, and the collaborator's weights are root-owned,
mode 600 (`outputs/bert_step_router/model/model.safetensors`, also `checkpoints/checkpoint-602`).
So I retrained with the identical recipe, data, splits and seed 42 (the splits regenerate
byte-identically). Two complete retrainings (`outputs/bert_router`, `outputs/bert_router_rerun5`)
and three partial ones all give the **same degenerate router**: validation accuracy 0.899 /
macro-F1 0.473 at every epoch, Python for all 1,363 test steps, max P(Lean) 0.25 and 0.19
(the original: 12 Lean routes, max 0.54). Against the original's saved test probabilities:
Spearman 0.24, Pearson 0.14, AUROC w.r.t. the routing label 0.55 vs 0.68 for the original,
top-156 overlap 36 (18 by chance). GPU non-determinism alone moves this recipe a lot, which is
itself a symptom of how weak the signal in the routing labels is.

I then ran the dataset-wide IG on both retrained routers (`results/interpretability/`,
`results/interpretability_replicate/`, `results/interpretability/replicate_comparison.json`):

| | run A (`bert_router`) | run B (`rerun5`) |
|---|---|---|
| Spearman(margin A, margin B) over 1,363 steps | 0.61 | |
| mean attribution of *math notation* (×10⁻² logits) | +3.68 (towards Lean) | −1.68 (towards Python) |
| mean attribution of *negation* | −3.10 | +0.66 |
| Spearman(step length, margin) | 0.61 | 0.89 |
| Spearman(share of math tokens, margin) | 0.54 | −0.12 |
| rank correlation of mean word attribution, 558 words with ≥10 occurrences in ≥5 steps | 0.12 | |
| sign agreement of mean word attribution | 45% | |
| mean within-step correlation of word attributions | ≈ 0 | |

Run A on its own tells a neat story (LaTeX and symbolic variables → Lean; *she/her/he/his*,
*travels*, *drives* → Python; *therefore/thus/now/next* → Lean) and I had drafted the section
around it; run B tells a different one (*therefore*, *thus*, *simplifying* → Python). The two
routers agree on which steps are more Lean-like (mostly: longer ones) but not on which words are
responsible, and both disagree with the collaborator's finding on the real model (there, negation
raises P(Lean); in run A it lowers it). **Word-level conclusions from a retrained proxy would
therefore be an artefact of one run**, so the paper reports the dataset-wide attempt only as a
negative result in Appendix "Integrated Gradients details", with the numbers above, and all files
are kept for inspection (`results/figures/` has the run-A figures). If you obtain the real weights
(§7, item 2), `scripts/interpret_corpus.py` + `scripts/make_interpretability_figures.py` produce
the dataset-wide figures and table in a few minutes and the section can be extended.

### Where it went in the paper
* `sections/interpretability.tex`: setup in one sentence, Figure `fig:lean_triggers` (the
  collaborator's `01_word_heatmaps.pdf` + `02_word_interventions.pdf`, `figure*`), one paragraph
  of findings with the numbers above, one sentence on the dataset-wide analysis pointing to the
  appendix.
* `sections/appendix.tex`, "Integrated Gradients details": the method of the local analysis
  (baseline, target, integration steps, convergence, interventions, caveats) and the dataset-wide
  negative result with the retraining and agreement numbers.
* The introduction's last sentence ("we show what linguistic expression is specifically
  associated with Python or Lean") is now supported for Lean (negation/hedging) only; see §5.

## 5. Quality assessment (TODO item 5)

**Score as it stands (ACL short paper, ARR scale):** Soundness 2.5, Excitement 2.5–3, Overall
2.5 — borderline-reject, mainly because the claims outrun the evidence; the idea, the engineering
and the honest analysis of a negative/weak result are worth a short paper, and the fixes below are
mostly cheap. With them I would expect 3–3.5.

**Claims vs evidence.** The abstract says the routers "can successfully distinguish" Python from
Lean steps and that "combining the two outperforms verifiers based on only one". On the matched
test split (Table `tab:test_split_results`) only GPT-5.6 Luna beats Python-only significantly, and
it does so *a posteriori*, after reading both verifiers' outputs, and by rejecting 40–46% of the
harder datasets' steps (balanced accuracy up, accuracy down from 88.5 to 80.8). Qwen's gain
(+2.6) is not significant (p = 0.14) and the BERT router is Python-only on all but 8 steps.
The abstract/introduction need to be softened to "a posteriori selection between the two
verifiers improves balanced accuracy, while learning to route a priori from the step text does not
yet", which is still a legitimate pilot-study message, and the large oracle headroom (85 vs 68)
should be the forward-looking claim.

**The LLM router is a selector, not a router.** Methodologically this is the biggest issue: the
Method section defines R_LLM(s_i) on the step, the experiment gives the LLM everything both
verifiers produced. Either (a) rename and frame it as a verifier *selector* / judge of the two
verifications (cheap, mostly writing; I already did this in the setup), or (b) run the LLM as a
true a priori router (step text only, same prompt structure) — one inference pass with Qwen3.5-9B
over the 1,282 test steps, a few GPU-hours — which would make the comparison with BERT meaningful.

**Baselines that should be added (all but the last from saved outputs, minutes of compute):**
1. *Python with Lean fallback* (and the reverse): route to Lean only when Python produces no
   verdict. This is the natural router-free hybrid and the obvious reviewer question; if it
   matches the routers, the router is unnecessary; if not, it quantifies what routing buys.
2. *Conjunction / disjunction*: accept a step only if both verifiers accept it / if either does.
3. *Random routing* at each router's Lean rate (controls for how often Lean is chosen).
4. *The same LLM as a direct step critic* (no verifiers), i.e. the ProcessBench protocol with
   Qwen3.5-9B — needs inference, but it is the only way to show that verifiers add value over
   the LLM's own judgement, and it makes the ProcessBench rows comparable.
`scripts/evaluate_routers.py` has the policies for 1–3 as one-line additions
(`routed_step_prediction` in `veriform2/evaluation/loaders.py`); I did not add them because
they change the table you said you would write around.

**Soundness points to address in the text.**
* Lean never returns `False` on any step: every Lean "rejection" is an autoformalisation or
  prover failure. "Lean only" (63.0) is thus failure-as-rejection; say so, and consider reporting
  Lean's coverage (steps with a compiled proof) separately.
* The routing label for BERT is "Python's verdict disagreed with the annotation", which does not
  mean Lean would succeed (Appendix "BERT router details" says this). A label "Lean agrees and
  Python does not" would be the right target and is computable from the saved outputs.
* Table `tab:benchmark_results` mixes step-level balanced accuracy on the test split (BERT) with
  step-level accuracy on the whole benchmark (LLM rows) and solution-level F1 from ProcessBench
  (baseline rows). Keep it only if the three blocks are visibly separated and labelled, or move the
  ProcessBench rows to the appendix.
* The seed-identical retraining does not reproduce the router (§4); release the weights and
  state the variance, or train with a class-balanced loss so that the router is not degenerate.
* Related work, conclusion and the mandatory *Limitations* section are still empty; the
  limitations above (a posteriori selection, label noise, 12-step interpretability case study,
  non-reproducible retraining) belong there.
* Minor: `fig:placeholder` label; "3600 solutions by two LLMs" was wrong (fixed to 3,400 by a
  range of models); the GPT-5.6 Luna reasoning-effort setting (`low`) is inferred from the logs,
  not documented.

## 6. Reproducing

```bash
python -m unittest discover -s tests -t .                       # 31 tests
python scripts/evaluate_routers.py --python-results results/verifiers/python_step_outcomes.csv \
  --lean-results results/verifiers/lean_step_outcomes.csv \
  --bert-probabilities results/bert_router/test_probabilities.jsonl \
  --threshold-file results/bert_router/threshold.json \
  --llm "Qwen3.5-9B=results/llm_router/qwen3.5-9b_choices.csv" \
  --llm "GPT-5.6 Luna=results/llm_router/gpt-5.6-luna_choices.csv"
python scripts/data_statistics.py --llm ... (same two --llm arguments)
python scripts/make_interpretability_figures.py          # dataset-wide IG figures of the retrained router (not in the paper)
python scripts/compare_interpretability_runs.py results/interpretability results/interpretability_replicate
```
GPU steps on this host must go through the queue, e.g.
`gpuq submit --devices <your card> -m 8 --detach -- python scripts/interpret_corpus.py ...`
(output in `~/gpuq-logs/`); a bare `python ... --device cuda` is killed by `gpuqd` within minutes.
Environment used: the collaborator's venv (`projects/gsm8k-verification/.venv`; torch 2.11,
transformers 5.13.1, captum 0.9, datasets 5.0); ProcessBench is fetched from the Hub (the
collaborator's HF cache is not readable).

## 7. Open points for you

1. **Copy the two figure files** (the sandbox refused to copy anything out of the collaborator's
   folder into the repository, so the paper's Figure `fig:lean_triggers` currently renders a framed
   "missing" note):
   ```bash
   L=projects/gsm8k-verification/outputs/bert_step_router/lean_triggers
   cp $L/paper_three/01_word_heatmaps.pdf      Paper/figures/lean_triggers_heatmaps.pdf
   cp $L/paper_three/02_word_interventions.pdf Paper/figures/lean_triggers_interventions.pdf
   mkdir -p results/lean_triggers && cp $L/attributions.jsonl $L/summary.md $L/findings.md $L/run.json results/lean_triggers/
   ```
   The last line makes the appendix sentence "the attribution and intervention records of all 12
   steps are released with the code" true (160 KB); drop the sentence otherwise.
2. **Get the BERT weights made readable** (`chmod o+r` on `outputs/bert_step_router/model/*` by
   the collaborator or root). Then `scripts/interpret_corpus.py --model <that dir> --rows
   results/bert_router/test_probabilities.jsonl` and `scripts/make_interpretability_figures.py`
   give the dataset-wide figures and category table for the *real* router in minutes, and the
   dataset-wide paragraph of Appendix "Integrated Gradients details" should be rewritten around
   them.
3. **Decide on the claims and baselines in §5**, in particular whether to re-run the LLM router a
   priori; I can add baselines 1–3 to `evaluate_routers.py` and the table on request.
4. **Compile on Overleaf** and check the two `figure*` environments and the appendix tables.
5. **Clean-up I could not do** (the sandbox blocks deletions): `outputs/partial_runs/` (three
   crashed attribution runs), `outputs/bert_router_rerun2`, `_rerun3`, `_rerun4` (partial
   trainings, ~1.3–2.6 GB each), `outputs/bert_router_rerun5/checkpoints` and
   `outputs/bert_router/checkpoints` (keep the `model/` directories if you want to inspect the
   retrained routers). `Paper_archive_previous_project/` can go once you have checked §2.
6. `references.bib` still contains the unused entries of the previous project; `fig:placeholder`
   is still the label of the method figure.
7. The background Claude session you moved aside last night ("Paper refactoring and TODOs
   implementation") is idle and contains nothing beyond the first session's transcript; it can be
   closed (`claude attach "Paper refactoring"` and exit, or `claude stop "Paper refactoring"`).

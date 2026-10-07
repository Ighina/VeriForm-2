# What drives the strongest Lean predictions?

The clearest recurring cues in these examples are **negation and qualifying language**. This is a local model-sensitivity finding, supported by both Integrated Gradients and direct word interventions; it is not evidence that these words universally trigger Lean or that the router has verified the mathematics.

The analysis covers all **12 test examples with P(Lean) ≥ 50%**, as requested. Their probabilities are only **50.03–53.97%**. None of the 1,363 saved test examples reaches 80% or 90%. Fresh model inference reproduced the selected saved probabilities to within 6e-8. Five selected examples have gold routing label Lean and seven have gold routing label Python.

| Repeated candidate | Examples with ≥0.5 percentage-point drops under both deletion and masking | Largest single-occurrence deletion drop |
|---|---:|---:|
| However | 6 | 1.88 pp |
| doesn't | 2 | 4.79 pp |
| isn't | 2 | 0.96 pp |

These counts apply to the tested candidates, not every occurrence of these words in the dataset. Other locally supported candidates include **not, can't, don't, unclear**, and passage-specific words such as **handshakes, approach, members**.

Concrete examples:

- **omnimath-238, step 4:** deleting “doesn't” reduces P(Lean) from **51.91% to 47.12%**; masking it reduces the probability to **42.96%**. This crosses the default 50% boundary. The gold route is Python.
- **gsm8k-81, step 4:** deleting one occurrence of “doesn't” reduces P(Lean) from **50.78% to 48.50%**. The gold route is Lean, so removing a cue can also remove support for a correct prediction.
- **olympiadbench-423, step 3:** deleting the tested occurrence of “not” reduces P(Lean) from **50.82% to 48.59%**. The gold route is Python.
- **olympiadbench-379, step 1:** deleting “can't” lowers P(Lean) by **1.74 pp** and deleting “unclear” lowers it by **1.54 pp**. Both masking checks also reduce P(Lean).

Together, these suggest the router has learned cues associated with qualification, uncertainty, and negative statements. Those cues can occur in both correct and incorrect routes. Individual numeric and subject-specific words also matter, so the behavior cannot be reduced to a single keyword rule.

The decision threshold is a separate cause of routing behavior. Some perturbations cross the **default 50%** boundary, but **none crosses the validation-tuned 3.408% boundary**. The smallest probability after any tested intervention, including joint removal/masking of the top five candidates, is **34.49%**. Thus, the tested words strengthen Lean preference, while the low tuned threshold keeps these examples routed to Lean even after those cues are weakened.

Method: Captum LayerIntegratedGradients at `bert.embeddings`, explaining the Lean logit against same-length PAD embeddings with CLS/SEP preserved. Sum over embedding dimensions; L2-normalize for visualization. Character offsets combine WordPieces into word occurrences. Test up to eight positive-attribution words separately by deletion and masking, three low-absolute-attribution comparison words, and the top five candidates jointly. All 12 original inputs fit within the token limit. Deletion/masking can alter meaning and tokenization; the comparison words are not matched statistical controls.

**Numerical check:** 11/12 explanations meet the 0.01-logit completeness-residual tolerance. `olympiadbench-165`, step 2, has residual **−0.01932** at the 1,600-step cap and is flagged in the heatmap report. Its direct perturbation probabilities do not depend on the IG approximation, but its attribution ranking warrants caution. In particular, one of the two “isn't” cases comes from this flagged heatmap.

See [all token heatmaps and perturbation tables](report.html), the [per-example summary](summary.md), and [raw attributions and interventions](attributions.jsonl). The implementation passed 19 focused tests covering attribution completeness, prediction parity, truncation, WordPiece aggregation, perturbation spans, and existing router behavior.

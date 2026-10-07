# Strongest Lean predictions: local trigger analysis

Selection: saved test P(Lean) ≥ 0.5; 12 examples. Attribution target: Lean logit.

These are near-boundary predictions, not high-confidence predictions. A positive perturbation drop means deleting or masking the word lowers P(Lean). The tests change the input and do not establish a universal or semantic cause.

| Example / step | P(Lean) | Gold route | Strongest confirmed local word | Deletion drop (pp) | Mask drop (pp) | IG residual |
|---|---:|---|---|---:|---:|---:|
| math-391 / 2 | 53.97% | Lean | However | 1.361 | 2.026 | -0.00323 |
| omnimath-568 / 3 | 52.64% | Lean | However | 0.787 | 2.325 | 0.00177 |
| olympiadbench-137 / 7 | 52.62% | Python | isn't | 0.707 | 1.777 | 0.00774 |
| olympiadbench-379 / 1 | 52.49% | Python | can't | 1.736 | 4.130 | 0.00902 |
| olympiadbench-165 / 2 | 52.29% | Lean | don't | 1.185 | 2.247 | -0.01932 |
| olympiadbench-659 / 1 | 52.05% | Python | However | 1.883 | 1.838 | -0.00860 |
| omnimath-238 / 4 | 51.91% | Python | doesn't | 4.790 | 8.952 | -0.00469 |
| olympiadbench-186 / 2 | 51.68% | Lean | handshakes | 1.009 | 0.879 | 0.00438 |
| olympiadbench-423 / 3 | 50.82% | Python | not | 2.229 | 2.735 | 0.00659 |
| gsm8k-81 / 4 | 50.78% | Lean | doesn't | 2.275 | 2.978 | 0.00957 |
| olympiadbench-896 / 2 | 50.38% | Python | approach | 2.152 | 4.645 | 0.00007 |
| math-507 / 1 | 50.03% | Python | members | 1.240 | 1.888 | -0.00049 |

## Repeated local candidates

Only words reducing P(Lean) by at least 0.5 percentage points under both perturbations in at least two examples are listed. Counts concern only tested candidates.

| Word | Examples | Range of smaller deletion/mask drop (pp) |
|---|---:|---:|
| however | 6 | 0.79–1.84 |
| doesn't | 2 | 2.28–4.79 |
| isn't | 2 | 0.71–0.96 |

Completeness check: 11/12 examples meet the 0.01-logit residual tolerance.

Open report.html for every token heatmap and all candidate/control perturbations.
Raw attributions, tested character spans, probabilities and convergence flags are in attributions.jsonl.

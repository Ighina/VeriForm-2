"""Layer Integrated Gradients (Sundararajan et al., 2017) on BERT's embedding layer.

Attributions are computed with Captum's ``LayerIntegratedGradients`` against a
baseline in which every non-special token is replaced by ``[PAD]`` (``[CLS]``
and ``[SEP]`` are kept, as are the attention mask, positions and segments).  The
explained quantity is either a class logit or the *routing margin*
``logit(Lean) - logit(Python)``, whose sign decides the default route.  Summing
attributions over the embedding dimension gives one signed score per WordPiece;
by completeness these scores sum to the difference between the input and the
baseline score up to the reported convergence residual.
"""
from __future__ import annotations

import html
import math
from typing import Any

import torch

LABELS = {0: "Python", 1: "Lean"}
TARGETS = ("Python", "Lean", "margin")


def _forward_factory(model, target: str):
    def forward(input_ids, attention_mask, token_type_ids):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        if target == "margin":
            return logits[:, 1] - logits[:, 0]
        return logits[:, {"Python": 0, "Lean": 1}[target]]
    return forward


def explain(model, tokenizer, text: str, target: str = "margin", n_steps: int = 50,
            internal_batch_size: int = 25, delta_tolerance: float = 0.01,
            max_steps: int | None = None) -> dict[str, Any]:
    """Attribute the routing score of one step to its input tokens.

    When ``max_steps`` is given, the number of integration steps is doubled until
    the completeness residual is within ``delta_tolerance`` or the cap is reached.
    """
    from captum.attr import LayerIntegratedGradients

    if not text.strip():
        raise ValueError("Step must not be empty")
    if model.config.id2label != LABELS:
        raise ValueError("Unexpected label mapping")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if n_steps < 2 or internal_batch_size < 1 or not math.isfinite(delta_tolerance) or delta_tolerance <= 0:
        raise ValueError("Require n_steps >= 2, a positive batch size and a positive tolerance")
    if max_steps is not None and max_steps < n_steps:
        raise ValueError("max_steps must be at least n_steps")
    model.eval()
    device = next(model.parameters()).device
    max_length = min(512, tokenizer.model_max_length, model.config.max_position_embeddings)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length,
                        return_special_tokens_mask=True)
    special = encoded.pop("special_tokens_mask").bool().to(device)
    inputs = {k: v.to(device) for k, v in encoded.items()}
    ids = inputs["input_ids"]
    baseline = ids.clone()
    baseline[~special] = tokenizer.pad_token_id
    mask = inputs["attention_mask"]
    segments = inputs.get("token_type_ids", torch.zeros_like(ids))
    forward = _forward_factory(model, target)

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, token_type_ids=segments).logits[0]
        probabilities = logits.softmax(-1)
        input_score = forward(ids, mask, segments)[0].item()
        baseline_score = forward(baseline, mask, segments)[0].item()
    lig = LayerIntegratedGradients(forward, model.bert.embeddings)
    with torch.enable_grad():
        attributions, delta = lig.attribute(
            ids, baselines=baseline, additional_forward_args=(mask, segments), n_steps=n_steps,
            method="gausslegendre", internal_batch_size=internal_batch_size, return_convergence_delta=True)
    if abs(delta.item()) > delta_tolerance and max_steps is not None and n_steps < max_steps:
        return explain(model, tokenizer, text, target, min(2 * n_steps, max_steps), internal_batch_size,
                       delta_tolerance, max_steps)
    raw = attributions.sum(-1).squeeze(0).detach().cpu()
    norm = raw.norm().item()
    normalized = raw / norm if norm else torch.zeros_like(raw)
    full_count = len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
    result = dict(
        text=text, target=target, p_lean=probabilities[1].item(),
        probabilities={LABELS[i]: probabilities[i].item() for i in LABELS},
        argmax_prediction=LABELS[int(logits.argmax())], margin=(logits[1] - logits[0]).item(),
        input_score=input_score, baseline_score=baseline_score, score_difference=input_score - baseline_score,
        attribution_sum=raw.sum().item(), convergence_delta=delta.item(), delta_tolerance=delta_tolerance,
        converged=abs(delta.item()) <= delta_tolerance, n_steps=n_steps, original_token_count=full_count,
        truncated=full_count > ids.shape[1], tokens=tokenizer.convert_ids_to_tokens(ids[0].tolist()),
        special_tokens_mask=special[0].tolist(), attributions=raw.tolist(),
        normalized_attributions=normalized.tolist(),
    )
    if not all(math.isfinite(x) for x in result["attributions"] + [result["convergence_delta"]]):
        raise ValueError("Non-finite attributions")
    return result


def render_html(results: list[dict[str, Any]]) -> str:
    """Token heatmaps for a list of ``explain`` results."""
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>BERT router: Integrated Gradients</title>',
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}"
        "article{border-top:1px solid #ccc;padding:20px 0}mark{display:inline-block;padding:3px;margin:2px;"
        "border-radius:3px}pre{white-space:pre-wrap}</style>",
        "<h1>BERT step router: Integrated Gradients</h1>",
        "<p>Blue tokens increase the explained score (towards Lean for the margin target) relative to the "
        "PAD baseline; orange tokens decrease it. Scores are L2-normalised within each example; hover for raw "
        "values. WordPiece tokens beginning with ## continue a word.</p>",
    ]
    for r in results:
        identity = html.escape(f"{r.get('dataset', '')} / {r.get('example_id', 'custom')} · step {r.get('step_index', '—')}")
        parts.append(f"<article><h2>{identity}</h2><pre>{html.escape(r['text'])}</pre>")
        parts.append(f"<p>Target: {r['target']}; P(Lean)={r['p_lean']:.4f}; margin={r['margin']:.4f}; "
                     f"gold: {html.escape(str(r.get('gold_label', 'unknown')))}; "
                     f"attribution sum={r['attribution_sum']:.4f} vs score difference={r['score_difference']:.4f}; "
                     f"residual={r['convergence_delta']:.3g} ({'ok' if r['converged'] else 'above tolerance'})</p>")
        for token, raw, score in zip(r["tokens"], r["attributions"], r["normalized_attributions"]):
            color = "0,114,178" if score >= 0 else "213,94,0"
            parts.append(f'<mark title="raw={raw:.5g}" style="background:rgba({color},{min(abs(score), 1):.3f})">'
                         f"{html.escape(token)}</mark>")
        parts.append("</article>")
    return "\n".join(parts) + "</html>\n"

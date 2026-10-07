"""Inference with a trained BERT router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import LABELS


class BertRouter:
    """Routes steps to Lean when ``P(Lean) >= threshold`` and to Python otherwise."""

    def __init__(self, model_dir: str | Path, threshold: float = 0.5, device: str = "cpu",
                 batch_size: int = 16):
        import torch
        from transformers import AutoTokenizer, BertForSequenceClassification

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = BertForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
        if self.model.config.id2label != LABELS:
            raise ValueError("Unexpected label mapping")
        self.threshold, self.device, self.batch_size = float(threshold), device, batch_size

    @classmethod
    def from_threshold_file(cls, model_dir: str | Path, threshold_file: Path, **kwargs) -> "BertRouter":
        config = json.loads(threshold_file.read_text())
        return cls(model_dir, threshold=float(config["threshold"]), **kwargs)

    def lean_probabilities(self, texts: Sequence[str]) -> list[float]:
        values: list[float] = []
        max_length = min(512, self.tokenizer.model_max_length)
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(list(texts[start:start + self.batch_size]), padding=True,
                                   truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                values.extend(self.model(**batch).logits.softmax(-1)[:, 1].cpu().tolist())
        return values

    def route(self, texts: Sequence[str]) -> list[str]:
        return [LABELS[int(p >= self.threshold)] for p in self.lean_probabilities(texts)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Route one reasoning step with the BERT router.")
    parser.add_argument("step", help="Step text")
    parser.add_argument("--model", default="outputs/bert_router/model")
    parser.add_argument("--threshold-file", type=Path, help="threshold.json written by the tuning script")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.step.strip():
        parser.error("Step must not be empty")
    if args.threshold_file:
        router = BertRouter.from_threshold_file(args.model, args.threshold_file, device=args.device)
    else:
        router = BertRouter(args.model, device=args.device)
    probability = router.lean_probabilities([args.step])[0]
    print(json.dumps({"route": router.route([args.step])[0], "p_lean": probability,
                      "threshold": router.threshold}))
    return 0

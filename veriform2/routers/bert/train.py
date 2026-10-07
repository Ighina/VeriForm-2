"""Fine-tune ``bert-base-uncased`` to predict the Python/Lean routing label from step text."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ...data import DEFAULT_DATASET, iter_processbench_rows
from ...evaluation.loaders import read_rows
from .data import LABELS, build_examples, split_examples, summarize_splits

DEFAULT_MODEL = "google-bert/bert-base-uncased"


def model_dataset(rows: list[dict[str, Any]], tokenizer, max_length: int):
    """Tokenise step text only: no metadata or verifier artefacts reach the model."""
    from datasets import Dataset

    dataset = Dataset.from_list([{"text": e["text"], "labels": e["label"]} for e in rows])
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
                       batched=True, remove_columns=["text"])


def compute_metrics(prediction) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y, pred = prediction.label_ids, prediction.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace, splits: dict[str, list[dict[str, Any]]]) -> None:
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    from transformers import (AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_length
    model = BertForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    encoded = {s: model_dataset(rows, tokenizer, args.max_length) for s, rows in splits.items()}
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"), learning_rate=args.learning_rate,
            num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, weight_decay=0.01, eval_strategy="epoch",
            save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
            metric_for_best_model="macro_f1", greater_is_better=True, seed=args.seed,
            data_seed=args.seed, report_to="none", logging_steps=25),
        train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(args.output_dir / "model"))
    tokenizer.save_pretrained(args.output_dir / "model")
    # The test split is touched only once, after validation has selected the checkpoint.
    result = trainer.predict(encoded["test"])
    pred, labels = result.predictions.argmax(axis=-1), result.label_ids
    report = {
        "metrics": result.metrics,
        "classification_report": classification_report(
            labels, pred, labels=[0, 1], target_names=["Python", "Lean"], output_dict=True, zero_division=0),
        "confusion_matrix_Python_Lean": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
        "per_source": {},
    }
    majority = Counter(e["label"] for e in splits["train"]).most_common(1)[0][0]
    report["training_majority_baseline_test_accuracy"] = float(np.mean(labels == majority))
    for source in sorted({e["dataset"] for e in splits["test"]}):
        mask = np.array([e["dataset"] == source for e in splits["test"]])
        report["per_source"][source] = classification_report(
            labels[mask], pred[mask], labels=[0, 1], target_names=["Python", "Lean"],
            output_dict=True, zero_division=0)
    write_json(args.output_dir / "test_metrics.json", report)
    with (args.output_dir / "test_predictions.jsonl").open("w") as handle:
        for row, prediction in zip(splits["test"], pred):
            handle.write(json.dumps({**row, "prediction": LABELS[int(prediction)]}) + "\n")
    print(json.dumps(report["metrics"], indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=DEFAULT_DATASET)
    parser.add_argument("--python-results", type=Path, required=True,
                        help="Python verifier step_results.jsonl (or compact CSV with target_step)")
    parser.add_argument("--non-boolean", choices=["exclude", "false"], default="exclude")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert_router"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--prepare-only", action="store_true", help="Write the splits without training")
    args = parser.parse_args(argv)
    if not 1 <= args.max_length <= 512 or args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("Require positive epochs, batch size and learning rate; max length in 1..512")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new path to preserve previous runs")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    examples, counts = build_examples(iter_processbench_rows(args.annotations),
                                      read_rows(args.python_results), args.non_boolean)
    if not examples:
        raise SystemExit("No eligible labelled steps")
    splits = split_examples(examples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (args.output_dir / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "filter_counts": counts, "splits": summarize_splits(splits), "id2label": LABELS}
    write_json(args.output_dir / "data_report.json", report)
    print(json.dumps(report["filter_counts"], indent=2))
    if not args.prepare_only:
        train(args, splits)
    return 0

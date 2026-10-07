"""Direct LLM step critic: the LLM judges a step with no verifier output.

This is the ProcessBench protocol applied step by step: the model reads the
problem, the earlier steps and the target step and says whether the target
step is correct.  It is the natural control for the routers, which give the
same LLM the artefacts of both verifiers: if the critic matches the routers,
the verifiers add nothing over the model's own judgement.

One greedy call per step (thinking mode disabled for local models), with the
same backends, decoding settings and checkpointing as the LLM router.  The
answer is a JSON object ``{"verdict": "correct" | "incorrect", "reason": ...}``;
unparseable answers are stored with an error and count as unresolved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..data import DEFAULT_DATASET, load_examples
from ..evaluation.loaders import read_rows
from ..routers.llm_router import DEFAULT_MODEL, HuggingFaceRouter, LLMRouter, OpenAIRouter

VERDICTS = ("correct", "incorrect")

SYSTEM_PROMPT = """You are a meticulous step-local mathematical proof reviewer.
You are given a problem, the earlier steps of a candidate solution and one
target step. Judge exactly the target step: is it a correct step given the
problem and the earlier steps? Use only the problem and the earlier context;
assume the earlier steps are given, and do not judge the whole solution or any
later step. A step is incorrect when it contains a computational error, an
invalid inference, a false claim or a misreading of the problem. A step that
makes no new mathematical claim (restating the problem, announcing a plan) is
correct.

Return only one JSON object with exactly this shape:
{"verdict":"correct|incorrect",
 "reason":"brief step-local justification"}"""


@dataclass(frozen=True)
class CriticStep:
    key: str
    example_id: str
    split: str
    step_index: int
    problem: str
    target_step: str
    context: list[str]


@dataclass(frozen=True)
class Judgement:
    key: str
    example_id: str
    split: str
    step_index: int
    target_step: str
    verdict: bool | None
    reason: str | None
    model: str
    backend: str
    raw_response: str | None
    error: str | None


def iter_steps(rows_path: Path, dataset: str | Path = DEFAULT_DATASET) -> Iterable[CriticStep]:
    """Join the (example_id, step_index) rows of a split with the ProcessBench problems and steps."""
    examples = load_examples(dataset)
    seen: set[tuple[str, int]] = set()
    for row in read_rows(rows_path):
        example_id, index = str(row["example_id"]), int(row["step_index"])
        if (example_id, index) in seen:
            continue
        seen.add((example_id, index))
        example = examples[example_id]
        steps = [str(s) for s in example["steps"]]
        yield CriticStep(key=f"{example_id}:{index}", example_id=example_id, split=example_id.partition("-")[0],
                         step_index=index, problem=str(example["problem"]), target_step=steps[index],
                         context=steps[:index])


def build_prompt(step: CriticStep) -> str:
    context = "\n".join(f"{i}. {v}" for i, v in enumerate(step.context, 1)) or "(none)"
    return f"""PROBLEM:
{step.problem}

EARLIER STEPS:
{context}

TARGET STEP (judge only this):
{step.target_step}

Return only the verdict JSON."""


def parse_judgement(text: str) -> dict[str, Any]:
    """Extract and validate the ``{"verdict": ..., "reason": ...}`` object."""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    value: Any = None
    decoder = json.JSONDecoder()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        for match in re.finditer(r"\{", cleaned):
            try:
                candidate, _ = decoder.raw_decode(cleaned[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
    if not isinstance(value, dict) or "verdict" not in value:
        # Long reasons can be cut by the token limit, leaving an unterminated JSON
        # object; the verdict field comes first and is still recoverable.
        match = re.search(r'"verdict"\s*:\s*"(correct|incorrect)"', cleaned, re.IGNORECASE)
        if not match:
            raise ValueError("model response did not contain a verdict JSON object")
        value = {"verdict": match.group(1), "reason": re.search(r'"reason"\s*:\s*"(.*)', cleaned, re.DOTALL)}
        value["reason"] = value["reason"].group(1).rstrip('"} \n') if value["reason"] else ""
    verdict = str(value["verdict"]).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"invalid verdict: {value['verdict']!r}")
    reason = value.get("reason")
    return {"verdict": verdict == "correct", "reason": str(reason).strip() if reason is not None else ""}


class MockCritic(LLMRouter):
    """Offline plumbing backend; its judgements are not experimental data."""

    def _call(self, user: str) -> str:
        return json.dumps({"verdict": "correct", "reason": "mock backend plumbing result"})


def read_checkpoint(path: Path) -> list[Judgement]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [Judgement(**json.loads(line)) for line in handle if line.strip()]


def write_outputs(output_dir: Path, rows: list[Judgement], final: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "judgements" if final else "judgements.partial"
    (output_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(Judgement.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def write_compact(path: Path, rows: list[Judgement]) -> None:
    """The per-step verdict file read by ``scripts/evaluate_routers.py --method``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("example_id", "step_index", "verdict", "error"))
        writer.writeheader()
        for r in rows:
            writer.writerow({"example_id": r.example_id, "step_index": r.step_index,
                             "verdict": "" if r.verdict is None else str(r.verdict),
                             "error": "" if r.error is None else r.error.split(":", 1)[0]})


def write_summary(output_dir: Path, rows: list[Judgement]) -> None:
    successful = [r for r in rows if r.error is None]
    summary = {"steps": len(rows), "judged": len(successful), "errors": len(rows) - len(successful),
               "verdicts": dict(Counter("correct" if r.verdict else "incorrect" for r in successful))}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, required=True,
                        help="Steps to judge: jsonl/csv with example_id and step_index (e.g. the BERT test split)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="ProcessBench source (Hub id or JSON file)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_critic"))
    parser.add_argument("--export", type=Path, help="Also write the compact verdict csv to this path")
    parser.add_argument("--backend", choices=("huggingface", "openai", "mock"), default="huggingface")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=os.getenv("HUGGINGFACE_DEVICE"))
    parser.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Discard the existing checkpoint")
    args = parser.parse_args(argv)
    if args.backend == "openai" and not args.api_key:
        parser.error("OPENAI_API_KEY or --api-key is required for the OpenAI backend")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = list(iter_steps(args.rows, args.dataset))
    if args.max_steps is not None:
        steps = steps[: args.max_steps]
    print(f"Judging {len(steps)} steps.", flush=True)
    rows = [] if args.overwrite else read_checkpoint(args.output_dir / "judgements.partial.jsonl")
    done = {r.key for r in rows if r.error is None}
    pending = [s for s in steps if s.key not in done]
    print(f"Checkpoint has {len(done)} completed steps; {len(pending)} pending.", flush=True)
    if pending:
        if args.backend == "huggingface":
            critic: LLMRouter = HuggingFaceRouter(args.model, args.hf_token, args.device, args.max_tokens,
                                                 system_prompt=SYSTEM_PROMPT)
        elif args.backend == "openai":
            critic = OpenAIRouter(args.model, args.api_key, args.base_url, args.timeout, args.max_tokens,
                                  args.reasoning_effort, system_prompt=SYSTEM_PROMPT)
        else:
            critic = MockCritic()
        by_key = {r.key: r for r in rows}
        fields = ("key", "example_id", "split", "step_index", "target_step")
        for number, step in enumerate(pending, 1):
            print(f"[{number}/{len(pending)}] judging {step.key}", flush=True)
            common = {k: getattr(step, k) for k in fields}
            raw = None
            try:
                raw = critic._call(build_prompt(step))
                row = Judgement(**common, **parse_judgement(raw), model=args.model, backend=args.backend,
                                raw_response=raw, error=None)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                print(f"  warning: {message}", flush=True)
                row = Judgement(**common, verdict=None, reason=None, model=args.model, backend=args.backend,
                                raw_response=raw, error=message)
            by_key[step.key] = row
            rows = [by_key[k] for k in sorted(by_key)]
            write_outputs(args.output_dir, rows)
    write_outputs(args.output_dir, rows, final=True)
    write_summary(args.output_dir, rows)
    if args.export:
        write_compact(args.export, rows)
    return 0

"""Python verifier: synthesise a step-local checker with a code LLM and execute it.

For every reasoning step the model receives the problem, the preceding steps
and the target step, and must emit a short Python program that sets ``result``
to ``True`` or ``False`` by checking *only* the inference made by the target
step.  The program is then executed in :mod:`veriform2.verifiers.sandbox`.

Results are checkpointed after every step (``step_results.partial.jsonl``) so an
interrupted run can be resumed with ``--previous-run``.
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

from ..data import DEFAULT_DATASET, iter_processbench_rows
from .sandbox import OUTCOMES, run_in_sandbox

DEFAULT_MODEL = "Qwen/Qwen3-Coder-Next"

SYSTEM_PROMPT = """Follow the user's verifier specification exactly. Return only safe,
valid Python source with no Markdown or explanation."""

SYNTH_PROMPT = """You are generating a Python verifier for exactly ONE reasoning step.

Your task is to determine whether the TARGET STEP makes a new, locally
verifiable mathematical inference from the supplied premises.

Premises:
- The original problem statement may be used to interpret notation and facts.
- The context contains reasoning steps that occurred before the target.
- Statements introduced only inside the target must not be treated as premises.

Scope restriction:
- Verify only the explicit mathematical claim or transformation made by the
  target step.
- Do not solve the original problem unless the target step itself explicitly
  performs that solution.
- Do not compute later quantities, derive the final answer, or use constraints
  that are unnecessary for checking the target.
- Do not verify the whole reasoning chain.
- A valid solution to the complete problem does not by itself prove that the
  target step is correct.

Generate a checker for every target step. A target that copies a given,
introduces notation, states an assumption, or describes what is to be found
must still be checked for faithful consistency with the problem and earlier
context. Do not abstain and do not emit an exception for such a step.

- Generate executable Python that checks only the target statement or inference.
- Derive `result` from executable operations.
- Do not hard-code `result = True` or `result = False`.
- Do not use the final solution unless the target explicitly derives it.
- Assign exactly one Boolean value to `result`.

Output requirements:
- Output only valid Python source.
- Do not use Markdown fences.
- Do not include comments or explanations.
- Do not define unused functions.
- Keep the program under 20 lines.
- Do not import modules, access files, call input(), or use network/system APIs.

Representation: linear

Problem:
{problem}

Target step:
{target}

Earlier context:
{context_text}

Write the step-local Python verifier now."""


def build_synthesis_prompt(problem: str, target: str, context: list[str]) -> str:
    context_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(context)) or "(no prior context)"
    return SYNTH_PROMPT.format(problem=problem, target=target, context_text=context_text)


def clean_code(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


@dataclass
class StepResult:
    result_id: str
    example_id: str
    representation: str
    step_id: str
    step_index: int
    target_step: str
    context: list[str]
    outcome: str
    prompt: str | None
    code: str | None
    raw_response: str | None
    error: str | None


class Synthesizer:
    """Turns a (problem, context, target step) triple into Python checker source."""

    last_raw_response: str | None = None

    def synthesize(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAISynthesizer(Synthesizer):
    def __init__(self, model: str, api_key: str | None, base_url: str | None, timeout: float):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise SystemExit("Install the OpenAI SDK: pip install openai") from error
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def synthesize(self, prompt: str) -> str:
        self.last_raw_response = None
        response = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned an empty response")
        self.last_raw_response = content
        return clean_code(content)


class HFSynthesizer(Synthesizer):
    """Greedy decoding (up to 512 new tokens) with a local Transformers model."""

    def __init__(self, model: str, token: str | None, device: str | None, max_new_tokens: int = 512):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise SystemExit("Install torch, transformers and accelerate for local synthesis") from error
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model} on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, token=token)
        kwargs: dict[str, Any] = {
            "dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True, "token": token,
        }
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()

    def synthesize(self, prompt: str) -> str:
        self.last_raw_response = None
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
        self.last_raw_response = text
        if not text.strip():
            raise RuntimeError("Model returned an empty response")
        return clean_code(text)


class MockSynthesizer(Synthesizer):
    """Offline plumbing backend; its outcomes are not experimental results."""

    def synthesize(self, prompt: str) -> str:
        self.last_raw_response = "result = True"
        return "result = True"


def linear_steps(row: dict[str, Any]) -> Iterable[tuple[str, int, str, list[str]]]:
    steps = row.get("steps") or []
    for index, target in enumerate(steps):
        yield f"linear_{index + 1}", index, target, list(steps[:index])


def save_jsonl(path: Path, results: list[StepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in results), encoding="utf-8")


def save_step_outcomes(path: Path, results: list[StepResult]) -> None:
    """Compact per-step outcome table (no prompts or code)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("result_id", "example_id", "representation", "step_id", "step_index", "outcome", "error")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({f: getattr(item, f) if f != "error" else (item.error or "") for f in fields})


def save_synthesized_code(output_dir: Path, model: str, example_id: str, results: list[StepResult]) -> None:
    """Write the generated checkers of one example as a commented Python file."""
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s)  # noqa: E731
    debug_dir = output_dir / "debug_code" / safe(model)
    debug_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Synthesized step checkers for {example_id} using model {model}", ""]
    for result in results:
        lines.append(f"# Step: {result.result_id} / {result.step_id}")
        lines.append(f"# Target: {result.target_step}")
        lines.append(f"# Sandbox outcome: {result.outcome}")
        if result.error:
            lines.append(f"# Error: {result.error}")
        lines.append(result.code if result.code is not None else "# [no code generated]")
        lines.append("\n# " + "-" * 78 + "\n")
    (debug_dir / f"{safe(example_id)}.py").write_text("\n".join(lines), encoding="utf-8")


def load_results(path: Path) -> list[StepResult]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(StepResult(**{k: row.get(k) for k in StepResult.__dataclass_fields__}))
    return rows


def write_summary(results: list[StepResult], output_dir: Path, split: str) -> dict[str, Any]:
    counts = Counter(r.outcome for r in results)
    summary = {
        "steps": len(results),
        "outcomes": {outcome: counts[outcome] for outcome in OUTCOMES},
        "dataset_split": split,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face id or local ProcessBench JSON")
    parser.add_argument("--split", default="all", choices=("all", "gsm8k", "math", "olympiadbench", "omnimath"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/python_verifier"))
    parser.add_argument("--previous-run", type=Path, help="Resume from this output directory")
    parser.add_argument("--backend", choices=("huggingface", "openai", "mock"), default="huggingface")
    parser.add_argument("--model", default=os.getenv("VERIFORM_PYTHON_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--device", default=os.getenv("HUGGINGFACE_DEVICE"))
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds allowed for each checker")
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args(argv)
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be at least 1")
    if args.backend == "openai" and not args.api_key:
        parser.error("OPENAI_API_KEY or --api-key is required for the openai backend")
    if args.previous_run is not None:
        args.output_dir = args.previous_run
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.backend == "huggingface":
        synthesizer: Synthesizer = HFSynthesizer(args.model, args.hf_token, args.device)
    elif args.backend == "openai":
        synthesizer = OpenAISynthesizer(args.model, args.api_key, args.base_url, 60.0)
    else:
        synthesizer = MockSynthesizer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / "step_results.partial.jsonl"
    results = load_results(partial if partial.is_file() else args.output_dir / "step_results.jsonl")
    completed = {(r.example_id, r.step_index) for r in results}
    if results:
        print(f"[Resume] {len(completed)} completed steps will be skipped", flush=True)

    examples = [
        row for row in iter_processbench_rows(args.dataset)
        if args.split == "all" or row.get("split") == args.split
    ]
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    for number, row in enumerate(examples, 1):
        example_id = str(row["id"])
        print(f"[{number}/{len(examples)}] {example_id}", flush=True)
        for step_id, index, target, context in linear_steps(row):
            if (example_id, index) in completed:
                continue
            prompt = build_synthesis_prompt(row.get("problem", ""), target, context)
            code = error = None
            try:
                code = synthesizer.synthesize(prompt)
                outcome, error = run_in_sandbox(code, args.timeout)
            except Exception as exc:  # Keep the run going; record the failure per step.
                outcome, error = "Autoformalisation failure", f"{type(exc).__name__}: {exc}"
            results.append(StepResult(
                result_id=f"{example_id}.linear.{index}", example_id=example_id, representation="linear",
                step_id=step_id, step_index=index, target_step=target, context=context, outcome=outcome,
                prompt=prompt, code=code, raw_response=synthesizer.last_raw_response, error=error,
            ))
            completed.add((example_id, index))
            save_synthesized_code(args.output_dir, args.model, example_id,
                                  [r for r in results if r.example_id == example_id])
            save_jsonl(partial, results)
            print(f"  [{step_id}] {outcome}", flush=True)
    save_jsonl(args.output_dir / "step_results.jsonl", results)
    save_step_outcomes(args.output_dir / "step_outcomes.csv", results)
    print(json.dumps(write_summary(results, args.output_dir, args.split), indent=2))
    return 0

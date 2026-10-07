"""LLM router: an instruction-tuned LLM chooses the verifier for each step.

The router is given the problem, the earlier steps, the target step and the
complete artefact bundles produced by *both* verifiers for that step (the
synthesised Python checker with its execution outcome, and the Lean statement,
proof attempt and compiler outcome).  It answers with one of

``python`` / ``lean``
    trust that verifier's verdict for the step;
``tie``
    both verifications are sound and equivalent;
``neither``
    the step is checkable but both verifications are wrong (step treated as incorrect);
``inconclusive``
    the step makes no new checkable inference (step treated as correct).

Every classification is produced in two calls: a draft, then a mandatory
self-review that must apply the decision procedure in order.  Only the reviewed
answer is kept; both raw responses are stored for auditing.  Runs are
checkpointed after each step and can be resumed.
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

from ..evaluation.loaders import read_rows
from ..verifiers.lean_verifier import lean_blocks, load_step_outcomes, runtime_artifacts

CHOICES = ("python", "lean", "tie", "neither", "inconclusive")
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

SYSTEM_PROMPT = """You are an LLM agent jointly comparing the complete Python and
Lean verification artifact bundles for exactly one mathematical reasoning step.

You are a meticulous step-local mathematical proof reviewer.
Compare two candidate verifiers for exactly one reasoning step. Do not reward a
candidate for proving the full problem, using later facts, or proving a stronger
claim. A proof may be syntactically valid yet semantically misaligned.

Use only the problem and earlier context. Judge exactly the target step, not the
whole solution and not any later step. Inspect the generated source,
autoformalisation, proof, model responses, execution/compilation outcomes,
alignment results, errors, and runtime logs. Compilation or execution success
alone is not enough: judge semantic fidelity, soundness, non-vacuity, use of
only prior context, and suitability for the exact target. Compare the bundles
jointly; do not produce separate verifier verdicts.

Choose python when only the Python bundle is a sound and faithful verification,
or when both work but Python is materially better suited. Choose lean analogously.
Choose tie only when both bundles are sound, faithful, and equally suitable.
Always choose inconclusive when the target step makes no new checkable inference,
whether or not either candidate abstains. Any non-abstaining candidate in that
case is semantically misaligned. Also choose inconclusive when both candidates
appropriately abstain or when the comparison cannot be decided.
Choose neither only when the target step is verifiable and both candidates are
incorrect. Choose tie when both are equally correct.

Return only one JSON object with exactly this shape:
{"choice":"python|lean|tie|neither|inconclusive",
 "reason":"brief step-local joint comparison"}"""

REVIEW_PROMPT = """Audit the draft classification below and return the corrected final
JSON object. Apply this decision procedure in order:

1. First decide whether the TARGET STEP makes a new checkable mathematical
   inference from the problem and earlier context.
2. If it does not, choice MUST be "inconclusive", regardless of whether Python
   or Lean abstains, succeeds, compiles, executes, or proves a stronger claim.
   A non-abstaining verifier is semantically misaligned in this case.
3. Only if the target is verifiable may you compare the artifact bundles:
   choose "neither" if both are incorrect, "tie" if both are equally correct,
   or "python"/"lean" for the sole or materially better correct verifier.
4. Use "inconclusive" if the evidence cannot support a reliable decision.

Do not repeat a draft choice that conflicts with its own reason. Return exactly
{{"choice":"python|lean|tie|neither|inconclusive","reason":"brief justification"}}
and no other fields or prose.

DRAFT RESPONSE:
{draft}

ORIGINAL JOINT COMPARISON:
{comparison}"""


@dataclass(frozen=True)
class Step:
    key: str
    result_id: str
    example_id: str
    split: str
    step_index: int
    step_id: str
    problem: str
    target_step: str
    context: list[str]
    python_bundle: dict[str, Any]
    lean_code: str
    lean_metadata: dict[str, str]
    lean_runtime_artifacts: dict[str, str]


@dataclass(frozen=True)
class Classification:
    key: str
    result_id: str
    example_id: str
    split: str
    step_index: int
    step_id: str
    target_step: str
    choice: str | None
    reason: str | None
    model: str
    backend: str
    raw_response: str | None
    error: str | None


def iter_steps(python_results: Path, lean_run: Path, split: str = "all",
               max_examples: int | None = None) -> Iterable[Step]:
    """Join Python results with Lean artefacts; yield steps with complete bundles."""
    metadata = load_step_outcomes(lean_run)
    lean_cache: dict[str, dict[int, str]] = {}
    example_order: list[str] = []
    for row in read_rows(python_results):
        if row.get("representation", "linear") != "linear":
            continue
        example_id = str(row["example_id"])
        row_split = example_id.partition("-")[0]
        if split != "all" and row_split != split:
            continue
        if example_id not in example_order:
            example_order.append(example_id)
        if max_examples is not None and example_order.index(example_id) >= max_examples:
            continue
        index = int(row["step_index"])
        lean_row = metadata.get((example_id, index))
        lean_path = lean_run / "steps/LinearDAGModel" / f"{example_id}.lean"
        if lean_row is None or not lean_path.is_file():
            continue
        if example_id not in lean_cache:
            lean_cache[example_id] = lean_blocks(lean_path)
        lean_code = lean_cache[example_id].get(index)
        if lean_code is None:
            continue
        prompt = str(row.get("prompt") or "")
        match = re.search(r"\nProblem:\n(.*?)\n\nTarget step:\n", prompt, re.DOTALL)
        yield Step(
            key=f"{example_id}:linear:{index}", result_id=str(row["result_id"]), example_id=example_id,
            split=row_split, step_index=index, step_id=str(row["step_id"]),
            problem=match.group(1).strip() if match else "", target_step=str(row["target_step"]),
            context=[str(v) for v in row.get("context") or []], python_bundle=dict(row),
            lean_code=lean_code, lean_metadata=dict(lean_row),
            lean_runtime_artifacts=runtime_artifacts(lean_run, lean_row["result_id"]),
        )


def build_prompt(step: Step) -> str:
    context = "\n".join(f"{i}. {v}" for i, v in enumerate(step.context, 1)) or "(none)"
    return f"""PROBLEM:
{step.problem}

EARLIER CONTEXT:
{context}

TARGET STEP (judge only this):
{step.target_step}

PYTHON ARTIFACT BUNDLE (full JSONL record):
{json.dumps(step.python_bundle, indent=2, ensure_ascii=False)}

LEAN AUTOFORMALISATION AND PROOF:
```lean
{step.lean_code}
```

LEAN PIPELINE AND RUNTIME OUTCOMES:
{json.dumps(step.lean_metadata, indent=2, ensure_ascii=False)}

LEAN GENERATION PROMPTS AND RAW MODEL/RUNTIME OUTPUTS:
{json.dumps(step.lean_runtime_artifacts, indent=2, ensure_ascii=False)}

Compare the complete bundles jointly and return only the routing JSON."""


def build_review_prompt(step: Step, draft_response: str) -> str:
    return REVIEW_PROMPT.format(draft=draft_response, comparison=build_prompt(step))


def parse_classification(text: str) -> dict[str, str]:
    """Extract and validate the ``{"choice": ..., "reason": ...}`` object."""
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
    if not isinstance(value, dict):
        raise ValueError("model response did not contain a JSON object")
    if set(value) != {"choice", "reason"}:
        raise ValueError("response must contain exactly choice and reason")
    if value["choice"] not in CHOICES:
        raise ValueError(f"invalid choice: {value['choice']!r}")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a non-empty string")
    return {"choice": value["choice"], "reason": value["reason"].strip()}


class LLMRouter:
    """Draft-then-review classification; subclasses implement one model call."""

    def _call(self, user: str) -> str:
        raise NotImplementedError

    def classify(self, step: Step) -> tuple[dict[str, str], str]:
        draft = self._call(build_prompt(step))
        review = self._call(build_review_prompt(step, draft))
        raw = json.dumps({"draft": draft, "review": review}, ensure_ascii=False)
        return parse_classification(review), raw


class OpenAIRouter(LLMRouter):
    def __init__(self, model: str, api_key: str | None, base_url: str | None, timeout: float,
                 max_tokens: int, reasoning_effort: str | None, system_prompt: str = SYSTEM_PROMPT):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise SystemExit("The OpenAI backend requires the openai package.") from error
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model, self.max_tokens, self.reasoning_effort = model, max_tokens, reasoning_effort
        self.system_prompt = system_prompt

    def _call(self, user: str) -> str:
        request: dict[str, Any] = {
            "model": self.model, "max_completion_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}],
        }
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        response = self.client.chat.completions.create(**request)
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            raise RuntimeError("model returned an empty response")
        return raw


class HuggingFaceRouter(LLMRouter):
    """Greedy decoding with a local chat model (thinking mode disabled)."""

    def __init__(self, model: str, token: str | None, device: str | None, max_tokens: int,
                 system_prompt: str = SYSTEM_PROMPT):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise SystemExit("Local inference requires transformers, accelerate and torch.") from error
        self.torch, self.max_tokens, self.system_prompt = torch, max_tokens, system_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model} on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=token, trust_remote_code=True)
        kwargs: dict[str, Any] = {"token": token, "trust_remote_code": True,
                                  "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()

    def _call(self, user: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"add_generation_prompt": True, "return_tensors": "pt",
                                  "return_dict": True, "enable_thinking": False}
        try:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded, max_new_tokens=self.max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        if not raw:
            raise RuntimeError("model returned an empty response")
        return raw


class MockRouter(LLMRouter):
    """Offline plumbing backend; its classifications are not experimental data."""

    def _call(self, user: str) -> str:
        return json.dumps({"choice": "inconclusive", "reason": "mock backend plumbing result"})


def read_checkpoint(path: Path) -> list[Classification]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [Classification(**json.loads(line)) for line in handle if line.strip()]


def write_outputs(output_dir: Path, rows: list[Classification], final: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "classifications" if final else "classifications.partial"
    (output_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(Classification.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def write_summary(output_dir: Path, rows: list[Classification]) -> None:
    successful = [r for r in rows if r.error is None]
    summary = {"steps": len(rows), "classified": len(successful), "errors": len(rows) - len(successful),
               "choices": dict(Counter(r.choice for r in successful))}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-results", type=Path, required=True, help="step_results.jsonl of the Python verifier")
    parser.add_argument("--lean-run", type=Path, required=True, help="Lean run directory (contains steps/)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_router"))
    parser.add_argument("--split", choices=("all", "gsm8k", "math", "olympiadbench", "omnimath"), default="all")
    parser.add_argument("--backend", choices=("huggingface", "openai", "mock"), default="huggingface")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=os.getenv("HUGGINGFACE_DEVICE"))
    parser.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                        help="OpenAI reasoning effort (omit for the model default)")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Discard the existing checkpoint")
    args = parser.parse_args(argv)
    for name in ("max_examples", "max_steps", "max_tokens"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.backend == "openai" and not args.api_key:
        parser.error("OPENAI_API_KEY or --api-key is required for the OpenAI backend")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = list(iter_steps(args.python_results, args.lean_run, args.split, args.max_examples))
    if args.max_steps is not None:
        steps = steps[: args.max_steps]
    print(f"Aligned {len(steps)} ProcessBench steps with complete Python/Lean bundles.", flush=True)
    rows = [] if args.overwrite else read_checkpoint(args.output_dir / "classifications.partial.jsonl")
    done = {r.key for r in rows if r.error is None}
    pending = [s for s in steps if s.key not in done]
    print(f"Checkpoint has {len(done)} completed steps; {len(pending)} pending.", flush=True)
    if pending:
        if args.backend == "huggingface":
            router: LLMRouter = HuggingFaceRouter(args.model, args.hf_token, args.device, args.max_tokens)
        elif args.backend == "openai":
            router = OpenAIRouter(args.model, args.api_key, args.base_url, args.timeout,
                                  args.max_tokens, args.reasoning_effort)
        else:
            router = MockRouter()
        by_key = {r.key: r for r in rows}
        fields = ("key", "result_id", "example_id", "split", "step_index", "step_id", "target_step")
        for number, step in enumerate(pending, 1):
            print(f"[{number}/{len(pending)}] classifying {step.key}", flush=True)
            common = {k: getattr(step, k) for k in fields}
            try:
                verdict, raw = router.classify(step)
                row = Classification(**common, **verdict, model=args.model, backend=args.backend,
                                     raw_response=raw, error=None)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                print(f"  warning: {message}", flush=True)
                row = Classification(**common, choice=None, reason=None, model=args.model,
                                     backend=args.backend, raw_response=None, error=message)
            by_key[step.key] = row
            rows = [by_key[k] for k in sorted(by_key)]
            write_outputs(args.output_dir, rows)
    write_outputs(args.output_dir, rows, final=True)
    write_summary(args.output_dir, rows)
    return 0

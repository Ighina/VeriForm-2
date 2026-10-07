#!/usr/bin/env python3
"""Autoformalise and prove every ProcessBench step with the Lean pipeline.

This driver requires the external ``veriform`` package (Goedel-Formalizer-V2-8B,
Goedel-Prover-V2-8B served with vLLM, and a Lean 4 + Mathlib workspace reachable
through ``check_lean_environment``).  Each step is formalised with all preceding
steps as context, a proof is attempted and compiled, and the per-step outcomes
are checkpointed to ``<run>/steps/LinearDAGModel/step_outcomes.csv``, the file
consumed by :mod:`veriform2.verifiers.lean_verifier` and the evaluation scripts.

    python scripts/run_lean_verifier.py --split all --output-root outputs/lean_runs
    python scripts/run_lean_verifier.py --previous-run outputs/lean_runs/20260806_171011
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401

try:
    from veriform.autoformalization_v2.dag import LinearDAGModel
    from veriform.autoformalization_v2.deepseek.prover.lean.verifier import check_lean_environment
    from veriform.autoformalization_v2.formalizer import GoedelFormalizer
    from veriform.autoformalization_v2.perturber import StandardPerturber
    from veriform.autoformalization_v2.pipeline import StandardPipeline
    from veriform.autoformalization_v2.prover import PROOF_OUTCOMES, GoedelProver
    from veriform.data_collection.dataset_loaders import DAGProcessBenchLoader
except ImportError as error:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "The Lean verifier needs the external `veriform` package (see README.md): " + str(error)
    ) from error

FIELDS = (
    "result_id", "example_id", "representation", "step_id", "formalization_valid", "alignment_status",
    "proof_status", "proof_compiled", "semantic_outcome", "result", "formalization_error",
    "alignment_error", "proof_error", "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="Qwen/ProcessBench")
    parser.add_argument("--split", choices=("all", "gsm8k", "math", "olympiadbench", "omnimath"), default="all")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/lean_runs"))
    parser.add_argument("--previous-run", type=Path, help="Resume this run directory")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25, help="Per vLLM engine")
    args = parser.parse_args()
    if args.num_examples is not None and args.num_examples < 1:
        parser.error("--num-examples must be at least 1")
    return args


def save_outcomes(path: Path, rows: list[dict[str, str]]) -> None:
    invalid = {row.get("result") for row in rows} - set(PROOF_OUTCOMES)
    if invalid:
        raise ValueError(f"Unknown proof outcomes: {sorted(invalid)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_outcomes(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Checkpoint not found: {path}")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("result_id", "") for row in rows]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise SystemExit(f"Checkpoint has missing or duplicate result ids: {path}")
    return rows


def prepare_resume(chains, rows):
    """Keep fully processed chains; rerun interrupted ones from their first step."""
    by_chain: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_chain.setdefault(row["result_id"].partition(".")[0], []).append(row)
    completed = {str(c.chain_id) for c in chains if len(by_chain.get(str(c.chain_id), ())) == len(c.steps)}
    kept = [row for row in rows if row["result_id"].partition(".")[0] in completed]
    remaining = [c for c in chains if str(c.chain_id) not in completed]
    return remaining, kept, len(completed)


def main() -> int:
    args = parse_args()
    python_bin = str(Path(sys.executable).parent)  # keep vLLM's JIT tools discoverable under schedulers
    if python_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join((python_bin, os.environ.get("PATH", "")))
    lake_path, workspace = check_lean_environment()
    print(f"Lean preflight passed: {lake_path} ({workspace})", flush=True)

    chains = DAGProcessBenchLoader(data_path=args.dataset, split=args.split, seed=42, structure="steps").load()
    if args.num_examples is not None:
        chains = chains[: args.num_examples]
    run_dir = args.previous_run.resolve() if args.previous_run else args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "steps" / LinearDAGModel.__name__
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if args.previous_run:
        chains, rows, done = prepare_resume(chains, load_outcomes(out_dir / "step_outcomes.csv"))
        print(f"Resume: {done} complete chains skipped; {len(chains)} remaining.", flush=True)
    (run_dir / "run_status.txt").write_text(f"running structure=steps split={args.split}\n")

    pipeline = StandardPipeline(
        StandardPerturber(p=0.0, operator_swap=False, value_change=False, logical_negation=False),
        GoedelFormalizer(tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization,
                         debug_dir=run_dir / "steps"),
        GoedelProver(batch_size=1, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization,
                     debug_dir=run_dir / "steps"),
        dag_model_class=LinearDAGModel,
    )
    for chain in chains:
        example_id = str(chain.metadata.get("original_index", chain.chain_id))
        lean_path = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', example_id)}.lean"

        def checkpoint(dag, step_index, chain=chain, example_id=example_id, lean_path=lean_path):
            node = dag.nodes[step_index]
            rows.append({
                "result_id": f"{chain.chain_id}.{step_index}", "example_id": example_id, "representation": "steps",
                "step_id": f"step_{node.node_id}",
                "formalization_valid": str(node.formalization_status == "formalized"),
                "alignment_status": node.alignment_status, "proof_status": node.proof_status,
                "proof_compiled": "True" if node.proof_status == "compiled" else "False" if node.proof_status == "rejected" else "",
                "semantic_outcome": node.proof_outcome or "Prover failure", "result": node.proof_outcome or "Prover failure",
                "formalization_error": node.status_reason or "", "alignment_error": node.alignment_error or "",
                "proof_error": node.proof_error or "",
                "error": node.proof_error or node.alignment_error or node.status_reason or "",
            })
            lean_path.write_text(dag.lean())
            save_outcomes(out_dir / "step_outcomes.csv", rows)

        pipeline.on_step_complete = checkpoint
        try:
            lean_path.write_text(pipeline(chain))
        finally:
            pipeline.on_step_complete = None
    save_outcomes(out_dir / "step_outcomes.csv", rows)
    (run_dir / "run_status.txt").write_text(f"complete structure=steps split={args.split}\n")
    print(f"Run complete: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

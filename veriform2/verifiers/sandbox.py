"""Restricted execution of synthesised Python checkers.

A checker is a short Python program that must assign a Boolean to ``result``.
Before execution the source is parsed and rejected if it uses imports,
attribute access, function or class definitions, exception handling or other
constructs that could escape the sandbox.  The accepted program then runs in a
fresh isolated interpreter (``python -I``) with a minimal builtins table, a CPU
time limit and a memory limit.

Outcomes mirror the Lean pipeline so the two verifiers share one vocabulary:
``True``/``False`` are Boolean verdicts, ``Autoformalisation failure`` means no
executable checker was produced, and ``Prover failure`` means the checker did not
yield a Boolean (runtime error, time-out, non-Boolean result).
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile

OUTCOMES = ("True", "False", "Autoformalisation failure", "Prover failure")

_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Attribute, ast.Lambda, ast.ClassDef, ast.FunctionDef,
    ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.Try, ast.Global, ast.Nonlocal,
    ast.Delete, ast.Yield, ast.YieldFrom, ast.Await, ast.Raise,
)

_WRAPPER = """
import json
_SAFE = {'abs': abs, 'min': min, 'max': max, 'sum': sum, 'len': len, 'range': range,
         'all': all, 'any': any, 'True': True, 'False': False}
_source = %r
_scope = {'__builtins__': _SAFE}
exec(compile(_source, '<synthesized>', 'exec'), _scope, _scope)
_value = _scope.get('result')
if type(_value) is not bool:
    raise TypeError('result must be a bool')
print(json.dumps(_value))
"""


def static_check(code: str) -> str | None:
    """Return a rejection reason, or ``None`` when the checker may be executed."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        return f"syntax error: {error.msg}"
    if any(isinstance(node, _FORBIDDEN_NODES) for node in ast.walk(tree)):
        return "sandbox policy rejected unsafe Python syntax"
    assigns_result = any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "result"
        for node in ast.walk(tree)
    )
    if not assigns_result:
        return "checker does not assign result"
    return None


def run_in_sandbox(code: str, timeout: float = 10.0) -> tuple[str, str | None]:
    """Execute a checker and return ``(outcome, error)``."""
    rejection = static_check(code)
    if rejection is not None:
        return "Autoformalisation failure", rejection

    def limit_resources() -> None:  # pragma: no cover - runs in the child process
        try:
            import resource

            seconds = max(1, int(timeout))
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        except (ImportError, OSError, ValueError):
            pass

    try:
        with tempfile.TemporaryDirectory(prefix="veriform-sandbox-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _WRAPPER % code],
                cwd=directory, text=True, capture_output=True, timeout=timeout,
                check=False, preexec_fn=limit_resources,
            )
    except subprocess.TimeoutExpired:
        return "Prover failure", f"sandbox timed out after {timeout}s"
    if completed.returncode != 0:
        return "Prover failure", (completed.stderr.strip() or "sandbox process failed")[:1000]
    value = completed.stdout.strip()
    if value == "true":
        return "True", None
    if value == "false":
        return "False", None
    return "Prover failure", "sandbox produced no Boolean result"

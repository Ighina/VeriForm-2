#!/usr/bin/env python3
"""Run the Python verifier (checker synthesis + sandboxed execution) over ProcessBench."""
import _paths  # noqa: F401
from veriform2.verifiers.python_verifier import main

if __name__ == "__main__":
    raise SystemExit(main())

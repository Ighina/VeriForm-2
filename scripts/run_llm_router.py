#!/usr/bin/env python3
"""Run the LLM router over steps that have both a Python and a Lean verification bundle."""
import _paths  # noqa: F401
from veriform2.routers.llm_router import main

if __name__ == "__main__":
    raise SystemExit(main())

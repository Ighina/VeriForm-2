#!/usr/bin/env python3
"""Run the verifier-free LLM step critic (ProcessBench protocol) over a set of steps."""
import _paths  # noqa: F401
from veriform2.baselines.llm_critic import main

if __name__ == "__main__":
    raise SystemExit(main())

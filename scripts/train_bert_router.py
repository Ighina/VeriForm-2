#!/usr/bin/env python3
"""Build routing labels, split ProcessBench by problem, and fine-tune the BERT router."""
import _paths  # noqa: F401
from veriform2.routers.bert.train import main

if __name__ == "__main__":
    raise SystemExit(main())

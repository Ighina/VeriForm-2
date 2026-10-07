#!/usr/bin/env python3
"""Select the BERT router's Lean threshold on validation data and save split probabilities."""
import _paths  # noqa: F401
from veriform2.routers.bert.threshold import main

if __name__ == "__main__":
    raise SystemExit(main())

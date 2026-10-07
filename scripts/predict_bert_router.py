#!/usr/bin/env python3
"""Route a single reasoning step with a trained BERT router."""
import _paths  # noqa: F401
from veriform2.routers.bert.predict import main

if __name__ == "__main__":
    raise SystemExit(main())

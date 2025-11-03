# src/opencount_ci/__main__.py
"""Entry point when running as python -m opencount_ci"""
import sys
from pathlib import Path

# Add parent directory to path if needed
if __package__ is None:
    parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent))
    from opencount_ci.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
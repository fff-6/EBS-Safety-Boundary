"""Thin wrapper to run the boundary stability experiment from repo root.

This exists to support commands like:

    uv run python eval_xstest_boundary_stability.py ...

The implementation lives in `eval_scripts/eval_xstest_boundary_stability.py`.
"""

from __future__ import annotations

from eval_scripts.eval_xstest_boundary_stability import main


if __name__ == "__main__":
    main()

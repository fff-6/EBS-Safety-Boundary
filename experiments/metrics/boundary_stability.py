"""Thin wrapper for the boundary-stability evaluation entrypoint."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("eval_scripts.eval_xstest_boundary_stability", run_name="__main__")

"""Thin wrapper for the RedBench evaluation entrypoint."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("eval_scripts.eval_ebs_redbench", run_name="__main__")

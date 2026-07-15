"""Thin wrapper for the main experiment entrypoint."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("eval_scripts.eval_ebs_main_experiment", run_name="__main__")

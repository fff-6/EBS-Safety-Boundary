"""Thin wrapper for the experience-generation entrypoint."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("ebs.train", run_name="__main__")

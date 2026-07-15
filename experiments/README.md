# Experiments

This directory groups the repository's user-facing experiment entrypoints and
experiment entrypoints.

The project's main experimental workflow is organized into two parts:

- `experience_generation/`
  Generate or update reusable experience banks.
- `metrics/`
  Run benchmark and paper-facing evaluation metrics.

Important:

- these files are convenience entrypoints
- core runtime logic still lives in the original code packages for now
- compatibility paths remain in place while the repository is being cleaned up
- the single source of truth for configuration remains under `configs/`

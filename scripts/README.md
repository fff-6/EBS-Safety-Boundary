# Scripts

This directory contains lightweight entrypoint wrappers and convenience scripts.

Current contents include:

- `cli_chat.py`: thin wrapper around the CLI chat entrypoint
- compatibility PowerShell wrappers for minimal smoke-size runs
- `validate_experiment_inputs.py`: validates construction counts, prompt fields, duplicates, required RedBench subsets,
  and exact construction/evaluation overlap
- `run_full_experience_construction.ps1`: canonical 800-sample/4000-rollout full experience construction and artifact export
- `run_xstest_five_seeds.ps1`: canonical five-seed Vanilla/EBS XSTest comparison and aggregation

Experiment-focused launchers now live under:

- `experiments/experience_generation/`

Run the input validator and the minimal smoke test before starting paid full-scale API calls. The full protocol,
outputs, resume behavior, and acceptance criteria are documented in the repository README files.

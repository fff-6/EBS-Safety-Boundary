# Metrics

This subdirectory groups the experiment entrypoints used to run benchmark
metrics and paper-facing evaluation summaries.

Current wrappers:

- `main_experiment.py`: main HarmBench + XSTest paper run
- `redbench.py`: RedBench-style evaluation wrapper
- `boundary_stability.py`: XSTest boundary stability evaluation

These wrappers keep the execution surface together without changing the
underlying evaluation implementation.

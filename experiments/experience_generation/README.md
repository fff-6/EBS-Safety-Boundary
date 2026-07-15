# Experience Generation

This subdirectory groups the experiment entrypoints related to generating or
updating experience banks.

Current items:

- `train.py`: thin wrapper around `ebs/train.py`
- `run_three_bucket_minimal.ps1`: minimal mixed-bucket smoke run
- `run_evil_data_minimal.ps1`: minimal ethics-bucket smoke run

These wrappers do not change training logic. They only provide a clearer place
to discover the experience-generation workflow.

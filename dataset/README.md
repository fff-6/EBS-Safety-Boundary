# Dataset preparation

Third-party datasets are not included in this repository. Download them from their official sources, review their
licenses or access conditions, and keep them under this ignored `dataset/` directory.

## Paper evaluation: RedBench

The main evaluation commands expect `dataset/redbench_dataset.json`, a JSON object keyed by benchmark name and then
split name. Download every official RedBench subset and convert it to that layout from the project root:

```bash
uv run python scripts/prepare_redbench.py
```

The script uses Hugging Face Datasets and writes only to the ignored `dataset/` directory. RedBench already contains
the HarmBench and XSTest subsets used by the paper-facing main experiment.

- RedBench: https://huggingface.co/datasets/knoveleng/redbench
- RedEval: https://github.com/knoveleng/redeval
- HarmBench: https://github.com/centerforaisafety/HarmBench
- XSTest: https://github.com/paul-rottger/xstest

## Experience generation

The experience-generation examples expect these locally prepared JSONL files:

- `wildjailbreak_harmful.jsonl`: harmful samples selected from WildJailbreak.
- `star_benign.jsonl`: benign samples selected from STAR-benign-915.
- `evil_data.jsonl`: the ethics-category data used by your experiment. This is a paper-specific processed split, so
  prepare it from the source stated in the paper and do not redistribute it without permission.

Official sources:

- WildJailbreak: https://huggingface.co/datasets/allenai/wildjailbreak (requires accepting the access conditions).
- STAR-1 and STAR-benign-915: https://github.com/UCSC-VLAA/STAR-1 and
  https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915.

Each JSONL line must be an object containing one prompt field (`prompt`, `problem`, `question`, `instruction`, or
`query`). Optional fields include `response`, `harmful_label`, `level`, and `experience_bucket`. Use
`experience_bucket` values `harmful`, `benign`, or `ethics` when an explicit category is needed.

By default EBS reads built-in dataset directories from `dataset/`. Set `EBS_DATA_DIR` to use another location.

## Optional RedEval human-jailbreak attack

The paper main experiment uses the `direct` attack and does not require human-jailbreak templates. To reproduce the
optional `human_jailbreak` RedEval attack, obtain `redeval/resources/human_jailbreaks.py` from the official RedEval
repository and place it at `dataset/human_jailbreaks.py`. This file remains ignored by Git.

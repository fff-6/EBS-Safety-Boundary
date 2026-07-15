# EBS

## 🚀 Getting Started

Follow the steps below to configure the environment, build an experience bank, and run the paper experiments.

### 1. Setup

Install the locked dependencies in the project root:

```bash
uv sync
```

Copy the environment template. On PowerShell:

```powershell
Copy-Item .env.example .env
```

On Linux or macOS:

```bash
cp .env.example .env
```

Edit `.env` and configure the target and judge models. The main paper setup uses:

```ini
EBS_LLM_TYPE=chat.completions
EBS_LLM_MODEL=Qwen/Qwen3-8B
EBS_LLM_BASE_URL=https://api.siliconflow.cn/v1
EBS_LLM_API_KEY=replace-me

TARGET_LLM_MODEL=Qwen/Qwen3-8B
TARGET_LLM_BASE_URL=https://api.siliconflow.cn/v1
TARGET_LLM_API_KEY=replace-me

EBS_JUDGE_MODEL=Qwen/Qwen3-14B
EBS_JUDGE_BASE_URL=https://api.siliconflow.cn/v1
EBS_JUDGE_API_KEY=replace-me

ATTACK_JUDGE_API_KEY=replace-me
OPENAI_API_KEY=replace-me
```

Run the tests:

```bash
uv run pytest -q
```

### Datasets

Datasets are not distributed with this repository. Before running the experiments, follow
[`dataset/README.md`](dataset/README.md) to download RedBench, WildJailbreak, STAR-benign-915, HarmBench, and XSTest
from their official sources. The main experiment expects the converted RedBench file at
`dataset/redbench_dataset.json`; experience generation uses locally prepared JSONL files under `dataset/`.

### 2. Build the Experience Bank

Use `ebs/train.py` to generate experience.

**Key Arguments**

- `--experiment_name`: Experiment name.
- `--dataset`: Training files and sample counts.
- `--epochs`: Number of epochs.
- `--batchsize`: Samples per batch.
- `--grpo_n`: Rollouts per input.
- `--rollout_concurrency`: Number of concurrent rollout tasks.
- `--rollout_temperature`: Generation temperature.
- `--rollout_max_tokens`: Maximum output tokens.
- `--experience_update_method`: Use `critique` or `summary` to update experience.
- `--task_timeout`: Timeout per task in seconds.

The following command uses 20 harmful, 20 benign, and 20 ethics samples with the generation settings from Section
4.3 of the paper:

```bash
uv run python ebs/train.py \
  --mode agent \
  --domain ebs \
  --experiment_name ebs_three_bucket \
  --dataset dataset/wildjailbreak_harmful.jsonl:20,dataset/star_benign.jsonl:20,dataset/evil_data.jsonl:20 \
  --epochs 1 \
  --batchsize 20 \
  --grpo_n 5 \
  --rollout_concurrency 5 \
  --rollout_temperature 0.6 \
  --rollout_max_tokens 4096 \
  --experience_update_method critique \
  --task_timeout 1800
```

Place the training files in `dataset/`. See `dataset/README.md` for the accepted format.

### 3. Reuse the Experience Bank

Use `ebs/main.py` to run EBS on a local dataset.

**Key Arguments**

- `--experiment_name`: Experiment name.
- `--dataset`: Evaluation data.
- `--dataset_truncate`: Run only the first N samples.
- `--experience_file`: Experience-bank file.
- `--rollout_concurrency`: Number of concurrent tasks.
- `--rollout_model`: Target model override.
- `--judge_model`: Judge model override.

```bash
uv run python ebs/main.py \
  --mode agent \
  --domain ebs \
  --experiment_name ebs_reuse \
  --dataset dataset/evil_data.jsonl \
  --dataset_truncate 20 \
  --experience_file "exp/experiences 600-300.json" \
  --rollout_concurrency 5
```

Remove `--experience_file` to run Vanilla.

### 4. Run the Main Experiment

Use `eval_scripts.eval_ebs_main_experiment` to run HarmBench and XSTest.

**Key Arguments**

- `--run_mode`: `combined`, `harmbench`, or `xstest`.
- `--benchmark_limits`: Number of samples per benchmark.
- `--target_model`: Target model.
- `--experience_file`: Experience bank used by EBS.

Qwen3-8B Vanilla:

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_vanilla \
  --run_mode combined \
  --benchmark_limits "HarmBench:300" \
  --target_model "Qwen/Qwen3-8B"
```

Qwen3-8B + EBS:

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_ebs \
  --run_mode combined \
  --benchmark_limits "HarmBench:300" \
  --target_model "Qwen/Qwen3-8B" \
  --experience_file "exp/experiences 600-300.json"
```

XSTest:

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_xstest_ebs \
  --run_mode xstest \
  --target_model "Qwen/Qwen3-8B" \
  --experience_file "exp/experiences 600-300.json"
```

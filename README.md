# EBS

## 🚀 Getting Started

Follow the steps below to configure the environment, build an experience bank, and run the paper experiments.

### 1. Setup

Requirements:

- Python 3.12 or later (required by this implementation)
- Git
- `uv`

Clone the repository and enter the project directory:

```bash
git clone https://github.com/fff-6/EBS-Safety-Boundary.git
cd EBS-Safety-Boundary
```

Check Python and install `uv` if it is not already available:

```bash
python --version
python -m pip install --upgrade uv
uv --version
```

Create the local `.venv` and install the dependencies recorded in `uv.lock`:

```bash
uv sync
```

All commands below use `uv run`, so activating the virtual environment is optional. To activate it manually, use:

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Use the environment created by `uv`; avoid installing the project dependencies into a separate Conda or system
environment.

Copy the environment template. On PowerShell:

```powershell
Copy-Item .env.example .env
```

On Linux or macOS:

```bash
cp .env.example .env
```

Edit `.env` and configure the target and judge models. The defaults below match the paper's main model setup:

```ini
EBS_LLM_TYPE=chat.completions
EBS_LLM_MODEL=Qwen/Qwen3-8B
EBS_LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
EBS_LLM_API_KEY=replace-me

TARGET_LLM_MODEL=Qwen/Qwen3-8B
TARGET_LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
TARGET_LLM_API_KEY=replace-me

EBS_JUDGE_MODEL=Qwen/Qwen3-14B
EBS_JUDGE_BASE_URL=https://your-openai-compatible-endpoint/v1
EBS_JUDGE_API_KEY=replace-me

ATTACK_JUDGE_API_KEY=replace-me
OPENAI_API_KEY=replace-me
```

The paper uses Qwen3-8B as the main target model and Qwen3-14B as the automatic judge. Replace the example URLs and
keys with any hosted service or local endpoint that supports OpenAI-compatible Chat Completions. Python 3.12 is the
source-code requirement, not a hardware claim from the paper.

For reference, the paper reports Ubuntu 22.x, an NVIDIA GeForce RTX 4060, and 32 GB RAM. Matching this hardware is
not required when the target and judge models are accessed through APIs; local inference requires suitable GPU memory.

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

# EBS

## 🚀 快速开始

按以下步骤完成环境配置、经验生成和论文主实验。

### 1. 安装环境

在项目根目录安装依赖：

```bash
uv sync
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

Linux 或 macOS：

```bash
cp .env.example .env
```

编辑 `.env`，配置目标模型和 Judge。论文主要使用：

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

运行测试：

```bash
uv run pytest -q
```

### 数据集

本仓库不提供第三方数据集。运行实验前，请按照 [`dataset/README.md`](dataset/README.md) 中的说明，从官方来源下载
RedBench、WildJailbreak、STAR-benign-915、HarmBench 和 XSTest。论文主实验默认读取转换后的
`dataset/redbench_dataset.json`，经验生成使用放在 `dataset/` 下的本地 JSONL 文件。

### 2. 生成经验库

使用 `ebs/train.py` 生成经验。

**主要参数**

- `--experiment_name`：实验名称。
- `--dataset`：训练数据文件及采样数量。
- `--epochs`：训练轮数。
- `--batchsize`：每批样本数。
- `--grpo_n`：每条输入的 Rollout 次数。
- `--rollout_concurrency`：并发数。
- `--rollout_temperature`：生成温度。
- `--rollout_max_tokens`：最大生成长度。
- `--experience_update_method`：使用 `critique` 或 `summary` 更新经验。
- `--task_timeout`：单条任务超时秒数。

以下命令使用 harmful、benign 和 ethics 三类数据，每类取 20 条，并采用论文第 4.3 节的生成参数：

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

训练数据需自行放入 `dataset/`。数据格式见 `dataset/README.md`。

### 3. 复用经验库

使用 `ebs/main.py` 在指定数据上运行 EBS。

**主要参数**

- `--experiment_name`：实验名称。
- `--dataset`：评测数据。
- `--dataset_truncate`：只运行前 N 条数据。
- `--experience_file`：经验库文件。
- `--rollout_concurrency`：并发数。
- `--rollout_model`：目标模型。
- `--judge_model`：Judge 模型。

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

删除 `--experience_file` 即为 Vanilla。

### 4. 运行论文主实验

使用 `eval_scripts.eval_ebs_main_experiment` 运行 HarmBench 和 XSTest。

**主要参数**

- `--run_mode`：`combined`、`harmbench` 或 `xstest`。
- `--benchmark_limits`：各 Benchmark 的样本数量。
- `--target_model`：目标模型。
- `--experience_file`：EBS 使用的经验库。

Qwen3-8B Vanilla：

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_vanilla \
  --run_mode combined \
  --benchmark_limits "HarmBench:300" \
  --target_model "Qwen/Qwen3-8B"
```

Qwen3-8B + EBS：

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_ebs \
  --run_mode combined \
  --benchmark_limits "HarmBench:300" \
  --target_model "Qwen/Qwen3-8B" \
  --experience_file "exp/experiences 600-300.json"
```

只运行 XSTest：

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_xstest_ebs \
  --run_mode xstest \
  --target_model "Qwen/Qwen3-8B" \
  --experience_file "exp/experiences 600-300.json"
```

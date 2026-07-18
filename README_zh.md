# EBS

EBS 是一种无需参数更新的安全边界校准方法，通过多次 Rollout、通用与风险双重评分及 Critique，将有害、正常和伦理请求中的有效处理策略提炼为分类经验库。
推理时，EBS 根据请求文本路由并检索相关经验注入上下文，从而在不更新模型参数的情况下增强有害请求拒绝能力，同时减少对正常请求的过度拒绝。

## 🚀 快速开始

按以下步骤完成环境配置、经验库构建、经验复用和效果评测。

### 1. 安装环境

环境要求：

- Python 3.12 及以上（本项目源码要求）
- Git
- `uv`

克隆仓库并进入项目目录：

```bash
git clone https://github.com/fff-6/EBS-Safety-Boundary.git
cd EBS-Safety-Boundary
```

检查 Python 版本；如果尚未安装 `uv`，使用 pip 安装：

```bash
python --version
python -m pip install --upgrade uv
uv --version
```

根据 `uv.lock` 创建项目虚拟环境 `.venv` 并安装依赖：

```bash
uv sync
```

后续命令均使用 `uv run`，因此不必手动激活虚拟环境。如需激活，PowerShell 使用：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux 或 macOS 使用：

```bash
source .venv/bin/activate
```

建议统一使用 `uv` 创建的环境，不要再将项目依赖混装到其他 Conda 环境或系统 Python 中。

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

Linux 或 macOS：

```bash
cp .env.example .env
```

编辑 `.env`，配置目标模型和 Judge。项目默认使用以下模型组合：

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

项目默认使用 Qwen3-8B 作为目标模型，Qwen3-14B 作为自动 Judge。请将示例 URL 和 API Key 替换为任意
支持 OpenAI-compatible Chat Completions 的托管服务或本地接口。Python 3.12 是源码运行要求，与推理
硬件无关。

参考运行环境为 Ubuntu 22.x、NVIDIA GeForce RTX 4060 和 32 GB 内存。通过 API 调用目标模型和 Judge
时不要求使用相同硬件；只有本地推理时才需要根据模型大小准备相应的显存。

运行测试：

```bash
uv run pytest -q
```

### 数据集

本仓库不提供第三方数据集。运行实验前，请按照 [`dataset/README.md`](dataset/README.md) 中的说明，从官方来源下载
RedBench、WildJailbreak、STAR-benign-915、HarmBench 和 XSTest。完整评测默认读取转换后的
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

完整经验库构建使用 400 条 harmful、200 条 benign 和 200 条 ethics 数据。每条样本生成 5 次相互独立的原始
rollout；前序批次积累的经验不会回灌到后续 rollout。

```bash
uv run python ebs/train.py \
  --mode agent \
  --domain ebs \
  --experiment_name ebs_full_experience_construction \
  --dataset dataset/wildjailbreak_harmful.jsonl:400,dataset/star_benign.jsonl:200,dataset/evil_data.jsonl:200 \
  --epochs 1 \
  --batchsize 20 \
  --grpo_n 5 \
  --rollout_concurrency 5 \
  --rollout_temperature 0.6 \
  --rollout_max_tokens 4096 \
  --experience_update_method critique \
  --task_timeout 1800
```

训练数据需自行放入 `dataset/`。数据格式见 `dataset/README.md`。完整命令也可通过
`scripts/run_full_experience_construction.ps1` 执行。`--iterative_experience_construction` 用于迭代构建消融，
默认完整流程不启用该参数。

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
  --experience_file "artifacts/experience_banks/ebs_full_800samples_4000rollouts.json" \
  --rollout_concurrency 5
```

删除 `--experience_file` 即为 Vanilla。

### 4. 运行主实验

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
  --benchmark_limits "HarmBench:320" \
  --target_model "Qwen/Qwen3-8B"
```

Qwen3-8B + EBS：

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_ebs \
  --run_mode combined \
  --benchmark_limits "HarmBench:320" \
  --target_model "Qwen/Qwen3-8B" \
  --experience_file "artifacts/experience_banks/ebs_full_800samples_4000rollouts.json"
```

只运行 XSTest：

```bash
uv run python -m eval_scripts.eval_ebs_main_experiment \
  --experiment_name qwen3_8b_xstest_ebs \
  --run_mode xstest \
  --target_model "Qwen/Qwen3-8B" \
  --experience_file "artifacts/experience_banks/ebs_full_800samples_4000rollouts.json"
```

项目检索默认值为：`K=8`、路由阈值 `delta=0.35`，低置信度时严格按主桶 6 条、次桶 2 条检索。
当前默认使用确定性的 256 维哈希向量，并将检索 token budget 设为 `0`（不截断），从而保证实际注入
数量与 K 一致。运行完整实验前，请先执行经验库构建脚本，生成
`ebs_full_800samples_4000rollouts.json`。

XSTest 主比较固定使用 `42,43,44,45,46` 五个随机种子，可运行
`scripts/run_xstest_five_seeds.ps1` 自动执行并聚合。目标模型默认参数为 `temperature=0.6`、
`top_p=0.9`、`max_tokens=256`；Qwen3-14B Judge 默认为 `temperature=0.2`、`top_p=0.9`、
`max_tokens=16`。

## 完整流程

正式复现前运行 `uv run python scripts/validate_experiment_inputs.py` 检查数据数量、重复项和构建集/评测集重叠；Smoke Test 仅用于验证环境，不能作为正式结果。

依次运行 `.\scripts\run_full_experience_construction.ps1` 和 `.\scripts\run_xstest_five_seeds.ps1`，即可重建 800 样本、4000 Rollout 的冻结经验库，并完成 Vanilla/EBS 五随机种子 XSTest 对照与聚合。正式主实验使用 HarmBench 320 条和完整 XSTest（250 safe + 200 unsafe contrast）。

正式结果使用完整构建生成的 `ebs_full_800samples_4000rollouts.json` 及其 manifest。

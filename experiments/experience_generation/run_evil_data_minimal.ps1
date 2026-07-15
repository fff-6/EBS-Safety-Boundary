$ErrorActionPreference = "Stop"

$experimentName = "evil_data_minimal_v1"
$datasetPath = "dataset/evil_data.jsonl"
$datasetTruncate = 16
$batchSize = 8
$epochs = 1
$grpoN = 2
$rolloutConcurrency = 2
$rolloutTemperature = 0.6
$rolloutMaxTokens = 2048
$taskTimeout = 900

Write-Host "Running minimal evil_data GRPO experiment: $experimentName"

uv run python ebs/train.py `
  --mode agent `
  --domain ebs `
  --experiment_name $experimentName `
  --dataset $datasetPath `
  --dataset_truncate $datasetTruncate `
  --given_ground_truth True `
  --epochs $epochs `
  --batchsize $batchSize `
  --grpo_n $grpoN `
  --rollout_concurrency $rolloutConcurrency `
  --rollout_temperature $rolloutTemperature `
  --rollout_max_tokens $rolloutMaxTokens `
  --task_timeout $taskTimeout

$ErrorActionPreference = "Stop"

$experimentName = "three_bucket_minimal_v1"
$datasetRecipe = "dataset/wildjailbreak_harmful.jsonl:8,dataset/star_benign.jsonl:8,dataset/evil_data.jsonl:8"
$batchSize = 8
$epochs = 1
$grpoN = 2
$rolloutConcurrency = 2
$rolloutTemperature = 0.6
$rolloutMaxTokens = 2048
$taskTimeout = 900

Write-Host "Running minimal three-bucket GRPO experiment: $experimentName"
Write-Host "Dataset recipe: $datasetRecipe"

uv run python ebs/train.py `
  --mode agent `
  --domain ebs `
  --experiment_name $experimentName `
  --dataset $datasetRecipe `
  --given_ground_truth True `
  --epochs $epochs `
  --batchsize $batchSize `
  --grpo_n $grpoN `
  --rollout_concurrency $rolloutConcurrency `
  --rollout_temperature $rolloutTemperature `
  --rollout_max_tokens $rolloutMaxTokens `
  --task_timeout $taskTimeout

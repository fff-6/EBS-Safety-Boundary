$ErrorActionPreference = "Stop"

$experimentName = "ebs_three_bucket_400_200_200_no_critique"
$datasetRecipe = "dataset/wildjailbreak_harmful.jsonl:400,dataset/star_benign.jsonl:200,dataset/evil_data.jsonl:200"
$batchSize = 100
$epochs = 1
$grpoN = 1
$rolloutConcurrency = 1
$rolloutTemperature = 0.6
$rolloutMaxTokens = 2048
$taskTimeout = 900
$experienceUpdateMethod = "summary"

Write-Host "Running no-critique three-bucket GRPO experiment: $experimentName"
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
  --task_timeout $taskTimeout `
  --experience_update_method $experienceUpdateMethod

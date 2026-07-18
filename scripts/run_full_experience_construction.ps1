$ErrorActionPreference = "Stop"

uv run python scripts/validate_experiment_inputs.py

uv run python ebs/train.py `
  --mode agent `
  --domain ebs `
  --experiment_name ebs_full_experience_construction `
  --dataset "dataset/wildjailbreak_harmful.jsonl:400,dataset/star_benign.jsonl:200,dataset/evil_data.jsonl:200" `
  --given_ground_truth True `
  --epochs 1 `
  --batchsize 20 `
  --grpo_n 5 `
  --rollout_concurrency 5 `
  --rollout_temperature 0.6 `
  --rollout_max_tokens 4096 `
  --task_timeout 1800 `
  --experience_update_method critique

$source = "data/ebs/train/ebs_full_experience_construction/step_40/experiences.json"
$destination = "artifacts/experience_banks/ebs_full_800samples_4000rollouts.json"
if (-not (Test-Path -LiteralPath $source)) {
  throw "Full construction did not produce the expected final bank: $source"
}
Copy-Item -LiteralPath $source -Destination $destination -Force
$bank = Get-Content -LiteralPath $destination -Raw -Encoding UTF8 | ConvertFrom-Json
$manifest = [ordered]@{
  artifact = (Split-Path -Leaf $destination)
  status = "generated_with_full_protocol"
  created_at = (Get-Date).ToUniversalTime().ToString("o")
  git_commit = (git rev-parse HEAD).Trim()
  sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
  construction_samples = [ordered]@{ harmful = 400; benign = 200; ethics = 200; total = 800 }
  rollouts_per_sample = 5
  total_rollouts = 4000
  experience_update_method = "critique"
  iterative_experience_construction = $false
  random_seed = 42
  rule_counts = [ordered]@{
    harmful = @($bank.harmful.PSObject.Properties).Count
    benign = @($bank.benign.PSObject.Properties).Count
    ethics = @($bank.ethics.PSObject.Properties).Count
  }
}
$manifest.rule_counts.total = $manifest.rule_counts.harmful + $manifest.rule_counts.benign + $manifest.rule_counts.ethics
$manifestPath = "artifacts/experience_banks/ebs_full_800samples_4000rollouts.manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "Exported regenerated experience bank to $destination"

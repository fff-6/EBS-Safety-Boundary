$ErrorActionPreference = "Stop"

$seeds = 42, 43, 44, 45, 46
$experienceFile = "artifacts/experience_banks/ebs_full_800samples_4000rollouts.json"
if (-not (Test-Path -LiteralPath $experienceFile)) {
  throw "Missing regenerated full experience bank. Run scripts/run_full_experience_construction.ps1 first."
}
foreach ($seed in $seeds) {
  uv run python -m eval_scripts.eval_ebs_main_experiment `
    --experiment_name "qwen3_8b_vanilla_seed_$seed" `
    --run_mode xstest `
    --seed $seed `
    --target_model "Qwen/Qwen3-8B"

  uv run python -m eval_scripts.eval_ebs_main_experiment `
    --experiment_name "qwen3_8b_ebs_seed_$seed" `
    --run_mode xstest `
    --seed $seed `
    --target_model "Qwen/Qwen3-8B" `
    --experience_file $experienceFile
}

uv run python -m eval_scripts.aggregate_seed_metrics `
  --input_glob "data/ebs/eval/main_experiment/xstest/qwen3_8b_vanilla_seed_*/main_metrics.json" `
  --output "data/ebs/eval/main_experiment/xstest/qwen3_8b_vanilla_five_seed_aggregate.json"

uv run python -m eval_scripts.aggregate_seed_metrics `
  --input_glob "data/ebs/eval/main_experiment/xstest/qwen3_8b_ebs_seed_*/main_metrics.json" `
  --output "data/ebs/eval/main_experiment/xstest/qwen3_8b_ebs_five_seed_aggregate.json"

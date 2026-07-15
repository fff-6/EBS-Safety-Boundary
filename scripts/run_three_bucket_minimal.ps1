$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $scriptDir "..\\experiments\\experience_generation\\run_three_bucket_minimal.ps1"

& $target

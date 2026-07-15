$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $scriptDir "..\\experiments\\experience_generation\\run_evil_data_minimal.ps1"

& $target

param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$PythonExe = "C:\ProgramData\Anaconda3\envs\p311cuda\python.exe"
)

$ErrorActionPreference = "Stop"
$trainerDir = Join-Path $RepoRoot "trainer"
$targetTrainer = Join-Path $trainerDir "trainer.py"
$targetSSOD = Join-Path $trainerDir "ssod_trainer.py"
$sourceTrainer = Join-Path $PSScriptRoot "patched\trainer\trainer.py"
$sourceSSOD = Join-Path $PSScriptRoot "patched\trainer\ssod_trainer.py"

foreach ($path in @($targetTrainer, $targetSSOD, $sourceTrainer, $sourceSSOD)) {
    if (-not (Test-Path $path)) {
        throw "Required file not found: $path"
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $targetTrainer "$targetTrainer.before_staged_ssl_$stamp.bak" -Force
Copy-Item $targetSSOD "$targetSSOD.before_staged_ssl_$stamp.bak" -Force
Copy-Item $sourceTrainer $targetTrainer -Force
Copy-Item $sourceSSOD $targetSSOD -Force

& $PythonExe -m py_compile $targetTrainer $targetSSOD
if ($LASTEXITCODE -ne 0) {
    throw "Python syntax check failed. Restore the .bak files."
}

Write-Host "Patch installed successfully." -ForegroundColor Green
Write-Host "Backups:" -ForegroundColor Cyan
Write-Host "  $targetTrainer.before_staged_ssl_$stamp.bak"
Write-Host "  $targetSSOD.before_staged_ssl_$stamp.bak"
Write-Host "Syntax check: PASS" -ForegroundColor Green

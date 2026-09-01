[CmdletBinding()]
param(
    [switch]$SkipPerformance
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Label,
        [Parameter(Mandatory)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments)]
        [string[]]$Arguments
    )

    Write-Host "`n==> $Label"
    & $Command @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Push-Location $projectRoot

try {
    Invoke-Checked "Python lint" $python -m ruff check .
    Invoke-Checked "Python formatting" $python -m ruff format --check .
    Invoke-Checked "Python type checking" $python -m mypy edgeforge_core edgeforge_api apps worker scripts
    Invoke-Checked "Python tests" $python -m pytest -q

    if (-not $SkipPerformance) {
        Invoke-Checked "Performance regression" $python -m scripts.check_performance_regression
    }

    Push-Location (Join-Path $projectRoot "dashboard")

    try {
        Invoke-Checked "Dashboard lint" "npm" run lint
        Invoke-Checked "Dashboard build" "npm" run build
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}

Write-Host "`nAll EdgeForge checks passed."

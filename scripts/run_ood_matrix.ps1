param(
    [string]$PythonExe = "python",
    [string]$OodSplitRoot = "data/splits_labeled/ood",
    [string]$OutputRoot = "results_ood",
    [string[]]$Models = @("simplecnn", "resnet", "artifactnet")
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Title,
        [string]$Command
    )

    Write-Host ""
    Write-Host "==== $Title ====" -ForegroundColor Cyan
    Write-Host $Command -ForegroundColor DarkGray
    Invoke-Expression $Command
}

$familyDirs = Get-ChildItem -Path $OodSplitRoot -Directory | Sort-Object Name

foreach ($model in $Models) {
    foreach ($familyDir in $familyDirs) {
        $heldOutFamily = $familyDir.Name
        $resultDir = Join-Path $OutputRoot "$model\$heldOutFamily"

        Run-Step `
            -Title "Train $model with held-out family $heldOutFamily" `
            -Command "$PythonExe src/train_cropped_baseline.py --model $model --train-split `"$($familyDir.FullName)\train.txt`" --val-split `"$($familyDir.FullName)\val.txt`" --test-split `"$($familyDir.FullName)\test.txt`" --output-root `"$resultDir`""

        Run-Step `
            -Title "Plot evaluation for $model / $heldOutFamily" `
            -Command "$PythonExe src/plot_evaluation_suite.py --experiment-dir `"$resultDir`""
    }
}

$experimentDirs = @()
foreach ($model in $Models) {
    foreach ($familyDir in $familyDirs) {
        $experimentDirs += (Join-Path $OutputRoot "$model\$($familyDir.Name)")
    }
}

$experimentArgs = $experimentDirs -join " "

Run-Step `
    -Title "Build OOD benchmark report" `
    -Command "$PythonExe src/build_ood_benchmark_report.py --experiment-dirs $experimentArgs --output-dir results_reports"

Write-Host ""
Write-Host "OOD matrix suite tamamlandi." -ForegroundColor Green

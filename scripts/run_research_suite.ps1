param(
    [string]$PythonExe = "python",
    [string]$InputList = "data/fma_subset_1000.txt",
    [string]$RealSpecDir = "data/processed/fma_specs",
    [string]$FakeAudioRoot = "data/reconstructed",
    [string]$FakeSpecDir = "data/processed/fake_specs",
    [string]$SplitDir = "data/splits_labeled",
    [string]$ReportDir = "results_reports"
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

Run-Step `
    -Title "Build real spectrograms" `
    -Command "$PythonExe src/build_spectrogram_dataset.py --input-list `"$InputList`" --output-dir `"$RealSpecDir`""

Run-Step `
    -Title "Generate Griffin-Lim families" `
    -Command "$PythonExe src/generate_griffinlim_recons.py --input-list `"$InputList`" --output-root `"$FakeAudioRoot`" --family-name griffinlim_mel32 --n-iter 32"

Run-Step `
    -Title "Generate fast Griffin-Lim family" `
    -Command "$PythonExe src/generate_griffinlim_recons.py --input-list `"$InputList`" --output-root `"$FakeAudioRoot`" --family-name griffinlim_mel8 --n-iter 8"

Run-Step `
    -Title "Generate degradation families" `
    -Command "$PythonExe src/generate_degradation_recons.py --input-list `"$InputList`" --output-root `"$FakeAudioRoot`" --families resample_8k quantize_8bit smoothed_noise"

Run-Step `
    -Title "Build fake spectrogram families" `
    -Command "$PythonExe src/build_fake_spectrogram_dataset.py --input-dir `"$FakeAudioRoot`" --output-dir `"$FakeSpecDir`""

Run-Step `
    -Title "Create leakage-free labeled split" `
    -Command "$PythonExe src/create_labeled_split.py --real-dir `"$RealSpecDir`" --fake-dir `"$FakeSpecDir`" --output-dir `"$SplitDir`""

Run-Step `
    -Title "Create OOD held-out family splits" `
    -Command "$PythonExe src/create_ood_splits.py --manifest-path `"$SplitDir/manifest.csv`" --output-dir `"$SplitDir/ood`""

Run-Step `
    -Title "Train SimpleCNN baseline" `
    -Command "$PythonExe src/train_cropped_baseline.py --model simplecnn --train-split `"$SplitDir/train.txt`" --val-split `"$SplitDir/val.txt`" --test-split `"$SplitDir/test.txt`" --output-root results_simplecnn"

Run-Step `
    -Title "Train ResNet baseline" `
    -Command "$PythonExe src/train_cropped_baseline.py --model resnet --train-split `"$SplitDir/train.txt`" --val-split `"$SplitDir/val.txt`" --test-split `"$SplitDir/test.txt`" --output-root results_resnet"

Run-Step `
    -Title "Train ArtifactNet" `
    -Command "$PythonExe src/train_cropped_baseline.py --model artifactnet --train-split `"$SplitDir/train.txt`" --val-split `"$SplitDir/val.txt`" --test-split `"$SplitDir/test.txt`" --output-root results_artifactnet"

Run-Step `
    -Title "Train Attention MIL" `
    -Command "$PythonExe src/train_clip_attention.py --train-split `"$SplitDir/train.txt`" --val-split `"$SplitDir/val.txt`" --test-split `"$SplitDir/test.txt`" --output-root results_attention"

Run-Step `
    -Title "Plot evaluation suites" `
    -Command "$PythonExe src/plot_evaluation_suite.py --experiment-dir results_simplecnn"

Run-Step `
    -Title "Plot evaluation suites (ResNet)" `
    -Command "$PythonExe src/plot_evaluation_suite.py --experiment-dir results_resnet"

Run-Step `
    -Title "Plot evaluation suites (ArtifactNet)" `
    -Command "$PythonExe src/plot_evaluation_suite.py --experiment-dir results_artifactnet"

Run-Step `
    -Title "Plot evaluation suites (Attention)" `
    -Command "$PythonExe src/plot_evaluation_suite.py --experiment-dir results_attention"

Run-Step `
    -Title "Build benchmark report" `
    -Command "$PythonExe src/build_benchmark_report.py --experiment-dirs results_simplecnn results_resnet results_artifactnet results_attention --output-dir `"$ReportDir`""

Write-Host ""
Write-Host "Research suite tamamlandi." -ForegroundColor Green

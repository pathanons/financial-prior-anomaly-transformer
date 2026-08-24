param(
    [string]$Python = "C:\Users\Acer\miniconda3\python.exe",
    [string]$RunRoot = "D:\multi-prior-at-run-state",
    [string]$ExperimentName = "AT-State",
    [string]$PanelOut = "tmp\panels\AT-State",
    [string]$Tickers = "AKAM,AVGO,BLDR,LEN",
    [int]$NumEpochs = 7,
    [int]$Seed = 0,
    [int]$StateProjectionDim = 0
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $Repo

try {
    $common = @(
        "--dataset", "STOCK",
        "--data_path", "SP500",
        "--win_size", "60",
        "--batch_size", "32",
        "--num_epochs", "$NumEpochs",
        "--seed", "$Seed",
        "--train_start", "2018-01-01",
        "--train_end", "2021-12-31",
        "--val_start", "2022-01-01",
        "--val_end", "2022-12-31",
        "--test_start", "2023-01-01",
        "--test_end", "2024-12-31",
        "--k", "3",
        "--run_root", $RunRoot,
        "--experiment_name", $ExperimentName,
        "--model_save_path", (Join-Path $RunRoot "$ExperimentName\checkpoints"),
        "--prior_type", "state",
        "--score_type", "original",
        "--reconstruction_features", "log_return_1d,volume_z,gap",
        "--label_type", "contextual",
        "--e_layers", "4",
        "--n_heads", "16"
    )
    if ($StateProjectionDim -gt 0) {
        $common += @("--state_projection_dim", "$StateProjectionDim")
    }

    & $Python -u main.py --mode train @common
    & $Python -u main.py --mode test @common

    $runDir = Join-Path $RunRoot $ExperimentName
    & $Python scripts\experiments\plots\plot_at_score_examples.py `
        --scores (Join-Path $runDir "test_timeline_scores.csv") `
        --metrics (Join-Path $runDir "metrics.json") `
        --feature_dir "SP500_features_vw60_lw60" `
        --out_dir $PanelOut `
        --title_prefix $ExperimentName `
        --start "2023-01-01" `
        --end "2024-12-31" `
        --threshold_quantile 0.99 `
        --include_tickers $Tickers
}
finally {
    Pop-Location
}

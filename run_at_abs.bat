C:\Users\Acer\miniconda3\python.exe scripts\experiments\run_model_grid.py ^
  --model state ^
  --folds C ^
  --seeds 0 1 2 ^
  --tag linear_residual_mlp ^
  --token_embedding linear_residual_mlp

C:\Users\Acer\miniconda3\python.exe scripts\experiments\plots\plot_at_score_examples.py ^
  --scores "D:\multi-prior-at-run-walkforward-multiseed\AT-State_foldC_seed0_linear_residual_mlp\test_timeline_scores.csv" ^
  --metrics "D:\multi-prior-at-run-walkforward-multiseed\AT-State_foldC_seed0_linear_residual_mlp\metrics.json" ^
  --feature_dir "SP500_features_vw60_lw60" ^
  --out_dir "tmp\panels\AT-State-LinearResidualMLP" ^
  --title_prefix "AT-State LinearResidualMLP" ^
  --start "2023-01-01" ^
  --end "2024-12-31" ^
  --threshold_quantile 0.99 ^
  --top_n 25
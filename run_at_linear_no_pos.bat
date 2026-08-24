cd /d C:\Users\Acer\Documents\cmu\research\multi-prior-level\financial-prior-anomaly-transformer

C:\Users\Acer\miniconda3\python.exe scripts\experiments\run_model_grid.py ^
  --model state ^
  --folds C ^
  --seeds 0 1 2 ^
  --tag linear_token_no_pos ^
  --token_embedding linear ^
  --no_positional_embedding

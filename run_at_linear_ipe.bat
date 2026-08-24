cd /d C:\Users\Acer\Documents\cmu\research\multi-prior-level\financial-prior-anomaly-transformer

C:\Users\Acer\miniconda3\python.exe scripts\experiments\run_model_grid.py ^
  --model state ^
  --folds C ^
  --seeds 0 1 2 ^
  --tag linear_token_ipe_s1 ^
  --token_embedding linear ^
  --position_sigma 1.0

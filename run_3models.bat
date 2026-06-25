@echo off
setlocal

cd /d "%~dp0"
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" base
if errorlevel 1 exit /b 1

set RUN_ROOT=D:\multi-prior-at-run
set EVENT_TICKER=AKAM
set COMMON=main.py --dataset STOCK --data_path SP500 --win_size 60 --batch_size 32 --num_epochs 5 --train_start 2018-01-01 --train_end 2021-12-31 --val_start 2022-01-01 --val_end 2022-12-31 --test_start 2023-01-01 --test_end 2023-12-31 --k 3 --run_root %RUN_ROOT%

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"

call :run_model AT-Time "--prior_type time --score_type original"
if errorlevel 1 exit /b 1

call :run_model AT-TimeState "--prior_type time_state --score_type original"
if errorlevel 1 exit /b 1

call :run_model AT-TimeState-ReturnNLL "--prior_type time_state --use_return_nll true --nll_weight 0.1 --return_loss_weight 0.1 --score_type return_nll"
if errorlevel 1 exit /b 1

echo Done. Results are in %RUN_ROOT%
exit /b 0

:run_model
set EXP=%~1
set ARGS=%~2
set EXP_DIR=%RUN_ROOT%\%EXP%
if not exist "%EXP_DIR%\logs" mkdir "%EXP_DIR%\logs"

echo === %EXP% train ===
python -u %COMMON% --experiment_name %EXP% --mode train --model_save_path "%EXP_DIR%\checkpoints" %ARGS% > "%EXP_DIR%\logs\train.log" 2>&1
if errorlevel 1 exit /b 1

echo === %EXP% test ===
python -u %COMMON% --experiment_name %EXP% --mode test --model_save_path "%EXP_DIR%\checkpoints" %ARGS% > "%EXP_DIR%\logs\test.log" 2>&1
if errorlevel 1 exit /b 1

echo === %EXP% visualize ===
python -u %COMMON% --experiment_name %EXP% --mode visualize --model_save_path "%EXP_DIR%\checkpoints" %ARGS% --auto_case_ticker %EVENT_TICKER% > "%EXP_DIR%\logs\visualize.log" 2>&1
if errorlevel 1 exit /b 1

exit /b 0

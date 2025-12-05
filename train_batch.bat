@echo off
setlocal enabledelayedexpansion

set ADDNAME=proplus1

for %%I in (1) do (
    for %%W in (96) do (
        echo ==========================================
        echo Running configuration: idx=%%I, window=%%W
        echo ==========================================

        python chg_cmapsscfg.py --idx %%I --window %%W --addname %ADDNAME%

        echo Running main.py for cmapsst%ADDNAME%%%I_%%W ...
        python main.py --name cmapsst%ADDNAME%%%I_%%W --config_file ./Config/cmapss.yaml --gpu 0 --train

        echo Finished run for idx=%%I, window=%%W
        echo.
    )
)

echo All runs completed.

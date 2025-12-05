@echo off
setlocal enabledelayedexpansion
set DATASET=cmapsst

for %%M in (fid) do (
    set "METRIC=%%~M"
    echo ******************************************
    echo Running METRIC=!METRIC!
    echo ******************************************

    for %%A in ( proplus ) do (
        set "ADDNAME=%%~A"
        echo ------------------------------------------
        echo Running ADDNAME=!ADDNAME!
        echo ------------------------------------------

        for %%I in (1) do (
            for %%W in (96) do (
                echo ==========================================
                echo Running configuration: metric=!METRIC!, idx=%%I, window=%%W, addname=!ADDNAME!
                echo ==========================================

                python chg_cmapsscfg.py --idx %%I --window %%W --addname %ADDNAME%

                echo Running eval.py for cmapsst!ADDNAME!%%I_%%W ...
                python eval.py --dataset %DATASET% --addname "!ADDNAME!" --index %%I --window %%W --metric "!METRIC!" 

                echo Finished run for metric=!METRIC!, idx=%%I, window=%%W, addname=!ADDNAME!
                echo.
            )
        )
    )
)

echo All runs completed.

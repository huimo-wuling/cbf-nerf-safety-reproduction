@echo off
rem ---------------------------------------------------------------
rem  Run the paper experiments with the authors' original code.
rem  This file is intentionally pure ASCII so that no console
rem  code page can break it.
rem
rem  Usage:  run_upstream.bat            (single integrator)
rem          run_upstream.bat double     (double integrator)
rem          run_upstream.bat all        (both)
rem
rem  Requires: python (or set PYTHON to a full path) with torch.
rem ---------------------------------------------------------------
setlocal
chcp 65001 >nul
set "HERE=%~dp0"
set "UP=%HERE%upstream"
set "LOGS=%HERE%logs"
if not exist "%LOGS%" mkdir "%LOGS%"

set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
set PYTHONIOENCODING=utf-8

set "PY=%PYTHON%"
if not defined PY set "PY=python"

set "WHICH=%~1"
if "%WHICH%"=="" set "WHICH=single"

echo == repo: %HERE%
"%PY%" --version || goto :err

echo.
echo == [1/2] dataset stub
"%PY%" "%HERE%tools\make_dataset_stub.py" || goto :err

echo.
if /i "%WHICH%"=="single" goto :single
if /i "%WHICH%"=="double" goto :double
if /i "%WHICH%"=="all"    goto :all
echo usage: run_upstream.bat [single^|double^|all]
exit /b 1

:single
call :run single_cbf.py single_upstream.log
"%PY%" "%HERE%tools\plot_from_log.py" "%LOGS%\single_upstream.log" --csv || goto :err
goto :done

:double
call :run double_cbf.py double_upstream.log
"%PY%" "%HERE%tools\plot_from_log.py" "%LOGS%\double_upstream.log" --csv || goto :err
goto :done

:all
call :run single_cbf.py single_upstream.log
call :run double_cbf.py double_upstream.log
"%PY%" "%HERE%tools\plot_from_log.py" "%LOGS%\single_upstream.log" "%LOGS%\double_upstream.log" --csv || goto :err
"%PY%" "%HERE%tools\frames_from_video.py" "%UP%\video.avi" || goto :err
goto :done

:run
echo == [2/2] run upstream %~1  (log: logs\%~2)
pushd "%UP%"
"%PY%" -u "%~1" configs/Replica/room1.yaml --output output/Replica/room1 > "%LOGS%\%~2" 2>&1
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" echo    exit code: %RC%
exit /b 0

:err
echo.
echo FAILED. Check the message above.
exit /b 1

:done
echo.
echo == outputs
echo    %UP%\video.avi                 (single integrator, 2400x680)
echo    %LOGS%\*.log                   (raw stdout of the upstream scripts)
echo    %HERE%results\figures\         (PNG figures and metrics CSV)
exit /b 0

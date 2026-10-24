@echo off
rem ================================================================
rem  Reproduce the ICRA 2023 CBF+NeRF results with the authors'
rem  original code (upstream/ is unmodified except one line).
rem
rem  Just run:      run_upstream.bat
rem  Or with arg:   run_upstream.bat single | double | all
rem
rem  The interpreter is detected automatically (a python that has
rem  CUDA torch). Override with:  set PYTHON=D:\path\python.exe
rem
rem  This file is pure ASCII on purpose: no console code page can
rem  break it (unlike a UTF-8 .ps1 without BOM under PowerShell 5.1).
rem ================================================================
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "HERE=%~dp0"
set "UP=%HERE%upstream"
set "LOGS=%HERE%logs"
if not exist "%LOGS%" mkdir "%LOGS%"

set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
set PYTHONIOENCODING=utf-8

set "WHICH=%~1"
if "%WHICH%"=="" set "WHICH=single"

rem ---------- find a python that has CUDA torch ----------
set "PY="
call :try "%PYTHON%"
call :try "%NERF_PYTHON%"
call :try "%CONDA_PREFIX%\python.exe"
for %%R in ("D:\miniconda\envs" "D:\Anaconda3\envs" "C:\ProgramData\miniconda3\envs" "%USERPROFILE%\miniconda3\envs" "%USERPROFILE%\anaconda3\envs") do (
  if exist "%%~R" (
    for /d %%E in ("%%~R\*") do call :try "%%~E\python.exe"
  )
)
call :try "python"

if not defined PY (
  echo.
  echo [ERROR] No python with CUDA torch was found.
  echo         Install one, or point to it explicitly:
  echo             set PYTHON=D:\miniconda\envs\nerf-cbf\python.exe
  echo             run_upstream.bat %WHICH%
  exit /b 1
)
echo == interpreter: %PY%

echo.
echo == [1/2] placeholder dataset
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
echo == [2/2] run upstream %~1   (log: logs\%~2)
"%PY%" "%HERE%tools\run_and_tee.py" "%LOGS%\%~2" "%UP%\%~1" configs/Replica/room1.yaml --output output/Replica/room1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo    exit code: %RC%
exit /b 0

:try
if "%~1"=="" exit /b 0
if defined PY exit /b 0
"%~1" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 set "PY=%~1"
exit /b 0

:err
echo.
echo FAILED. Read the message above.
exit /b 1

:done
echo.
echo == done. Outputs:
echo    %UP%\video.avi                     surface sequence, paper Fig.2
echo    %LOGS%\*.log                       raw stdout of the upstream scripts
echo    %HERE%results\figures\             PNG figures and metrics CSV
exit /b 0

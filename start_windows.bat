@echo off
setlocal
set PYTHONUTF8=1
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 goto use_py311
py -3.10 -c "import sys" >nul 2>nul
if not errorlevel 1 goto use_py310
goto use_python
:use_py311
py -3.11 bootstrap.py %*
goto check_result
:use_py310
py -3.10 bootstrap.py %*
goto check_result
:use_python
python bootstrap.py %*
:check_result
if errorlevel 1 (
  echo.
  echo Failed. Check config.json and the error above.
  pause
  exit /b 1
)
echo.
echo Finished. See outputs/latest_run.json.
pause

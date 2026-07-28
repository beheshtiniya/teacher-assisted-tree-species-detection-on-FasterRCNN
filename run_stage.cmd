@echo off
setlocal EnableExtensions
if "%~1"=="" (echo Usage: run_stage.cmd STAGE [--dry-run] [--force] & exit /b 2)
set "CFG=%~dp0config\paths.local.json"
if not exist "%CFG%" (echo ERROR: %CFG% not found. Run configure_paths.cmd first. & exit /b 2)
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "(Get-Content -Raw '%CFG%' ^| ConvertFrom-Json).python_executable"`) do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (echo ERROR: python_executable is empty in config. & exit /b 3)
"%PYTHON_EXE%" "%~dp0run_stage.py" %*
exit /b %ERRORLEVEL%

@echo off
setlocal EnableExtensions
if "%~1"=="" goto :USAGE
if "%~2"=="" goto :USAGE
"%~2" "%~dp0configure_paths.py" --data-root "%~1" --python "%~2" %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
:USAGE
echo Usage:
echo   configure_paths.cmd "DATA_ROOT" "PYTHON_EXE"
echo Example:
echo   configure_paths.cmd "E:\FASTRCNN\teacher_student" "C:\ProgramData\Anaconda3\envs\p311cuda\python.exe"
exit /b 2

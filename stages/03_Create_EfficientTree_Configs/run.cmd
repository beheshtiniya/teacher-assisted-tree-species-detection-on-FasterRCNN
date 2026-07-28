@echo off
call "%~dp0..\..\run_stage.cmd" 03 %*
exit /b %ERRORLEVEL%

@echo off
call "%~dp0..\..\run_stage.cmd" 12 %*
exit /b %ERRORLEVEL%

@echo off
call "%~dp0..\..\run_stage.cmd" 10 %*
exit /b %ERRORLEVEL%

@echo off
call "%~dp0..\..\run_stage.cmd" 05 %*
exit /b %ERRORLEVEL%

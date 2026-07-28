@echo off
call "%~dp0..\..\run_stage.cmd" 15 %*
exit /b %ERRORLEVEL%

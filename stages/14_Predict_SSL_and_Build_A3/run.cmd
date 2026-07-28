@echo off
call "%~dp0..\..\run_stage.cmd" 14 %*
exit /b %ERRORLEVEL%

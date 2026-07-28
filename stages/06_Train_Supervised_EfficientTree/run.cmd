@echo off
call "%~dp0..\..\run_stage.cmd" 06 %*
exit /b %ERRORLEVEL%

@echo off
call "%~dp0..\..\run_stage.cmd" 09 %*
exit /b %ERRORLEVEL%

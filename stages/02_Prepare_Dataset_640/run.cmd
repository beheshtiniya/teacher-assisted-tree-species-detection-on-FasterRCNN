@echo off
call "%~dp0..\..\run_stage.cmd" 02 %*
exit /b %ERRORLEVEL%

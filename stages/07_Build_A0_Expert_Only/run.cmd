@echo off
call "%~dp0..\..\run_stage.cmd" 07 %*
exit /b %ERRORLEVEL%

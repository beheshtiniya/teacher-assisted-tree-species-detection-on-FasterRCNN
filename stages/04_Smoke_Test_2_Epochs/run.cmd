@echo off
call "%~dp0..\..\run_stage.cmd" 04 %*
exit /b %ERRORLEVEL%

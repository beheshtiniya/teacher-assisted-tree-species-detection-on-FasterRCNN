@echo off
call "%~dp0..\..\run_stage.cmd" 16 %*
exit /b %ERRORLEVEL%

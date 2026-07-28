@echo off
call "%~dp0..\..\run_stage.cmd" 01 %*
exit /b %ERRORLEVEL%

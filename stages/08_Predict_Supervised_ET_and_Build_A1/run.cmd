@echo off
call "%~dp0..\..\run_stage.cmd" 08 %*
exit /b %ERRORLEVEL%

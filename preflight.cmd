@echo off
call "%~dp0run_stage.cmd" 1 --force %*
exit /b %ERRORLEVEL%

@echo off
call "%~dp0run_stage.cmd" 1 --to 16 %*
exit /b %ERRORLEVEL%

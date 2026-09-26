@echo off
rem Vision launcher for a source checkout on Windows: runs the checkout's virtualenv python.
rem Put this folder on your PATH, or copy the file anywhere and set VISION_HOME to your checkout.
setlocal
set "home=%VISION_HOME%"
if not defined home set "home=%~dp0.."
if not exist "%home%\.venv\Scripts\python.exe" goto novenv
rem UTF-8 for everything Vision reads and writes, whatever the console code page.
set "PYTHONUTF8=1"
"%home%\.venv\Scripts\python.exe" -m vision %*
exit /b %ERRORLEVEL%

:novenv
echo vision: no virtualenv at "%home%\.venv"; set VISION_HOME to your Vision checkout 1>&2
exit /b 1

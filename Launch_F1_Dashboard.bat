```bat
@echo off
title F1 Race Analytics Dashboard

cd /d "%~dp0"

echo ========================================
echo       F1 RACE ANALYTICS DASHBOARD
echo ========================================
echo.
echo Starting the dashboard...
echo Please wait for your browser to open.
echo.

start "" http://127.0.0.1:8050

".venv\Scripts\python.exe" dashboard.py

echo.
echo The dashboard process has stopped.
pause
```
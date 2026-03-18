@echo off
setlocal enabledelayedexpansion
title FractureAI Portable Runner

echo ===================================================
echo   FractureAI Portable Runner (Docker)
echo ===================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    echo and make sure it is running.
    pause
    exit /b 1
)

REM Check if .env exists in backend, create if not
if not exist backend\.env (
    if exist backend\.env.template (
        echo [INFO] Creating backend/.env from template...
        copy backend\.env.template backend\.env >nul
    ) else (
        echo [INFO] Creating new backend/.env file...
        echo DJANGO_DEBUG=True > backend\.env
        echo DJANGO_SECRET_KEY=portable-secret-key-!random! >> backend\.env
    )
)

REM Check if GEMINI_API_KEY is set
set "KEY_FOUND=0"
findstr /C:"GEMINI_API_KEY=AI" backend\.env >nul
if %errorlevel% equ 0 set "KEY_FOUND=1"

if "!KEY_FOUND!"=="0" (
    echo [IMPORTANT] Gemini API Key is required for the Chatbot.
    echo Please get one at: https://aistudio.google.com/app/apikey
    set /p "USER_KEY=Paste your Gemini API Key here (or press Enter to skip): "
    if not "!USER_KEY!"=="" (
        REM Remove existing key line if it exists but is invalid
        findstr /V "GEMINI_API_KEY" backend\.env > backend\.env.tmp
        move /Y backend\.env.tmp backend\.env >nul
        echo GEMINI_API_KEY=!USER_KEY! >> backend\.env
        echo [SUCCESS] API Key saved to backend/.env
    ) else (
        echo [WARNING] Continuing without API key. Chatbot will run in Limited Mode.
    )
)

echo.
echo [1/2] Building and Starting Containers...
echo This may take a few minutes on the first run...
echo.

docker-compose up --build -d --force-recreate

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker-compose failed to start. 
    echo 1. Make sure Docker Desktop is RUNNING.
    echo 2. Check if port 3000 or 8000 is already in use.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   SUCCESS! FractureAI is now running.
echo ===================================================
echo.
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000
echo.
echo Opening browser...
start http://localhost:3000

echo.
echo Keep this window open while using the app.
echo To stop the project, close this window and run: docker-compose down
echo.
pause

@echo off
setlocal

echo ==========================================
echo   FractureAI Project Setup Script
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed. Please install Python and try again.
    pause
    exit /b 1
)

REM Check if NPM is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] NPM/Node.js is not installed. Please install Node.js and try again.
    pause
    exit /b 1
)

echo.
echo [1/4] Setting up Python Virtual Environment...
cd backend
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/4] Installing Backend Dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install backend dependencies.
    pause
    exit /b 1
)
echo Backend dependencies installed successfully.

echo.
echo [3/4] Configuring Environment Variables...
if not exist .env (
    copy .env.template .env
    echo Created .env from .env.template. Please update it with your GEMINI_API_KEY.
) else (
    echo .env file already exists.
)
cd ..

echo.
echo [4/4] Installing Frontend Dependencies...
cd bone-fracture-web
call npm install
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install frontend dependencies.
    pause
    exit /b 1
)
cd ..

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo To run the project, use: run.bat
echo.
pause

@echo off
setlocal

echo ==========================================
echo   FractureAI Project Runner
echo ==========================================

REM Check if backend venv exists
if not exist backend\venv (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Check if .env exists
if not exist backend\.env (
    echo [WARNING] .env file not found in backend/. Creating from template...
    echo Please make sure to update backend/.env with your GEMINI_API_KEY.
    copy backend\.env.template backend\.env
)

echo.
echo Starting Backend Server in a new window...
start "FractureAI Backend" cmd /k "cd backend && call venv\Scripts\activate && python manage.py runserver"

echo.
echo Starting Frontend Server in current window...
echo (Press Ctrl+C to stop)
cd bone-fracture-web
npm start

pause

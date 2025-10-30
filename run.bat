@echo off
REM Script de lancement pour Windows
REM Process Mining Dashboard

echo ========================================
echo  Process Mining Dashboard Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created!
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate
echo.

REM Install/Update dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed!
echo.

REM Launch Streamlit app
echo [INFO] Starting Process Mining Dashboard...
echo [INFO] Opening http://localhost:8501 in your browser...
echo.
echo Press CTRL+C to stop the application
echo ========================================
echo.

streamlit run app.py

pause


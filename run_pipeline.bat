@echo off
REM Batch script for Windows users to run the complete pipeline

echo ========================================
echo Galaxy Morphology Unlearning Pipeline
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/4] Testing setup...
python test_setup.py
if errorlevel 1 (
    echo.
    echo ERROR: Setup test failed
    echo Please fix the issues above
    pause
    exit /b 1
)

echo.
echo [2/4] Running main pipeline...
echo This will take 20-30 minutes. Please wait...
echo.
python main.py
if errorlevel 1 (
    echo.
    echo ERROR: Pipeline failed
    pause
    exit /b 1
)

echo.
echo [3/4] Pipeline complete!
echo.
echo Generated files:
dir /b *.pth 2>nul
dir /b *.csv 2>nul
echo.

echo [4/4] Would you like to launch the web app? (Y/N)
set /p launch="Enter choice: "

if /i "%launch%"=="Y" (
    echo.
    echo Launching Streamlit app...
    echo Open your browser to http://localhost:8501
    echo Press Ctrl+C to stop the server
    echo.
    streamlit run app.py
) else (
    echo.
    echo To launch the web app later, run:
    echo   streamlit run app.py
)

echo.
echo ========================================
echo Pipeline Complete!
echo ========================================
pause

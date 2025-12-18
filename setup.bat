@echo off
REM Quick Setup and Run Script for Windows
REM Image Editing & 3D Asset Generation Pipeline

echo.
echo ========================================
echo Image to 3D Pipeline - Quick Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo ✓ Python found
python --version

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Docker is not installed
    echo You'll need Docker to use Hunyuan 3D model
    echo Download Docker Desktop from https://www.docker.com/products/docker-desktop/
)

echo ✓ Docker found
docker --version

REM Create virtual environment
echo.
echo Creating Python virtual environment...
if not exist venv (
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install requirements
echo.
echo Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed successfully

REM Create output directories
echo.
echo Creating output directories...
if not exist outputs mkdir outputs
if not exist assets mkdir assets
if not exist logs mkdir logs
echo ✓ Directories created

REM Check GPU
echo.
echo Checking GPU availability...
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

REM Verify Hunyuan Docker image
echo.
echo Checking Hunyuan 3D Docker image...
docker inspect feynmann/hunyuan_custom:v1 >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Hunyuan 3D Docker image not found
    echo.
    echo To pull the image, run:
    echo   docker pull feynmann/hunyuan_custom:v1
    echo.
    echo This will take 20-30 minutes and requires ~30GB storage
    echo.
) else (
    echo ✓ Hunyuan 3D image found
)

REM Display options
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Choose how to run the pipeline:
echo.
echo Option 1: Run with Docker Compose (Recommended)
echo   docker-compose up -d
echo.
echo Option 2: Run Gradio Web App
echo   python app.py
echo.
echo Then open your browser to:
echo   http://localhost:7860
echo.
echo For more information, see README.md
echo.

pause

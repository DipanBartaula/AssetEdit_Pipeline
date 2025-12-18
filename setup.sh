#!/bin/bash
# Quick Setup and Run Script for Linux/macOS
# Image Editing & 3D Asset Generation Pipeline

set -e

echo ""
echo "========================================"
echo "Image to 3D Pipeline - Quick Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10+ using your package manager"
    echo "Ubuntu: sudo apt-get install python3.10"
    exit 1
fi

echo "✓ Python found"
python3 --version

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "WARNING: Docker is not installed"
    echo "You'll need Docker to use Hunyuan 3D model"
    echo "Download Docker from https://www.docker.com/products/docker-desktop/"
fi

if command -v docker &> /dev/null; then
    echo "✓ Docker found"
    docker --version
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed successfully"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs assets logs
echo "✓ Directories created"

# Check GPU
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Verify Hunyuan Docker image
if command -v docker &> /dev/null; then
    echo ""
    echo "Checking Hunyuan 3D Docker image..."
    if ! docker inspect feynmann/hunyuan_custom:v1 > /dev/null 2>&1; then
        echo ""
        echo "WARNING: Hunyuan 3D Docker image not found"
        echo ""
        echo "To pull the image, run:"
        echo "  docker pull feynmann/hunyuan_custom:v1"
        echo ""
        echo "This will take 20-30 minutes and requires ~30GB storage"
        echo ""
    else
        echo "✓ Hunyuan 3D image found"
    fi
fi

# Display options
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Choose how to run the pipeline:"
echo ""
echo "Option 1: Run with Docker Compose (Recommended)"
echo "  docker-compose up -d"
echo ""
echo "Option 2: Run Gradio Web App"
echo "  python app.py"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:7860"
echo ""
echo "For more information, see README.md"
echo ""

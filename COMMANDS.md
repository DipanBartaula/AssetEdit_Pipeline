# Quick Execution Commands

## Setup

```bash
# Windows
.\setup.bat

# Linux/macOS
chmod +x setup.sh && ./setup.sh

# Verify installation
python test_setup.py
```

## Build & Deploy

```bash
# Build Docker image
docker build -t image-to-3d:latest .

# Start with Docker Compose
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

## Environment

```bash
# Set Lightning API Key (Windows PowerShell)
$env:LIGHTNING_API_KEY = "your-api-key"

# Set Lightning API Key (Linux/macOS)
export LIGHTNING_API_KEY="your-api-key"

# Or create .env file
echo LIGHTNING_API_KEY=your-api-key > .env
```

## Direct Execution

```bash
# Activate environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Run Gradio app directly
python app.py
```

## Access Interface

```
http://localhost:7860
```

## Cleanup

```bash
# Stop containers
docker-compose down -v

# Clear outputs
rm -rf outputs/*

# Remove image
docker rmi image-to-3d:latest
```

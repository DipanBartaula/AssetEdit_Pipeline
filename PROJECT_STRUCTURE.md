# Project Structure and File Overview

## Complete Project Layout

```
end_to_end_pipeline/
│
├── � Python Source Code
│   ├── app.py                      # Gradio web interface with GLB rendering
│   ├── pipeline.py                 # Pipeline orchestration (Lightning AI)
│   ├── test_setup.py               # Setup verification script
│   │
│   ├── models/                     # AI Model implementations
│   │   ├── __init__.py
│   │   ├── image_editor.py         # QWENImage Edit 2509 integration
│   │   └── hunyuan_3d.py          # Hunyuan 3D 2.1 Lightning AI integration
│   │
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       └── config.py               # Configuration and utilities
│
├── 🐳 Docker Configuration
│   ├── Dockerfile                  # Lightweight Docker build for Gradio
│   ├── docker-compose.yml          # Single service orchestration
│   └── .dockerignore              # Docker build exclusions
│
├── 📦 Python Dependencies
│   └── requirements.txt            # Python package dependencies
│
├── 🔧 Setup Scripts
│   ├── setup.bat                  # Windows setup automation
│   └── setup.sh                   # Linux/macOS setup automation
│
├── 📁 Output Directories (created at runtime)
│   ├── outputs/                   # Generated 3D assets (.glb format)
│   ├── assets/                    # Additional asset storage
│   └── logs/                      # Application logs
│
└── 📚 Documentation
    ├── PROJECT_STRUCTURE.md       # This file
    └── COMMANDS.md                # Execution commands only
```

---

## File Descriptions

### Core Application Files

#### `app.py` - Gradio Web Interface (GLB Support)
- **Purpose:** Interactive web frontend for the complete pipeline
- **Features:**
  - Image upload interface
  - Image editing controls
  - 3D generation controls
  - **GLB 3D Model Viewer** (Gradio Model3D component)
  - Full pipeline one-click execution
  - System status monitoring
- **Port:** 7860 (configurable)
- **Model Support:** .glb format rendering

#### `pipeline.py` - Pipeline Orchestration (Lightning AI)
- **Purpose:** Manages the complete workflow
- **Classes:**
  - `ImageTo3DPipeline` - Main orchestrator
- **Methods:**
  - `process_image_to_edited_image()` - Step 1: Edit
  - `generate_3d_from_image()` - Step 2: Generate 3D (Lightning AI)
  - `run_full_pipeline()` - Complete workflow
- **Backend:** Lightning AI for 3D generation

### Model Implementations

#### `models/image_editor.py` - Image Editing Module
- **Purpose:** QWENImage Edit 2509 model integration
- **Classes:**
  - `QWENImageEditor` - Image editing handler
- **Methods:**
  - `edit_image()` - Edit single image
  - `batch_edit_images()` - Process multiple images

#### `models/hunyuan_3d.py` - 3D Generation Module (Lightning AI)
- **Purpose:** Hunyuan 3D 2.1 via Lightning AI integration
- **Classes:**
  - `Hunyuan3DGenerator` - 3D asset generator
- **Methods:**
  - `generate_3d_asset()` - Generate GLB 3D model
  - `get_asset_preview()` - Retrieve results
  - `list_generated_assets()` - List all assets
- **Output Format:** .glb (web-ready)
- **Backend:** Lightning AI REST API

### Docker Configuration

#### `Dockerfile`
- **Purpose:** Build Docker image for the application
- **Base:** nvidia/cuda:12.1.0-runtime-ubuntu22.04
- **Includes:**
  - Python 3.10 environment
  - Gradio web interface
  - Application code
- **Size:** Lightweight (no model weights)
- **Port:** 7860 (Gradio)

#### `docker-compose.yml`
- **Purpose:** Single-container deployment with Lightning AI
- **Service:**
  - `pipeline-app` - Main application container
- **Environment:**
  - LIGHTNING_API_KEY (required)
- **Port:** 7860
- **Launch:** `docker-compose up -d`

### Utilities

#### `utils/config.py` - Configuration Module
- **Purpose:** Centralized configuration management
- **Classes:**
  - `Config` - Configuration constants
- **Functions:**
  - `setup_environment()` - Initialize environment
  - `setup_directories()` - Create directories
  - `check_docker_installation()` - Verify Docker
  - `check_gpu_availability()` - Verify GPU
  - `get_gpu_info()` - GPU details
- **Constants:**
  - Model names
  - Directory paths
  - Port numbers
  - Default parameters

### Setup and Verification

#### `setup.bat` (Windows)
- **Purpose:** Automated Windows environment setup
- **Steps:**
  1. Check Python installation
  2. Check Docker installation
  3. Create virtual environment
  4. Install dependencies
  5. Create directories
  6. Check GPU
- **Runtime:** ~15 minutes

#### `setup.sh` (Linux/macOS)
- **Purpose:** Automated Unix-like environment setup
- **Runtime:** ~15 minutes

#### `test_setup.py`
- **Purpose:** Verify complete installation
- **Tests:**
  1. Python version
  2. PyTorch installation
  3. Transformers library
  4. Gradio installation
  5. Pillow image library
  6. Docker availability
  7. Directory structure
  8. Required files
  9. Module imports

---

## Key Differences from Docker Version

### Lightning AI Integration
- ✅ No local Docker model hosting required
- ✅ Managed infrastructure via Lightning AI
- ✅ Lighter Docker image (no model weights)
- ✅ Scalable processing via API

### GLB Format Support
- ✅ Web-ready 3D model format
- ✅ Gradio Model3D viewer integration
- ✅ Better compatibility with web platforms
- ✅ Reduced file sizes vs PLY format

### Simplified Deployment
- ✅ Single container architecture
- ✅ No external service dependencies
- ✅ Environment variable based configuration
- ✅ Easier to scale and maintain

---

## Environment Variables

```bash
LIGHTNING_API_KEY      # Lightning AI API key (required)
CUDA_VISIBLE_DEVICES   # GPU selection (optional)
PYTHONUNBUFFERED       # 1 (for logging)
```

---

## Quick Reference

| Task | File |
|------|------|
| Run application | `app.py` |
| Test setup | `test_setup.py` |
| Configure system | `utils/config.py` |
| Edit pipeline logic | `pipeline.py` |
| Modify UI | `app.py` |
| Docker config | `docker-compose.yml` |
| Install packages | `requirements.txt` |
| Execution commands | `COMMANDS.md` |
| Project structure | `PROJECT_STRUCTURE.md` |

---

## Data Flow

```
User Input (Gradio UI)
    ↓
Image Upload
    ↓
QWENImage Edit (Local/GPU)
    ↓
Edited Image
    ↓
Send to Lightning AI API
    ↓
Hunyuan 3D Processing
    ↓
GLB File Download
    ↓
Gradio Model3D Viewer
    ↓
User Download
```

---

## Storage Requirements

```
Total Size:
├── Python environment (~5GB)
├── Model weights (~2GB)
├── Docker image (~2GB)
├── Per execution outputs (~50-200MB)
└── Total: 10GB minimum, 20GB+ recommended
```

---

**Project Version:** 2.0 (Lightning AI)
**Status:** Production Ready
**Last Updated:** December 2024


# 🎨 Image Editing & 3D Asset Generation Pipeline

A complete end-to-end pipeline for editing images using **Qwen Image Edit 2509** and generating 3D assets using **Hunyuan 3D 2.1** via Lightning AI. This project provides an interactive Gradio web interface for seamless workflow execution.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

This pipeline enables users to:

1. **Upload** an image through a web interface
2. **Edit** the image using AI-powered QWENImage Edit 2509 model
3. **Generate** 3D assets from the edited image using Hunyuan 3D 2.1
4. **Preview** the generated 3D model (.glb format) directly in the browser

The entire workflow is orchestrated through a user-friendly Gradio interface and can be deployed using Docker for production environments.

---

## Features

### 🖼️ Image Editing
- AI-powered image editing using **Qwen-Image-Edit-2509** model
- Text-based editing prompts
- Adjustable guidance scale for fine-tuning results
- Support for various image formats (PNG, JPG, JPEG)

### 🎮 3D Asset Generation
- High-quality 3D model generation via **Hunyuan 3D 2.1**
- Lightning AI backend for efficient GPU processing
- Output in web-ready **.glb format**
- Optional text prompts to guide 3D generation

### 🌐 Web Interface
- Interactive Gradio-based UI
- Step-by-step workflow execution
- One-click full pipeline execution
- Real-time 3D model preview with Model3D viewer
- System status monitoring

### 🐳 Docker Support
- Pre-configured Dockerfile and docker-compose
- NVIDIA CUDA support for GPU acceleration
- Easy deployment and scaling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gradio Web Interface                       │
│                         (app.py)                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                        │
│                      (pipeline.py)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│   QWENImage Editor  │         │  Hunyuan 3D 2.1     │
│  (image_editor.py)  │         │   (hunyuan_3d.py)   │
│                     │         │                     │
│ Qwen-Image-Edit-2509│         │   Lightning AI API  │
└─────────────────────┘         └─────────────────────┘
            │                               │
            ▼                               ▼
    ┌──────────────┐               ┌──────────────┐
    │ Edited Image │               │  .glb Model  │
    │    (.png)    │               │   (3D Asset) │
    └──────────────┘               └──────────────┘
```

---

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **CUDA**: 12.1+ (for GPU acceleration)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ (32GB recommended)
- **Docker**: 20.10+ (optional, for containerized deployment)

### Required Accounts
- **Lightning AI Account**: Required for 3D generation API access
  - Get your API key from [Lightning AI](https://lightning.ai/)

---

## Installation

### Option 1: Quick Setup (Recommended)

#### Windows
```bash
# Clone the repository
git clone <repository-url>
cd AssetEdit_Pipeline

# Run setup script
.\setup.bat
```

#### Linux/macOS
```bash
# Clone the repository
git clone <repository-url>
cd AssetEdit_Pipeline

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Verify Installation

```bash
python test_setup.py
```

This will check:
- Python version compatibility
- PyTorch and CUDA availability
- Required package installations
- Docker installation (optional)
- Directory structure

---

## Configuration

### Environment Variables

Set the Lightning AI API key:

#### Windows (PowerShell)
```powershell
$env:LIGHTNING_API_KEY = "your-api-key"
```

#### Linux/macOS
```bash
export LIGHTNING_API_KEY="your-api-key"
```

#### Using .env file
```bash
echo LIGHTNING_API_KEY=your-api-key > .env
```

### Configuration File

The `utils/config.py` provides utility functions for:
- Directory setup
- Docker installation verification
- GPU availability checking
- System configuration

---

## Usage

### Starting the Application

```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Run the Gradio app
python app.py
```

Access the interface at: **http://localhost:7860**

### Using the Web Interface

#### Step-by-Step Workflow

1. **Upload Image**: Click "Upload Image" and select your input image
2. **Edit Image**: 
   - Enter an edit prompt describing desired changes
   - Adjust guidance scale (1.0-20.0) for editing strength
   - Click "Edit Image"
3. **Generate 3D**:
   - Optionally add generation prompts
   - Click "Generate 3D Asset"
4. **Preview**: View the generated .glb model in the 3D viewer

#### Full Pipeline Execution

Use the "Execute Full Pipeline" button to run all steps automatically.

### Alternative: Standalone Image Editor

For image editing only (using diffusers):

```bash
python qwen_app.py
```

This launches a dedicated Qwen Image Edit Plus interface.

### Integrated Pipeline (Recommended)

The integrated pipeline provides a unified interface for both image editing and 3D generation:

```bash
# Launch the integrated web app
python integrated_app.py
```

Access at: **http://localhost:7860**

#### Command Line Usage

```bash
# Full pipeline: Edit + 3D generation
python integrated_pipeline.py -i input.jpg -e "Add sunglasses" -g "high quality model"

# Edit only
python integrated_pipeline.py -i input.jpg -e "Transform to fantasy style" --edit-only

# Generate 3D only
python integrated_pipeline.py -i input.jpg -g "detailed mesh" --generate-only
```

#### Python API Usage

```python
from integrated_pipeline import IntegratedPipeline, quick_edit_image, run_pipeline

# Quick edit an image
edited = quick_edit_image("input.jpg", "Add magical effects")

# Run full pipeline
result = run_pipeline(
    "input.jpg",
    edit_prompt="Make it a cyberpunk character",
    generation_prompt="detailed 3D model"
)
```

---

## Project Structure

```
AssetEdit_Pipeline/
│
├── 📜 Core Application
│   ├── app.py                 # Main Gradio web interface
│   ├── pipeline.py            # Pipeline orchestration logic
│   ├── qwen_app.py            # Standalone Qwen image editor
│   ├── integrated_pipeline.py # ⭐ Unified pipeline (Qwen + Hunyuan 3D)
│   └── integrated_app.py      # ⭐ Unified Gradio web interface
│
├── 🤖 Models
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_editor.py    # QWENImage Edit 2509 wrapper
│   │   └── hunyuan_3d.py      # Hunyuan 3D Lightning AI client
│
├── 🔧 Utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py          # Configuration and helpers
│
├── 🐳 Docker
│   ├── Dockerfile             # Docker image definition
│   ├── docker-compose.yml     # Container orchestration
│
├── 📦 Dependencies
│   └── requirements.txt       # Python packages
│
├── 🚀 Setup Scripts
│   ├── setup.bat              # Windows setup
│   └── setup.sh               # Linux/macOS setup
│
├── 📋 Documentation
│   ├── README.md              # This file
│   ├── COMMANDS.md            # Quick command reference
│   └── PROJECT_STRUCTURE.md   # Detailed structure docs
│
└── 🧪 Testing
    └── test_setup.py          # Installation verification
```

---

## API Reference

### ImageTo3DPipeline

Main pipeline class for orchestrating the workflow.

```python
from pipeline import ImageTo3DPipeline

pipeline = ImageTo3DPipeline(output_dir="./outputs")

# Edit image
edited_path = pipeline.process_image_to_edited_image(
    image_path="input.png",
    edit_prompt="Make the sky sunset orange",
    guidance_scale=7.5
)

# Generate 3D
result = pipeline.generate_3d_from_image(
    image_path=edited_path,
    generation_prompt="3D model with realistic textures"
)

# Full pipeline
result = pipeline.run_full_pipeline(
    input_image_path="input.png",
    edit_prompt="Edit description",
    generation_prompt="3D generation prompt"
)
```

### QWENImageEditor

Image editing using Qwen2-VL model.

```python
from models.image_editor import QWENImageEditor

editor = QWENImageEditor()
edited_image = editor.edit_image(
    image_path="input.png",
    prompt="Add sunglasses to the person",
    guidance_scale=7.5,
    num_inference_steps=50
)
edited_image.save("output.png")
```

### Hunyuan3DGenerator

3D asset generation via Lightning AI.

```python
from models.hunyuan_3d import Hunyuan3DGenerator

generator = Hunyuan3DGenerator(output_dir="./outputs")
result = generator.generate_3d_asset(
    image_path="input.png",
    prompt="High quality 3D model",
    output_name="my_asset"
)
# result["model_path"] contains the .glb file path
```

### IntegratedPipeline (Recommended)

Unified pipeline combining Qwen Image Edit Plus and Hunyuan 3D.

```python
from integrated_pipeline import IntegratedPipeline

# Initialize pipeline
pipeline = IntegratedPipeline(
    output_dir="./outputs",
    auto_load_models=False  # Lazy load to save memory
)

# Edit image only
edited_image, saved_path = pipeline.edit_image(
    image="input.png",
    prompt="Add sunglasses and a hat",
    true_cfg_scale=4.0,
    num_inference_steps=40
)

# Generate 3D only
result = pipeline.generate_3d(
    image="input.png",
    prompt="detailed 3D character model"
)

# Run full pipeline (Edit → 3D)
result = pipeline.run_full_pipeline(
    input_image="input.png",
    edit_prompt="Transform into a fantasy warrior",
    generation_prompt="high quality 3D game asset"
)

print(result["steps"]["image_editing"]["edited_image_path"])
print(result["steps"]["3d_generation"]["model_path"])
```

### Convenience Functions

Quick one-liner functions for common tasks:

```python
from integrated_pipeline import quick_edit_image, quick_generate_3d, run_pipeline

# Quick edit
edited = quick_edit_image("input.jpg", "Add wings", output_path="edited.png")

# Quick 3D generation
result = quick_generate_3d("input.jpg", prompt="clean mesh")

# Full pipeline in one line
result = run_pipeline("input.jpg", "Make it magical", "detailed model")
```

---

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t image-to-3d:latest .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Docker Compose Configuration

The `docker-compose.yml` configures:
- Port mapping: 7860:7860
- Volume mounts for outputs, assets, and logs
- Environment variable for Lightning API key
- Resource limits (4 CPUs, 8GB RAM)

### Environment Variables for Docker

```bash
# Set before running docker-compose
export LIGHTNING_API_KEY=your-api-key
docker-compose up -d
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Computer vision utilities |
| transformers | ≥4.36.0 | Hugging Face models |
| accelerate | ≥0.24.0 | Training acceleration |
| gradio | ≥4.0.0 | Web interface |
| diffusers | ≥0.24.0 | Diffusion models |
| pillow | ≥10.0.0 | Image processing |
| peft | ≥0.7.0 | Parameter-efficient fine-tuning |
| requests | ≥2.31.0 | HTTP client |
| opencv-python | ≥4.8.0 | Image processing |

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use a GPU with more VRAM. Try setting:
```python
torch.cuda.empty_cache()
```

#### Lightning API Key Missing
```
RuntimeError: LIGHTNING_API_KEY environment variable required
```
**Solution**: Set the API key as described in [Configuration](#configuration).

#### Model Download Issues
```
HTTPError: 403 Client Error
```
**Solution**: Ensure you have accepted the model license on Hugging Face and are logged in:
```bash
huggingface-cli login
```

#### Docker GPU Access
```
docker: Error response from daemon: could not select device driver
```
**Solution**: Install NVIDIA Container Toolkit:
```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Getting Help

1. Check the logs: `docker-compose logs` or `pipeline.log`
2. Run diagnostics: `python test_setup.py`
3. Verify GPU: `nvidia-smi`

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) - For the Qwen2-VL image editing model
- [Hunyuan 3D](https://github.com/Tencent/Hunyuan3D-2) - For the 3D generation model
- [Lightning AI](https://lightning.ai/) - For the 3D generation API infrastructure
- [Gradio](https://gradio.app/) - For the web interface framework
- [Hugging Face](https://huggingface.co/) - For model hosting and transformers library

---

## Contact

For questions and support, please open an issue on the repository.

---

**Happy Creating! 🎨🎮**

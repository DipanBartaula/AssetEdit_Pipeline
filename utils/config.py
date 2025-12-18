"""
Utility functions for the pipeline
"""

import os
import logging
from pathlib import Path
from typing import Optional
import subprocess

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    directories = [
        "./outputs",
        "./assets",
        "./logs",
        "./models",
        "./utils"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")


def check_docker_installation() -> bool:
    """Check if Docker is installed"""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        logger.info("Docker is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not installed or not in PATH")
        return False


def check_docker_image(image_name: str) -> bool:
    """Check if Docker image exists locally"""
    try:
        result = subprocess.run(
            ["docker", "inspect", image_name],
            capture_output=True,
            check=False
        )
        exists = result.returncode == 0
        status = "found" if exists else "not found"
        logger.info(f"Docker image {image_name}: {status}")
        return exists
    except Exception as e:
        logger.error(f"Error checking Docker image: {e}")
        return False


def pull_docker_image(image_name: str) -> bool:
    """Pull Docker image from registry"""
    try:
        logger.info(f"Pulling Docker image: {image_name}")
        subprocess.run(
            ["docker", "pull", image_name],
            check=True
        )
        logger.info(f"Successfully pulled: {image_name}")
        return True
    except Exception as e:
        logger.error(f"Error pulling Docker image: {e}")
        return False


def check_gpu_availability() -> bool:
    """Check if GPU is available"""
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        logger.info(f"GPU Available: {available}, Count: {count}")
        return available
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False


def get_gpu_info() -> dict:
    """Get GPU information"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "count": 0}
        
        return {
            "available": True,
            "count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda
        }
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return {"available": False, "error": str(e)}


def setup_environment():
    """Setup environment for pipeline execution"""
    logger.info("Setting up environment...")
    
    # Create directories
    setup_directories()
    
    # Check Docker
    docker_available = check_docker_installation()
    if not docker_available:
        logger.warning("Docker not found - you'll need Docker to use Hunyuan 3D")
    
    # Check GPU
    gpu_info = get_gpu_info()
    logger.info(f"GPU Info: {gpu_info}")
    
    return {
        "docker_available": docker_available,
        "gpu_info": gpu_info
    }


class Config:
    """Configuration class for pipeline"""
    
    # Model configuration
    QWEN_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    HUNYUAN_DOCKER_IMAGE = "feynmann/hunyuan_custom:v1"
    HUNYUAN_DOCKER_CONTAINER = "hunyuan_3d_generator"
    
    # Directory configuration
    OUTPUT_DIR = "./outputs"
    ASSETS_DIR = "./assets"
    LOGS_DIR = "./logs"
    
    # Gradio configuration
    GRADIO_HOST = "0.0.0.0"
    GRADIO_PORT = 7860
    GRADIO_SHARE = False
    
    # Generation configuration
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_NUM_INFERENCE_STEPS = 50
    
    # Device configuration
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """Get configuration as dictionary"""
        return {
            "qwen_model": cls.QWEN_MODEL_NAME,
            "hunyuan_image": cls.HUNYUAN_DOCKER_IMAGE,
            "output_dir": cls.OUTPUT_DIR,
            "gradio_port": cls.GRADIO_PORT,
            "device": cls.DEVICE,
            "guidance_scale": cls.DEFAULT_GUIDANCE_SCALE,
            "inference_steps": cls.DEFAULT_NUM_INFERENCE_STEPS
        }

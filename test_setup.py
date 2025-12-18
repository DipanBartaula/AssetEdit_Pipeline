"""
Test and Verify Pipeline Components
Run this to verify all components are properly installed and configured
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_python_version():
    """Check Python version"""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    logger.info(f"✓ Python version OK: {version.major}.{version.minor}.{version.micro}")
    return True


def test_torch():
    """Test PyTorch installation"""
    logger.info("Testing PyTorch...")
    try:
        import torch
        logger.info(f"✓ PyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        logger.info(f"✓ CUDA Available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            logger.info(f"✓ GPU Count: {device_count}")
            logger.info(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"✓ CUDA Version: {torch.version.cuda}")
        
        return True
    except Exception as e:
        logger.error(f"✗ PyTorch error: {e}")
        return False


def test_transformers():
    """Test Transformers library"""
    logger.info("Testing Transformers...")
    try:
        import transformers
        logger.info(f"✓ Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        logger.error(f"✗ Transformers error: {e}")
        return False


def test_gradio():
    """Test Gradio installation"""
    logger.info("Testing Gradio...")
    try:
        import gradio
        logger.info(f"✓ Gradio version: {gradio.__version__}")
        return True
    except Exception as e:
        logger.error(f"✗ Gradio error: {e}")
        return False


def test_pillow():
    """Test Pillow installation"""
    logger.info("Testing Pillow...")
    try:
        from PIL import Image
        logger.info(f"✓ Pillow OK")
        return True
    except Exception as e:
        logger.error(f"✗ Pillow error: {e}")
        return False


def test_docker():
    """Test Docker installation"""
    logger.info("Testing Docker...")
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"✓ {result.stdout.strip()}")
            return True
        else:
            logger.warning("Docker not in PATH")
            return False
    except Exception as e:
        logger.warning(f"Docker not available: {e}")
        return False


def test_docker_image():
    """Test Hunyuan Docker image"""
    logger.info("Testing Hunyuan 3D Docker image...")
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "inspect", "feynmann/hunyuan_custom:v1"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("✓ Hunyuan 3D image found locally")
            return True
        else:
            logger.warning("✗ Hunyuan 3D image not found")
            logger.info("  Run: docker pull feynmann/hunyuan_custom:v1")
            return False
    except Exception as e:
        logger.warning(f"Cannot check Docker image: {e}")
        return False


def test_directories():
    """Test directory structure"""
    logger.info("Testing directory structure...")
    required_dirs = [
        "./models",
        "./utils",
        "./outputs",
        "./assets"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            logger.info(f"✓ {dir_path}")
        else:
            logger.warning(f"✗ {dir_path} not found, creating...")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created {dir_path}")
    
    return all_exist


def test_files():
    """Test required files"""
    logger.info("Testing required files...")
    required_files = [
        "pipeline.py",
        "app.py",
        "models/image_editor.py",
        "models/hunyuan_3d.py",
        "utils/config.py",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✓ {file_path}")
        else:
            logger.error(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_imports():
    """Test critical imports"""
    logger.info("Testing critical imports...")
    try:
        from models.image_editor import QWENImageEditor
        logger.info("✓ QWENImageEditor")
        
        from models.hunyuan_3d import Hunyuan3DGenerator
        logger.info("✓ Hunyuan3DGenerator")
        
        from pipeline import ImageTo3DPipeline
        logger.info("✓ ImageTo3DPipeline")
        
        from utils.config import Config
        logger.info("✓ Config")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("Pipeline Verification Test Suite")
    logger.info("=" * 50)
    logger.info("")
    
    tests = [
        ("Python Version", test_python_version),
        ("PyTorch", test_torch),
        ("Transformers", test_transformers),
        ("Gradio", test_gradio),
        ("Pillow", test_pillow),
        ("Docker", test_docker),
        ("Docker Image", test_docker_image),
        ("Directories", test_directories),
        ("Required Files", test_files),
        ("Module Imports", test_imports),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
        logger.info("")
    
    # Summary
    logger.info("=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("")
    logger.info(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("")
        logger.info("✓ All tests passed! Ready to run the pipeline.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run: python app.py")
        logger.info("2. Open: http://localhost:7860")
        logger.info("3. Upload an image and start processing!")
        return 0
    else:
        logger.info("")
        logger.error("✗ Some tests failed. Please fix the issues above.")
        logger.info("")
        logger.info("Troubleshooting steps:")
        logger.info("1. Check Python version (3.10+)")
        logger.info("2. Install requirements: pip install -r requirements.txt")
        logger.info("3. For Docker issues, install Docker Desktop")
        logger.info("4. For GPU issues, check NVIDIA driver and CUDA")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

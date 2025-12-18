"""
Hunyuan 3D 2.1 Model for 3D Asset Generation via Lightning AI
"""

import torch
import logging
import subprocess
import json
import os
from pathlib import Path
from typing import Optional, List, Dict
import requests
import time

logger = logging.getLogger(__name__)


class Hunyuan3DGenerator:
    """
    3D asset generation using Hunyuan 3D 2.1 model via Lightning AI
    Outputs 3D models in .glb format for web rendering
    """

    def __init__(
        self,
        lightning_api_key: str = None,
        output_dir: str = "./outputs"
    ):
        """
        Initialize Hunyuan 3D Generator for Lightning AI
        
        Args:
            lightning_api_key: Lightning AI API key (from environment if not provided)
            output_dir: Directory for output 3D assets
        """
        self.api_key = lightning_api_key or os.environ.get("LIGHTNING_API_KEY", "")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lightning_api_url = "https://api.lightning.ai/v1/hunyuan3d"
        
        logger.info("Initialized Hunyuan 3D Generator for Lightning AI")
        logger.info(f"Output directory: {self.output_dir}")

    def start_docker_container(self) -> bool:
        """
        Deprecated - Lightning AI handles server infrastructure
        
        Returns:
            Always True as Lightning AI is managed service
        """
        logger.info("Lightning AI server is managed automatically")
        return True

    def stop_docker_container(self) -> bool:
        """
        Deprecated - Lightning AI handles server infrastructure
        
        Returns:
            Always True
        """
        logger.info("Lightning AI cleanup (handled automatically)")
        return True

    def generate_3d_asset(
        self,
        image_path: str,
        prompt: str = "",
        output_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Generate 3D asset from image via Lightning AI
        Returns .glb format for web rendering
        
        Args:
            image_path: Path to input image
            prompt: Additional text prompt for generation
            output_name: Name for output files
            
        Returns:
            Dictionary with generated asset paths or None on failure
        """
        try:
            if not self.api_key:
                logger.error("Lightning API key not configured")
                raise RuntimeError("LIGHTNING_API_KEY environment variable required")
            
            logger.info(f"Generating 3D asset from image: {image_path}")
            
            if output_name is None:
                output_name = f"asset_{int(time.time())}"
            
            # Read image as bytes
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Prepare request to Lightning AI
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/octet-stream"
            }
            
            params = {
                "prompt": prompt,
                "output_format": "glb"
            }
            
            logger.info("Sending request to Lightning AI for 3D generation...")
            
            response = requests.post(
                f"{self.lightning_api_url}/generate",
                headers=headers,
                params=params,
                data=image_data,
                timeout=600
            )
            response.raise_for_status()
            
            # Save GLB file
            glb_path = self.output_dir / f"{output_name}.glb"
            with open(glb_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"3D asset generated successfully: {output_name}")
            
            return {
                "status": "success",
                "asset_name": output_name,
                "model_path": str(glb_path),
                "format": "glb",
                "metadata": {
                    "prompt": prompt,
                    "source_image": image_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating 3D asset: {e}")
            return None

    def get_asset_preview(self, asset_name: str) -> Optional[dict]:
        """
        Get preview of generated asset (.glb format)
        
        Args:
            asset_name: Name of the asset
            
        Returns:
            Dictionary with preview paths
        """
        try:
            model_path = self.output_dir / f"{asset_name}.glb"
            
            result = {
                "asset_name": asset_name,
                "model_exists": model_path.exists(),
                "format": "glb"
            }
            
            if result["model_exists"]:
                result["model_path"] = str(model_path)
                result["model_size_mb"] = model_path.stat().st_size / (1024*1024)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting asset preview: {e}")
            return None

    def list_generated_assets(self) -> List[str]:
        """List all generated assets"""
        try:
            assets = []
            for file in self.output_dir.glob("*.ply"):
                assets.append(file.stem)
            return sorted(list(set(assets)))
        except Exception as e:
            logger.error(f"Error listing assets: {e}")
            return []

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_type": "Hunyuan 3D 2.1",
            "platform": "Lightning AI",
            "output_format": "glb",
            "output_dir": str(self.output_dir)
        }

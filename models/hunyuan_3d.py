"""Hunyuan 3D 2.1 Model for 3D Asset Generation via Docker.

This module provides a thin wrapper around a dockerized Hunyuan3D-2.1
pipeline. It is used by the higher-level pipelines (e.g. IntegratedPipeline)
to generate .glb assets from input images.
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Optional, List, Dict, Union
import time

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class Hunyuan3DGenerator:
    """3D asset generation using Hunyuan 3D 2.1 via Docker.

    The generator prepares an input image, runs a docker container with
    Hunyuan3D-2.1, and collects the resulting `.glb` mesh from a mounted
    volume. This replaces the previous Lightning AI HTTP API workflow.
    """

    def __init__(
        self,
        lightning_api_key: str = None,
        output_dir: str = "./outputs",
        docker_image: str = "your_username/hunyuan3d:latest",
    ):
        """Initialize Hunyuan 3D Generator.

        Args:
            lightning_api_key: Kept for backwards compatibility (unused).
            output_dir: Directory for output 3D assets and docker data mount.
            docker_image: Name of the Hunyuan3D Docker image to run.
        """
        # NOTE: ``lightning_api_key`` is intentionally ignored now that we rely
        # on a local Docker pipeline instead of Lightning AI's HTTP API.
        del lightning_api_key

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Directory mounted into the Docker container as /data
        self.data_dir = self.output_dir
        self.docker_image = docker_image

        logger.info("Initialized Hunyuan 3D Generator (Docker backend)")
        logger.info(f"Output/data directory: {self.data_dir}")

    def start_docker_container(self) -> bool:
        """Compatibility stub.

        Historically this method managed a long-lived backend. With the
        dockerized pipeline we simply run containers per request, so this is
        effectively a no-op kept for interface compatibility.
        """

        logger.info("Hunyuan3D Docker backend is started on demand per request")
        return True

    def stop_docker_container(self) -> bool:
        """Compatibility stub for cleanup.

        The docker container is run with ``--rm`` and exits after each
        generation, so there is nothing to stop here.
        """

        logger.info("Hunyuan3D Docker backend requires no explicit cleanup")
        return True

    def _save_image_to_data_dir(
        self,
        image: Union[str, Image.Image, "np.ndarray"],
        input_name: str = "input.png",
    ) -> Path:
        """Normalize different image types into an on-disk PNG in ``data_dir``.

        Args:
            image: Image path, PIL.Image, or numpy array.
            input_name: Target filename inside ``data_dir``.

        Returns:
            Path to the written image file.
        """

        target_path = self.data_dir / input_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(image, str):
            # Treat as a file path and copy
            src = Path(image)
            if not src.exists():
                raise FileNotFoundError(f"Input image not found: {src}")
            import shutil
            shutil.copy2(src, target_path)
            return target_path

        if isinstance(image, Image.Image):
            image.save(target_path)
            return target_path

        # Assume numpy array-like
        arr = np.array(image)
        img = Image.fromarray(arr)
        img.save(target_path)
        return target_path

    def generate_3d_asset(
        self,
        image: Union[str, Image.Image, "np.ndarray"],
        prompt: str = "",
        output_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """Generate 3D asset from an image via the Dockerized Hunyuan3D pipeline.

        Args:
            image: Input image (path, PIL image, or numpy array).
            prompt: Additional text prompt for generation (currently unused by
                the docker pipeline but kept for API compatibility).
            output_name: Base name for output files (without extension).

        Returns:
            Dictionary with generated asset metadata or ``None`` on failure.
        """

        try:
            if output_name is None:
                output_name = f"asset_{int(time.time())}"

            logger.info("Generating 3D asset via Docker backend")

            # 1) Prepare input image in the mounted data directory
            input_path = self._save_image_to_data_dir(image, input_name="input.png")
            logger.info(f"Input image prepared at: {input_path}")

            # 2) Expected output from the container
            container_output = self.data_dir / "output_shape.glb"
            final_output = self.data_dir / f"{output_name}.glb"

            # Remove any stale outputs
            if container_output.exists():
                container_output.unlink()
            if final_output.exists():
                final_output.unlink()

            # 3) Python script executed *inside* the container
            python_script = """
import sys
sys.path.insert(0, './hy3dshape')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

print('Loading Hunyuan3D pipeline...')
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')

print('Generating mesh from /data/input.png...')
mesh = pipeline(image='/data/input.png')[0]

print('Saving to /data/output_shape.glb...')
mesh.export('/data/output_shape.glb')

print('✓ Shape generation complete!')
"""

            docker_cmd = [
                "docker",
                "run",
                "--gpus",
                "all",
                "--rm",
                "-v",
                f"{self.data_dir.absolute()}:/data",
                "-w",
                "/workspace/Hunyuan3D-2.1",
                self.docker_image,
                "python3",
                "-c",
                python_script,
            ]

            logger.info("Running Hunyuan3D Docker container...")
            subprocess.run(docker_cmd, check=True)

            if not container_output.exists():
                raise RuntimeError("Docker pipeline did not produce output_shape.glb")

            # Rename to the requested output name
            container_output.rename(final_output)
            logger.info(f"3D asset generated successfully: {final_output}")

            return {
                "status": "success",
                "asset_name": output_name,
                "model_path": str(final_output),
                "format": "glb",
                "metadata": {
                    "prompt": prompt,
                    "source_image": str(input_path),
                },
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
            for file in self.output_dir.glob("*.glb"):
                assets.append(file.stem)
            return sorted(list(set(assets)))
        except Exception as e:
            logger.error(f"Error listing assets: {e}")
            return []

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_type": "Hunyuan 3D 2.1",
            "platform": "Docker",
            "output_format": "glb",
            "output_dir": str(self.output_dir),
        }

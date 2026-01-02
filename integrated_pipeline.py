"""
Integrated Pipeline: Qwen Image Edit + Hunyuan 3D Asset Generation
===================================================================
This module combines:
1. Qwen Image Edit 2509 - For AI-powered image editing
2. Hunyuan 3D 2.1 - For 3D asset generation from images

Author: Integrated from AssetEdit_Pipeline
"""

import os
import torch
import logging
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image
import numpy as np
from hunyuan3d_runner import Hunyuan3DRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# QWEN IMAGE EDITOR MODULE
# =============================================================================

class QwenImageEditor:
    """
    Image editing using Qwen Image Edit Plus (Qwen-Image-Edit-2509)
    Uses diffusers pipeline for high-quality image editing
    """
    
    def __init__(self, device: str = None):
        """
        Initialize Qwen Image Editor
        
        Args:
            device: Device to run on (cuda/cpu). Auto-detects if not provided.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.is_loaded = False
        
        logger.info(f"QwenImageEditor initialized (device: {self.device})")
    
    def load_model(self):
        """Load the Qwen Image Edit Plus pipeline"""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            from diffusers import QwenImageEditPlusPipeline
            
            logger.info("Loading Qwen Image Edit Plus pipeline...")
            
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                torch_dtype=torch.bfloat16
            )
            self.pipeline.to(self.device)
            self.pipeline.set_progress_bar_config(disable=None)
            
            self.is_loaded = True
            logger.info("Qwen Image Edit Plus loaded successfully!")
            
        except ImportError:
            logger.error("diffusers package not found. Install with: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen Image Edit Plus: {e}")
            raise
    
    def edit_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str,
        negative_prompt: str = "",
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 40,
        seed: int = 0,
        num_outputs: int = 1
    ) -> List[Image.Image]:
        """
        Edit an image based on text prompt
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            prompt: Text description of desired edits
            negative_prompt: Things to avoid in the output
            true_cfg_scale: True CFG scale value (default: 4.0)
            guidance_scale: Guidance scale for generation (default: 1.0)
            num_inference_steps: Number of inference steps (default: 40)
            seed: Random seed for reproducibility (default: 0)
            num_outputs: Number of images to generate (default: 1)
            
        Returns:
            List of edited PIL Images
        """
        if not self.is_loaded:
            self.load_model()
        
        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")
        
        try:
            generator = torch.manual_seed(seed)
            
            inputs = {
                "image": [image],
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt if negative_prompt else " ",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_outputs,
            }
            
            logger.info(f"Editing image with prompt: {prompt}")
            
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_images = output.images
            
            logger.info(f"Generated {len(output_images)} edited image(s)")
            return output_images
            
        except Exception as e:
            logger.error(f"Error during image editing: {e}")
            raise
    
    def edit_multiple_images(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        prompt: str,
        **kwargs
    ) -> List[Image.Image]:
        """
        Edit using multiple input images (for context/reference)
        
        Args:
            images: List of input images
            prompt: Edit prompt
            **kwargs: Additional arguments passed to edit_image
            
        Returns:
            List of edited PIL Images
        """
        if not self.is_loaded:
            self.load_model()
        
        # Convert all images to PIL
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))
        
        try:
            generator = torch.manual_seed(kwargs.get('seed', 0))
            
            inputs = {
                "image": pil_images,
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": kwargs.get('true_cfg_scale', 4.0),
                "negative_prompt": kwargs.get('negative_prompt', " "),
                "num_inference_steps": kwargs.get('num_inference_steps', 40),
                "guidance_scale": kwargs.get('guidance_scale', 1.0),
                "num_images_per_prompt": kwargs.get('num_outputs', 1),
            }
            
            logger.info(f"Editing {len(pil_images)} images with prompt: {prompt}")
            
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                return output.images
                
        except Exception as e:
            logger.error(f"Error during multi-image editing: {e}")
            raise


# =============================================================================
# HUNYUAN 3D GENERATOR MODULE
# =============================================================================

class Hunyuan3DGenerator:
    """
    3D asset generation using Hunyuan 3D 2.1 model via Docker-based Hunyuan3DRunner
    Outputs 3D models in .glb format for web rendering
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs",
        docker_image: Optional[str] = None,
    ):
        """
        Initialize Hunyuan 3D Generator
        
        Args:
            output_dir: Directory for output 3D assets
            docker_image: Docker image name for Hunyuan3D-2.1
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docker_image = docker_image or os.environ.get(
            "HUNYUAN3D_DOCKER_IMAGE",
            "belbaseankit17/hunyuan_custom_fixed:v4",
        )
        logger.info(f"Hunyuan3DGenerator initialized (output: {self.output_dir}, docker_image: {self.docker_image})")
    
    def _save_input_image(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """Ensure the input is saved as a PNG file and return its path"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.output_dir / "hunyuan_input.png"
        
        if isinstance(image, str):
            return image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type for Hunyuan3DGenerator")
        
        img.save(input_path)
        return str(input_path)
    
    def generate_3d_asset(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str = "",
        output_name: Optional[str] = None,
        output_format: str = "glb"
    ) -> Optional[Dict]:
        """
        Generate 3D asset from image using the Docker-based Hunyuan3DRunner
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            prompt: Additional text prompt for generation (unused, kept for API compatibility)
            output_name: Name for output files
            output_format: Output format (only 'glb' is currently supported)
            
        Returns:
            Dictionary with generated asset info or None on failure
        """
        if output_format != "glb":
            logger.warning("Hunyuan3DRunner currently only outputs GLB. Forcing output_format='glb'.")
            output_format = "glb"
        
        if output_name is None:
            output_name = f"asset_{int(time.time())}"
        
        try:
            input_image_path = self._save_input_image(image)
            runner = Hunyuan3DRunner(
                docker_image=self.docker_image,
                data_dir=str(self.output_dir),
            )
            logger.info(f"Starting Docker-based Hunyuan3D generation using image: {input_image_path}")
            output_file = runner.run(input_image_path, verbose=True)
            
            # The runner writes output_shape.glb into data_dir; rename to match output_name
            output_path = self.output_dir / f"{output_name}.{output_format}"
            try:
                generated_path = self.output_dir / "output_shape.glb"
                if generated_path.exists():
                    generated_path.rename(output_path)
                else:
                    # Fallback to whatever path runner reported
                    if output_file and Path(output_file).exists():
                        Path(output_file).rename(output_path)
            except Exception as e:
                logger.warning(f"Could not rename generated GLB file: {e}")
                output_path = Path(output_file) if output_file else None
            
            if not output_path or not Path(output_path).exists():
                raise RuntimeError("Hunyuan3DRunner did not produce an output GLB file")
            
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"3D asset generated via Docker: {output_path}")
            
            return {
                "status": "success",
                "asset_name": output_name,
                "model_path": str(output_path),
                "format": output_format,
                "file_size_mb": file_size_mb,
                "metadata": {
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Error generating 3D asset via Docker Hunyuan3DRunner: {e}")
            return None
    
    def get_asset_preview(self, asset_name: str) -> Optional[Dict]:
        """
        Get preview info for a generated asset
        
        Args:
            asset_name: Name of the asset
            
        Returns:
            Dictionary with asset info
        """
        model_path = self.output_dir / f"{asset_name}.glb"
        
        result = {
            "asset_name": asset_name,
            "model_exists": model_path.exists(),
            "format": "glb",
        }
        
        if result["model_exists"]:
            result["model_path"] = str(model_path)
            result["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)
        
        return result
    
    def list_assets(self) -> List[str]:
        """List all generated assets"""
        assets = set()
        for ext in ["glb", "ply", "obj"]:
            for file in self.output_dir.glob(f"*.{ext}"):
                assets.add(file.stem)
        return sorted(list(assets))


# =============================================================================
# INTEGRATED PIPELINE
# =============================================================================

class IntegratedPipeline:
    """
    Complete pipeline combining Qwen Image Edit and Hunyuan 3D
    
    Workflow:
    1. Load/Upload input image
    2. Edit image using Qwen Image Edit Plus
    3. Generate 3D asset from edited image using Hunyuan 3D
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs",
        auto_load_models: bool = False,
        hunyuan_docker_image: Optional[str] = None,
    ):
        """
        Initialize the integrated pipeline
        
        Args:
            output_dir: Output directory for all results
            auto_load_models: Whether to load models on init
            hunyuan_docker_image: Optional Docker image override for Hunyuan3D.
                If not provided, falls back to HUNYUAN3D_DOCKER_IMAGE env var or
                the default configured in Hunyuan3DGenerator.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.image_editor = QwenImageEditor()

        # Allow explicit override of the Hunyuan3D Docker image; otherwise rely
        # on the generator's own default/env logic.
        self.generator_3d = Hunyuan3DGenerator(
            output_dir=str(self.output_dir / "3d_assets"),
            docker_image=hunyuan_docker_image,
        )
        
        # Execution history
        self.execution_history = []
        
        if auto_load_models:
            self.image_editor.load_model()
        
        logger.info("IntegratedPipeline initialized")
    
    def edit_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str,
        save_output: bool = True,
        **kwargs
    ) -> Tuple[Image.Image, Optional[str]]:
        """
        Edit an image using Qwen Image Edit Plus
        
        Args:
            image: Input image
            prompt: Edit prompt
            save_output: Whether to save the edited image
            **kwargs: Additional arguments for editing
            
        Returns:
            Tuple of (edited image, saved path or None)
        """
        logger.info(f"[Edit] Starting image editing with prompt: {prompt}")
        
        edited_images = self.image_editor.edit_image(image, prompt, **kwargs)
        edited_image = edited_images[0]  # Take first result
        
        saved_path = None
        if save_output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_path = str(self.output_dir / f"edited_{timestamp}.png")
            edited_image.save(saved_path)
            logger.info(f"[Edit] Saved edited image: {saved_path}")
        
        return edited_image, saved_path
    
    def generate_3d(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str = "",
        output_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Generate 3D asset from image using Hunyuan 3D
        
        Args:
            image: Input image
            prompt: Generation prompt
            output_name: Output file name
            
        Returns:
            Dictionary with 3D asset info
        """
        logger.info(f"[3D] Starting 3D generation...")
        
        result = self.generator_3d.generate_3d_asset(
            image=image,
            prompt=prompt,
            output_name=output_name
        )
        
        return result
    
    def run_full_pipeline(
        self,
        input_image: Union[str, Image.Image, np.ndarray],
        edit_prompt: str,
        generation_prompt: str = "",
        edit_kwargs: Dict = None
    ) -> Dict:
        """
        Run complete pipeline: Edit Image -> Generate 3D
        
        Args:
            input_image: Input image
            edit_prompt: Prompt for image editing
            generation_prompt: Prompt for 3D generation
            edit_kwargs: Additional kwargs for image editing
            
        Returns:
            Dictionary with full pipeline results
        """
        execution_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        logger.info(f"[Pipeline] Starting execution {execution_id}")
        
        result = {
            "execution_id": execution_id,
            "edit_prompt": edit_prompt,
            "generation_prompt": generation_prompt,
            "steps": {},
            "status": "running"
        }
        
        edit_kwargs = edit_kwargs or {}
        
        try:
            # Step 1: Edit Image
            logger.info("[Pipeline] Step 1: Editing image...")
            edited_image, edited_path = self.edit_image(
                image=input_image,
                prompt=edit_prompt,
                save_output=True,
                **edit_kwargs
            )
            
            result["steps"]["image_editing"] = {
                "status": "success",
                "edited_image_path": edited_path
            }
            logger.info("[Pipeline] Step 1 ✓ Complete")
            
            # Step 2: Generate 3D
            logger.info("[Pipeline] Step 2: Generating 3D asset...")
            output_name = f"asset_{execution_id}"
            
            asset_result = self.generate_3d(
                image=edited_image,
                prompt=generation_prompt,
                output_name=output_name
            )
            
            if asset_result:
                result["steps"]["3d_generation"] = asset_result
                logger.info("[Pipeline] Step 2 ✓ Complete")
            else:
                result["steps"]["3d_generation"] = {"status": "failed"}
                raise RuntimeError("3D generation failed")
            
            result["status"] = "success"
            
            # Save execution results
            results_file = self.output_dir / f"pipeline_result_{execution_id}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            self.execution_history.append(result)
            logger.info(f"[Pipeline] Execution complete. Results: {results_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"[Pipeline] Execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result
    
    def get_status(self) -> Dict:
        """Get pipeline status"""
        return {
            "image_editor_loaded": self.image_editor.is_loaded,
            "output_directory": str(self.output_dir),
            "execution_count": len(self.execution_history),
            "generated_assets": self.generator_3d.list_assets(),
            "device": self.image_editor.device
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_edit_image(
    image_path: str,
    prompt: str,
    output_path: str = None,
    **kwargs
) -> Image.Image:
    """
    Quick function to edit an image
    
    Args:
        image_path: Path to input image
        prompt: Edit prompt
        output_path: Optional output path
        **kwargs: Additional editing arguments
        
    Returns:
        Edited PIL Image
    """
    editor = QwenImageEditor()
    results = editor.edit_image(image_path, prompt, **kwargs)
    edited = results[0]
    
    if output_path:
        edited.save(output_path)
        logger.info(f"Saved to: {output_path}")
    
    return edited


def quick_generate_3d(
    image_path: str,
    prompt: str = "",
    output_dir: str = "./outputs"
) -> Optional[Dict]:
    """
    Quick function to generate 3D asset from image
    
    Args:
        image_path: Path to input image
        prompt: Generation prompt
        output_dir: Output directory
        
    Returns:
        Dictionary with asset info
    """
    generator = Hunyuan3DGenerator(output_dir=output_dir)
    return generator.generate_3d_asset(image_path, prompt)


def run_pipeline(
    image_path: str,
    edit_prompt: str,
    generation_prompt: str = "",
    output_dir: str = "./outputs"
) -> Dict:
    """
    Quick function to run the full pipeline
    
    Args:
        image_path: Path to input image
        edit_prompt: Prompt for editing
        generation_prompt: Prompt for 3D generation
        output_dir: Output directory
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = IntegratedPipeline(output_dir=output_dir)
    return pipeline.run_full_pipeline(
        input_image=image_path,
        edit_prompt=edit_prompt,
        generation_prompt=generation_prompt
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated Image Edit + 3D Generation Pipeline"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "--edit-prompt", "-e",
        type=str,
        required=True,
        help="Prompt for image editing"
    )
    parser.add_argument(
        "--generation-prompt", "-g",
        type=str,
        default="",
        help="Prompt for 3D generation (optional)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--edit-only",
        action="store_true",
        help="Only edit image, skip 3D generation"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate 3D, skip editing"
    )
    
    args = parser.parse_args()
    
    if args.edit_only:
        # Just edit the image
        print(f"Editing image: {args.image}")
        edited = quick_edit_image(
            args.image,
            args.edit_prompt,
            output_path=f"{args.output_dir}/edited_output.png"
        )
        print(f"Done! Output saved to {args.output_dir}/edited_output.png")
        
    elif args.generate_only:
        # Just generate 3D
        print(f"Generating 3D from: {args.image}")
        result = quick_generate_3d(
            args.image,
            args.generation_prompt,
            args.output_dir
        )
        if result:
            print(f"Done! Model saved to: {result['model_path']}")
        else:
            print("3D generation failed")
            
    else:
        # Run full pipeline
        print("Running full pipeline...")
        result = run_pipeline(
            args.image,
            args.edit_prompt,
            args.generation_prompt,
            args.output_dir
        )
        
        if result["status"] == "success":
            print("\n✓ Pipeline completed successfully!")
            print(f"  Edited image: {result['steps']['image_editing']['edited_image_path']}")
            print(f"  3D model: {result['steps']['3d_generation']['model_path']}")
        else:
            print(f"\n✗ Pipeline failed: {result.get('error', 'Unknown error')}")

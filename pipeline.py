"""
Pipeline orchestration for image editing and 3D asset generation via Lightning AI
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import json

from models.image_editor import QWENImageEditor
from models.hunyuan_3d import Hunyuan3DGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageTo3DPipeline:
    """
    Complete pipeline: Image Editing -> 3D Asset Generation
    """

    def __init__(
        self,
        output_dir: str = "./outputs"
    ):
        """
        Initialize pipeline with Lightning AI backend
        
        Args:
            output_dir: Output directory for all results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        logger.info("Initializing pipeline components...")
        
        try:
            self.image_editor = QWENImageEditor()
            logger.info("Image editor initialized")
        except Exception as e:
            logger.warning(f"Image editor initialization warning: {e}")
            self.image_editor = None
        
        # Initialize Hunyuan 3D with Lightning AI
        self.hunyuan_3d = Hunyuan3DGenerator(
            output_dir=str(self.output_dir)
        )
        logger.info("Hunyuan 3D generator initialized (Lightning AI)")
        
        self.execution_history = []

    def process_image_to_edited_image(
        self,
        image_path: str,
        edit_prompt: str,
        guidance_scale: float = 7.5
    ) -> Optional[str]:
        """
        Step 1: Edit input image
        
        Args:
            image_path: Path to input image
            edit_prompt: Prompt for image editing
            guidance_scale: Guidance scale for editing
            
        Returns:
            Path to edited image or None
        """
        try:
            logger.info(f"Starting image editing: {image_path}")
            
            if not self.image_editor:
                logger.warning("Image editor not available, skipping editing")
                edited_image_path = image_path
            else:
                # Edit image
                edited_image = self.image_editor.edit_image(
                    image_path,
                    edit_prompt,
                    guidance_scale=guidance_scale
                )
                
                # Save edited image
                edited_image_path = self.output_dir / f"edited_{Path(image_path).stem}.png"
                edited_image.save(str(edited_image_path))
                logger.info(f"Edited image saved: {edited_image_path}")
            
            return str(edited_image_path)
            
        except Exception as e:
            logger.error(f"Error in image editing: {e}")
            return None

    def generate_3d_from_image(
        self,
        image_path: str,
        generation_prompt: str = ""
    ) -> Optional[Dict]:
        """
        Step 2: Generate 3D asset from image via Lightning AI
        
        Args:
            image_path: Path to image (can be edited or original)
            generation_prompt: Prompt for 3D generation
            
        Returns:
            Dictionary with 3D asset information or None
        """
        try:
            logger.info(f"Starting 3D asset generation from: {image_path}")
            logger.info("Using Lightning AI for 3D generation")
            
            # Generate 3D asset via Lightning AI
            output_name = f"asset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = self.hunyuan_3d.generate_3d_asset(
                image_path,
                generation_prompt,
                output_name
            )
            
            if result:
                logger.info(f"3D asset generated: {output_name}")
                return result
            else:
                logger.error("3D asset generation failed")
                return None
                
        except Exception as e:
            logger.error(f"Error in 3D generation: {e}")
            return None

    def run_full_pipeline(
        self,
        input_image_path: str,
        edit_prompt: str,
        generation_prompt: str = ""
    ) -> Optional[Dict]:
        """
        Run complete pipeline: Edit image -> Generate 3D
        
        Args:
            input_image_path: Path to input image
            edit_prompt: Prompt for editing
            generation_prompt: Prompt for 3D generation
            
        Returns:
            Dictionary with pipeline results
        """
        execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"[{execution_id}] Starting full pipeline execution")
        
        result = {
            "execution_id": execution_id,
            "input_image": str(input_image_path),
            "edit_prompt": edit_prompt,
            "generation_prompt": generation_prompt,
            "steps": {}
        }
        
        try:
            # Step 1: Edit image
            logger.info("[Step 1] Editing image...")
            edited_image_path = self.process_image_to_edited_image(
                input_image_path,
                edit_prompt
            )
            
            if not edited_image_path:
                raise RuntimeError("Image editing failed")
            
            result["steps"]["image_editing"] = {
                "status": "success",
                "edited_image_path": edited_image_path
            }
            logger.info(f"[Step 1] ✓ Image editing completed")
            
            # Step 2: Generate 3D from edited image
            logger.info("[Step 2] Generating 3D asset...")
            asset_result = self.generate_3d_from_image(
                edited_image_path,
                generation_prompt
            )
            
            if not asset_result:
                raise RuntimeError("3D asset generation failed")
            
            result["steps"]["3d_generation"] = asset_result
            logger.info(f"[Step 2] ✓ 3D asset generation completed")
            
            result["status"] = "success"
            
            # Save execution results
            results_file = self.output_dir / f"results_{execution_id}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to: {results_file}")
            
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        if self.hunyuan_3d:
            self.hunyuan_3d.stop_docker_container()
        logger.info("Cleanup completed")

    def get_status(self) -> dict:
        """Get pipeline status"""
        return {
            "image_editor_available": self.image_editor is not None,
            "backend": "Lightning AI",
            "output_directory": str(self.output_dir),
            "execution_history_count": len(self.execution_history),
            "generated_assets": self.hunyuan_3d.list_generated_assets() if self.hunyuan_3d else []
        }

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

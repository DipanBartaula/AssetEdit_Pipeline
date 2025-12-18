"""
Gradio Web Interface for Image Editing and 3D Asset Generation Pipeline
GLB 3D Model Rendering Support
"""

import gradio as gr
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import shutil
from PIL import Image
import numpy as np

from pipeline import ImageTo3DPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineUI:
    """Gradio UI wrapper for the pipeline"""

    def __init__(self, output_dir: str = "./outputs"):
        """Initialize UI with pipeline"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipeline = ImageTo3DPipeline(
            output_dir=str(self.output_dir)
        )
        
        self.current_state = {
            "input_image": None,
            "edited_image": None,
            "asset_name": None,
            "execution_log": ""
        }

    def upload_image(self, image_file) -> str:
        """Handle image upload"""
        try:
            if image_file is None:
                return "No image uploaded", None
            
            # Save uploaded image
            img = Image.open(image_file.name).convert("RGB")
            saved_path = self.output_dir / f"input_{Path(image_file.name).stem}.png"
            img.save(str(saved_path))
            
            self.current_state["input_image"] = str(saved_path)
            
            return f"Image uploaded successfully: {saved_path.name}", img
            
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            return f"Error uploading image: {str(e)}", None

    def edit_image(
        self,
        edit_prompt: str,
        guidance_scale: float = 7.5
    ) -> Tuple[str, Optional[Image.Image]]:
        """Execute image editing"""
        try:
            if not self.current_state["input_image"]:
                return "Please upload an image first", None
            
            if not edit_prompt:
                return "Please enter an edit prompt", None
            
            logger.info(f"Starting image editing with prompt: {edit_prompt}")
            
            # Edit image
            edited_path = self.pipeline.process_image_to_edited_image(
                self.current_state["input_image"],
                edit_prompt,
                guidance_scale
            )
            
            if not edited_path:
                return "Image editing failed", None
            
            self.current_state["edited_image"] = edited_path
            edited_img = Image.open(edited_path)
            
            message = f"Image edited successfully!\nPrompt: {edit_prompt}"
            return message, edited_img
            
        except Exception as e:
            logger.error(f"Error in image editing: {e}")
            return f"Error: {str(e)}", None

    def generate_3d_asset(
        self,
        generation_prompt: str = ""
    ) -> Tuple[str, Optional[str]]:
        """Execute 3D asset generation"""
        try:
            image_to_use = self.current_state["edited_image"] or self.current_state["input_image"]
            
            if not image_to_use:
                return "Please upload and optionally edit an image first", None
            
            logger.info(f"Starting 3D asset generation with prompt: {generation_prompt}")
            
            # Generate 3D asset
            result = self.pipeline.generate_3d_from_image(
                image_to_use,
                generation_prompt
            )
            
            if not result:
                return "3D asset generation failed", None
            
            asset_name = result.get("asset_name", "unknown")
            self.current_state["asset_name"] = asset_name
            
            message = f"""3D Asset Generation Completed!
Asset Name: {asset_name}
Status: {result.get('status', 'unknown')}
Model Path: {result.get('model_path', 'N/A')}
"""
            
            return message, asset_name
            
        except Exception as e:
            logger.error(f"Error in 3D generation: {e}")
            return f"Error: {str(e)}", None

    def get_asset_preview(self) -> Tuple[str, Optional[str]]:
        """Get preview of generated asset (.glb format)"""
        try:
            if not self.current_state["asset_name"]:
                return "No asset generated yet", None
            
            asset_info = self.pipeline.hunyuan_3d.get_asset_preview(
                self.current_state["asset_name"]
            )
            
            if not asset_info:
                return "Could not retrieve asset preview", None
            
            info_text = f"""Asset Information (GLB Format):
Asset Name: {asset_info.get('asset_name', 'N/A')}
Format: {asset_info.get('format', 'glb')}
File Exists: {asset_info.get('model_exists', False)}
Model Path: {asset_info.get('model_path', 'N/A')}
Size (MB): {asset_info.get('model_size_mb', 'N/A')}
"""
            
            if asset_info.get('model_exists'):
                glb_path = asset_info.get('model_path')
                return info_text, glb_path
            
            return info_text, None
            
        except Exception as e:
            logger.error(f"Error getting asset preview: {e}")
            return f"Error: {str(e)}", None

    def run_full_pipeline(
        self,
        edit_prompt: str,
        generation_prompt: str,
        guidance_scale: float = 7.5
    ) -> str:
        """Run complete pipeline in one go"""
        try:
            if not self.current_state["input_image"]:
                return "Please upload an image first"
            
            if not edit_prompt:
                return "Please enter an edit prompt"
            
            logger.info("Running full pipeline...")
            
            result = self.pipeline.run_full_pipeline(
                self.current_state["input_image"],
                edit_prompt,
                generation_prompt
            )
            
            if result.get("status") == "success":
                asset_name = result.get("steps", {}).get("3d_generation", {}).get("asset_name", "unknown")
                self.current_state["asset_name"] = asset_name
                
                summary = f"""Pipeline Execution Completed Successfully!

✓ Image Editing: Completed
  - Prompt: {edit_prompt}
  - Edited Image: {result.get('steps', {}).get('image_editing', {}).get('edited_image_path', 'N/A')}

✓ 3D Asset Generation: Completed
  - Asset Name: {asset_name}
  - Model Path: {result.get('steps', {}).get('3d_generation', {}).get('model_path', 'N/A')}

Execution ID: {result.get('execution_id', 'N/A')}
"""
            else:
                summary = f"""Pipeline Execution Failed!
Error: {result.get('error', 'Unknown error')}
Execution ID: {result.get('execution_id', 'N/A')}
"""
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
            return f"Error: {str(e)}"

    def get_pipeline_status(self) -> str:
        """Get current pipeline status"""
        try:
            status = self.pipeline.get_status()
            
            status_text = f"""Pipeline Status:

Image Editor Available: {status.get('image_editor_available', False)}
Hunyuan 3D Running: {status.get('hunyuan_3d_running', False)}
Output Directory: {status.get('output_directory', 'N/A')}
Executions Completed: {status.get('execution_history_count', 0)}

Generated Assets: {len(status.get('generated_assets', []))}
"""
            
            if status.get('generated_assets'):
                status_text += "Assets: " + ", ".join(status['generated_assets'][:5])
            
            return status_text
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return f"Error: {str(e)}"


def create_interface():
    """Create Gradio interface"""
    
    ui = PipelineUI()
    
    with gr.Blocks(title="Image to 3D Asset Pipeline", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown(
            """
            # 🎨 Image Editing & 3D Asset Generation Pipeline
            
            **Complete Workflow**: Edit Images with QWENImage Edit 2509 → Generate 3D Assets with Hunyuan 3D 2.1
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Image")
                input_image = gr.Image(
                    label="Upload Image",
                    type="filepath",
                    interactive=True
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False
                )
                upload_btn = gr.Button("Upload Image", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Pipeline Status")
                status_display = gr.Textbox(
                    label="System Status",
                    interactive=False,
                    lines=8
                )
                refresh_status_btn = gr.Button("Refresh Status")
        
        # Image Editing Section
        with gr.Row():
            gr.Markdown("## 🖼️ Image Editing Section")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Edit Configuration")
                edit_prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="Describe how you want to edit the image...",
                    lines=3
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    label="Guidance Scale"
                )
                edit_btn = gr.Button("🎨 Edit Image", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### Edited Image Preview")
                edited_image_display = gr.Image(
                    label="Edited Image",
                    interactive=False
                )
                edit_status = gr.Textbox(
                    label="Edit Status",
                    interactive=False,
                    lines=3
                )
        
        # 3D Generation Section
        with gr.Row():
            gr.Markdown("## 🎯 3D Asset Generation Section")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Generation Configuration")
                generation_prompt = gr.Textbox(
                    label="Generation Prompt (Optional)",
                    placeholder="Additional instructions for 3D generation...",
                    lines=3
                )
                generate_3d_btn = gr.Button("🚀 Generate 3D Asset", variant="primary", size="lg")
                generation_status = gr.Textbox(
                    label="Generation Status",
                    interactive=False,
                    lines=4
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### GLB 3D Model Viewer")
                asset_info = gr.Textbox(
                    label="Asset Information",
                    interactive=False,
                    lines=5
                )
                glb_model = gr.Model3D(
                    label="3D Model (GLB Format)",
                    interactive=False
                )
                refresh_preview_btn = gr.Button("🔄 Refresh Preview")
        
        # Full Pipeline Section
        with gr.Row():
            gr.Markdown("## ⚡ Full Pipeline Execution")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Run Complete Pipeline")
                gr.Markdown("Execute the entire workflow from image editing to 3D generation in one action.")
                
                full_pipeline_prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="Describe how you want to edit the image...",
                    lines=2
                )
                full_pipeline_generation = gr.Textbox(
                    label="Generation Prompt (Optional)",
                    placeholder="Additional 3D generation instructions...",
                    lines=2
                )
                full_pipeline_btn = gr.Button("⚡ Execute Full Pipeline", variant="primary", size="lg")
                full_pipeline_output = gr.Textbox(
                    label="Pipeline Output",
                    interactive=False,
                    lines=6
                )
        
        # Event handlers
        upload_btn.click(
            fn=ui.upload_image,
            inputs=[input_image],
            outputs=[upload_status, input_image]
        )
        
        refresh_status_btn.click(
            fn=ui.get_pipeline_status,
            outputs=[status_display]
        )
        
        edit_btn.click(
            fn=ui.edit_image,
            inputs=[edit_prompt, guidance_scale],
            outputs=[edit_status, edited_image_display]
        )
        
        generate_3d_btn.click(
            fn=ui.generate_3d_asset,
            inputs=[generation_prompt],
            outputs=[generation_status, gr.Textbox(visible=False)]
        )
        
        refresh_preview_btn.click(
            fn=ui.get_asset_preview,
            outputs=[asset_info, glb_model]
        )
        
        full_pipeline_btn.click(
            fn=ui.run_full_pipeline,
            inputs=[full_pipeline_prompt, full_pipeline_generation, guidance_scale],
            outputs=[full_pipeline_output]
        )
        
        # Load status on startup
        interface.load(fn=ui.get_pipeline_status, outputs=[status_display])
    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

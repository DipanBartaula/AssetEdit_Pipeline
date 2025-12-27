"""
Gradio Web Interface for Integrated Qwen + Hunyuan 3D Pipeline
===============================================================
A unified web interface for:
1. Image editing with Qwen Image Edit Plus
2. 3D asset generation with Hunyuan 3D 2.1
3. Full pipeline execution (Edit → 3D)
"""

import gradio as gr
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from integrated_pipeline import (
    IntegratedPipeline,
    QwenImageEditor,
    Hunyuan3DGenerator
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedPipelineUI:
    """Gradio UI for the integrated pipeline"""
    
    def __init__(self, output_dir: str = "./outputs"):
        """Initialize UI with pipeline components"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = IntegratedPipeline(
            output_dir=str(self.output_dir),
            auto_load_models=False  # Lazy load to save memory
        )
        
        # State tracking
        self.current_state = {
            "input_image": None,
            "edited_image": None,
            "edited_image_path": None,
            "asset_name": None,
            "asset_path": None
        }
        
        logger.info("IntegratedPipelineUI initialized")
    
    def load_image(self, image) -> Tuple[str, Optional[Image.Image]]:
        """Handle image upload"""
        try:
            if image is None:
                return "⚠️ No image uploaded", None
            
            # Handle different input types
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
                image_path = image
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert("RGB")
                image_path = self.output_dir / f"input_{int(time.time())}.png"
                img.save(str(image_path))
            else:
                img = image.convert("RGB")
                image_path = self.output_dir / f"input_{int(time.time())}.png"
                img.save(str(image_path))
            
            self.current_state["input_image"] = str(image_path)
            
            return f"✓ Image loaded successfully", img
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return f"✗ Error: {str(e)}", None
    
    def edit_image(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        num_steps: int = 40,
        seed: int = 0
    ) -> Tuple[str, Optional[Image.Image]]:
        """Edit image with Qwen Image Edit Plus"""
        try:
            if image is None:
                return "⚠️ Please upload an image first", None
            
            if not prompt or prompt.strip() == "":
                return "⚠️ Please enter an edit prompt", None
            
            logger.info(f"Editing image with prompt: {prompt}")
            
            # Ensure model is loaded
            if not self.pipeline.image_editor.is_loaded:
                logger.info("Loading Qwen Image Edit model (first use)...")
            
            # Edit image
            edited_images = self.pipeline.image_editor.edit_image(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                seed=seed
            )
            
            edited_image = edited_images[0]
            
            # Save edited image
            import time
            edited_path = self.output_dir / f"edited_{int(time.time())}.png"
            edited_image.save(str(edited_path))
            
            self.current_state["edited_image"] = edited_image
            self.current_state["edited_image_path"] = str(edited_path)
            
            message = f"""✓ Image edited successfully!
Prompt: {prompt}
Saved to: {edited_path.name}"""
            
            return message, edited_image
            
        except Exception as e:
            logger.error(f"Error editing image: {e}")
            return f"✗ Error: {str(e)}", None
    
    def generate_3d(
        self,
        image,
        prompt: str = ""
    ) -> Tuple[str, Optional[str]]:
        """Generate 3D asset from image"""
        try:
            # Use edited image if available, otherwise use input
            if image is None:
                if self.current_state["edited_image"] is not None:
                    image = self.current_state["edited_image"]
                elif self.current_state["input_image"]:
                    image = self.current_state["input_image"]
                else:
                    return "⚠️ Please upload or edit an image first", None
            
            import time
            output_name = f"asset_{int(time.time())}"
            
            logger.info(f"Generating 3D asset: {output_name}")
            
            result = self.pipeline.generator_3d.generate_3d_asset(
                image=image,
                prompt=prompt,
                output_name=output_name
            )
            
            if not result:
                return "✗ 3D generation failed", None
            
            self.current_state["asset_name"] = output_name
            self.current_state["asset_path"] = result["model_path"]
            
            message = f"""✓ 3D Asset Generated!
Asset Name: {output_name}
Format: {result['format']}
Size: {result['file_size_mb']:.2f} MB
Path: {result['model_path']}"""
            
            return message, result["model_path"]
            
        except Exception as e:
            logger.error(f"Error generating 3D: {e}")
            return f"✗ Error: {str(e)}", None
    
    def run_full_pipeline(
        self,
        image,
        edit_prompt: str,
        generation_prompt: str = "",
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        num_steps: int = 40,
        seed: int = 0
    ) -> Tuple[str, Optional[Image.Image], Optional[str]]:
        """Run the full pipeline: Edit → 3D"""
        try:
            if image is None:
                return "⚠️ Please upload an image first", None, None
            
            if not edit_prompt or edit_prompt.strip() == "":
                return "⚠️ Please enter an edit prompt", None, None
            
            logger.info("Running full pipeline...")
            
            result = self.pipeline.run_full_pipeline(
                input_image=image,
                edit_prompt=edit_prompt,
                generation_prompt=generation_prompt,
                edit_kwargs={
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_steps,
                    "seed": seed
                }
            )
            
            if result["status"] != "success":
                return f"✗ Pipeline failed: {result.get('error', 'Unknown error')}", None, None
            
            # Get outputs
            edited_path = result["steps"]["image_editing"]["edited_image_path"]
            edited_image = Image.open(edited_path)
            model_path = result["steps"]["3d_generation"]["model_path"]
            
            self.current_state["edited_image"] = edited_image
            self.current_state["edited_image_path"] = edited_path
            self.current_state["asset_path"] = model_path
            
            message = f"""✓ Pipeline Completed Successfully!

📝 Execution ID: {result['execution_id']}

🎨 Image Editing:
   • Prompt: {edit_prompt}
   • Output: {Path(edited_path).name}

🎯 3D Generation:
   • Prompt: {generation_prompt or 'None'}
   • Format: {result['steps']['3d_generation']['format']}
   • Model: {Path(model_path).name}
"""
            
            return message, edited_image, model_path
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            return f"✗ Error: {str(e)}", None, None
    
    def get_status(self) -> str:
        """Get pipeline status"""
        status = self.pipeline.get_status()
        
        return f"""📊 Pipeline Status

🔧 Components:
   • Image Editor Loaded: {'✓' if status['image_editor_loaded'] else '○ (will load on first use)'}
   • Device: {status['device']}

📁 Outputs:
   • Directory: {status['output_directory']}
   • Total Executions: {status['execution_count']}
   • Generated Assets: {len(status['generated_assets'])}

🎯 Current Session:
   • Input Image: {'✓' if self.current_state['input_image'] else '○'}
   • Edited Image: {'✓' if self.current_state['edited_image'] else '○'}
   • 3D Asset: {'✓' if self.current_state['asset_path'] else '○'}
"""


def create_interface():
    """Create the Gradio interface"""
    
    ui = IntegratedPipelineUI()
    
    with gr.Blocks(
        title="Qwen + Hunyuan 3D Pipeline",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .title { text-align: center; }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🎨 Integrated Image Edit & 3D Generation Pipeline
            
            **Qwen Image Edit Plus** → Edit images with AI | **Hunyuan 3D 2.1** → Generate 3D assets
            
            ---
            """,
            elem_classes=["title"]
        )
        
        with gr.Tabs():
            # =====================
            # TAB 1: Image Editing
            # =====================
            with gr.TabItem("🖼️ Image Editing", id="edit"):
                gr.Markdown("### Edit Images with Qwen Image Edit Plus")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        edit_input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            interactive=True
                        )
                        
                        edit_prompt = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="Describe the edits you want...",
                            lines=3,
                            value="The person is wearing sunglasses"
                        )
                        
                        edit_negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="Things to avoid...",
                            lines=2
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            edit_true_cfg = gr.Slider(
                                minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                                label="True CFG Scale"
                            )
                            edit_guidance = gr.Slider(
                                minimum=0.0, maximum=5.0, value=1.0, step=0.1,
                                label="Guidance Scale"
                            )
                            edit_steps = gr.Slider(
                                minimum=10, maximum=100, value=40, step=1,
                                label="Inference Steps"
                            )
                            edit_seed = gr.Slider(
                                minimum=0, maximum=1000000, value=0, step=1,
                                label="Random Seed"
                            )
                        
                        edit_btn = gr.Button("🎨 Edit Image", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        edit_output_image = gr.Image(
                            label="Edited Image",
                            type="pil",
                            interactive=False
                        )
                        edit_status = gr.Textbox(
                            label="Status",
                            lines=4,
                            interactive=False
                        )
                
                # Examples
                gr.Markdown("### 💡 Example Prompts")
                gr.Examples(
                    examples=[
                        ["The dog is wearing sunglasses and a hat"],
                        ["Add a beautiful sunset background"],
                        ["Transform into an oil painting style"],
                        ["Make the scene look like winter with snow"],
                        ["Add magical sparkles and glow effects"],
                    ],
                    inputs=[edit_prompt],
                )
                
                edit_btn.click(
                    fn=ui.edit_image,
                    inputs=[
                        edit_input_image, edit_prompt, edit_negative_prompt,
                        edit_true_cfg, edit_guidance, edit_steps, edit_seed
                    ],
                    outputs=[edit_status, edit_output_image]
                )
            
            # =====================
            # TAB 2: 3D Generation
            # =====================
            with gr.TabItem("🎯 3D Generation", id="3d"):
                gr.Markdown("### Generate 3D Assets with Hunyuan 3D 2.1")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            interactive=True
                        )
                        
                        gen_prompt = gr.Textbox(
                            label="Generation Prompt (Optional)",
                            placeholder="Additional details for 3D generation...",
                            lines=3
                        )
                        
                        gen_btn = gr.Button("🚀 Generate 3D Asset", variant="primary", size="lg")
                        gen_status = gr.Textbox(
                            label="Status",
                            lines=5,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gen_model = gr.Model3D(
                            label="3D Model (GLB)",
                            interactive=False
                        )
                
                gen_btn.click(
                    fn=ui.generate_3d,
                    inputs=[gen_input_image, gen_prompt],
                    outputs=[gen_status, gen_model]
                )
            
            # =====================
            # TAB 3: Full Pipeline
            # =====================
            with gr.TabItem("⚡ Full Pipeline", id="pipeline"):
                gr.Markdown(
                    """
                    ### Complete Workflow: Edit Image → Generate 3D
                    
                    Run the entire pipeline in one click. Upload an image, describe the edits,
                    and get both an edited image and a 3D model.
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        pipe_input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            interactive=True
                        )
                        
                        pipe_edit_prompt = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="Describe the image edits...",
                            lines=3,
                            value="Transform the subject into a fantasy character"
                        )
                        
                        pipe_gen_prompt = gr.Textbox(
                            label="3D Generation Prompt (Optional)",
                            placeholder="Additional 3D generation instructions...",
                            lines=2
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            pipe_true_cfg = gr.Slider(
                                minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                                label="True CFG Scale"
                            )
                            pipe_guidance = gr.Slider(
                                minimum=0.0, maximum=5.0, value=1.0, step=0.1,
                                label="Guidance Scale"
                            )
                            pipe_steps = gr.Slider(
                                minimum=10, maximum=100, value=40, step=1,
                                label="Inference Steps"
                            )
                            pipe_seed = gr.Slider(
                                minimum=0, maximum=1000000, value=0, step=1,
                                label="Random Seed"
                            )
                        
                        pipe_btn = gr.Button(
                            "⚡ Run Full Pipeline",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        pipe_status = gr.Textbox(
                            label="Pipeline Status",
                            lines=12,
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        pipe_edited_image = gr.Image(
                            label="Edited Image",
                            type="pil",
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        pipe_model = gr.Model3D(
                            label="3D Model (GLB)",
                            interactive=False
                        )
                
                pipe_btn.click(
                    fn=ui.run_full_pipeline,
                    inputs=[
                        pipe_input_image, pipe_edit_prompt, pipe_gen_prompt,
                        pipe_true_cfg, pipe_guidance, pipe_steps, pipe_seed
                    ],
                    outputs=[pipe_status, pipe_edited_image, pipe_model]
                )
            
            # =====================
            # TAB 4: Status
            # =====================
            with gr.TabItem("📊 Status", id="status"):
                gr.Markdown("### Pipeline Status & Information")
                
                status_display = gr.Textbox(
                    label="Current Status",
                    lines=15,
                    interactive=False
                )
                
                refresh_btn = gr.Button("🔄 Refresh Status")
                
                refresh_btn.click(
                    fn=ui.get_status,
                    outputs=[status_display]
                )
                
                # Load status on tab open
                interface.load(fn=ui.get_status, outputs=[status_display])
                
                gr.Markdown(
                    """
                    ---
                    
                    ### 📝 Notes
                    
                    - **Qwen Image Edit Plus** requires a GPU with ~16GB VRAM
                    - **Hunyuan 3D** uses Lightning AI (requires API key)
                    - Set `LIGHTNING_API_KEY` environment variable for 3D generation
                    - First image edit will take longer due to model loading
                    """
                )
        
        gr.Markdown(
            """
            ---
            <center>
            <small>Integrated Pipeline: Qwen Image Edit Plus + Hunyuan 3D 2.1</small>
            </center>
            """
        )
    
    return interface


if __name__ == "__main__":
    import time  # Import here for the UI class
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

"""
Integrated Pipeline: Qwen Image Edit → Hunyuan 3D Asset Generation
====================================================================
Complete workflow:
1. Upload an image
2. Edit the image with Qwen Image Edit Plus
3. Generate 3D asset from the edited image using Hunyuan 3D 2.1
4. View and download the 3D model (GLB format)
"""

import gradio as gr
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from integrated_pipeline import IntegratedPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
OUTPUT_DIR = "./outputs"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Optional: configure Hunyuan3D Docker image via environment variable
HUNYUAN_DOCKER_IMAGE = os.environ.get("HUNYUAN3D_DOCKER_IMAGE")

pipeline = IntegratedPipeline(
    output_dir=OUTPUT_DIR,
    auto_load_models=False,  # Lazy load to save memory
    hunyuan_docker_image=HUNYUAN_DOCKER_IMAGE,
)

logger.info("Pipeline initialized")


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def edit_image_step(
    image,
    prompt: str,
    negative_prompt: str,
    true_cfg_scale: float,
    guidance_scale: float,
    num_steps: int,
    seed: int
):
    """Step 1: Edit image with Qwen Image Edit Plus"""
    if image is None:
        return None, "⚠️ Please upload an image first"
    
    if not prompt or prompt.strip() == "":
        return None, "⚠️ Please enter an edit prompt"
    
    try:
        logger.info(f"[Step 1] Editing image with prompt: {prompt}")
        
        # Ensure model is loaded
        if not pipeline.image_editor.is_loaded:
            logger.info("Loading Qwen Image Edit model...")
        
        edited_images = pipeline.image_editor.edit_image(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else "",
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            seed=seed
        )
        
        edited_image = edited_images[0]
        
        # Save edited image
        timestamp = int(time.time())
        edited_path = Path(OUTPUT_DIR) / f"edited_{timestamp}.png"
        edited_image.save(str(edited_path))
        
        logger.info(f"[Step 1] ✓ Image edited and saved: {edited_path}")
        
        return edited_image, f"✓ Image edited successfully!\nSaved to: {edited_path.name}\n\nNow proceed to Step 2 to generate 3D asset."
        
    except Exception as e:
        logger.error(f"Error editing image: {e}")
        return None, f"✗ Error: {str(e)}"


def generate_3d_step(image, prompt: str):
    """Step 2: Generate 3D asset from edited image using Hunyuan 3D"""
    if image is None:
        return None, "⚠️ Please provide an edited image first (complete Step 1)"
    
    try:
        timestamp = int(time.time())
        output_name = f"asset_{timestamp}"
        
        logger.info(f"[Step 2] Generating 3D asset: {output_name}")
        
        result = pipeline.generator_3d.generate_3d_asset(
            image=image,
            prompt=prompt if prompt else "",
            output_name=output_name
        )
        
        if not result:
            return None, "✗ 3D generation failed. Ensure Docker is installed, GPU is available, and the Hunyuan3D Docker image is configured."
        
        model_path = result["model_path"]
        
        status = f"""✓ 3D Asset Generated!

📦 Asset Name: {output_name}
📁 Format: {result['format'].upper()}
💾 Size: {result.get('file_size_mb', 0):.2f} MB
📍 Path: {model_path}

You can now view the 3D model below and download it."""
        
        logger.info(f"[Step 2] ✓ 3D asset generated: {model_path}")
        
        return model_path, status
        
    except Exception as e:
        logger.error(f"Error generating 3D: {e}")
        return None, f"✗ Error: {str(e)}"


def run_complete_pipeline(
    image,
    edit_prompt: str,
    generation_prompt: str,
    true_cfg_scale: float,
    guidance_scale: float,
    num_steps: int,
    seed: int
):
    """Run the complete pipeline: Edit Image → Generate 3D → Return both"""
    if image is None:
        return None, None, "⚠️ Please upload an image first"
    
    if not edit_prompt or edit_prompt.strip() == "":
        return None, None, "⚠️ Please enter an edit prompt"
    
    try:
        logger.info("=" * 50)
        logger.info("Running Complete Pipeline: Edit → 3D")
        logger.info("=" * 50)
        
        # Run full pipeline
        result = pipeline.run_full_pipeline(
            input_image=image,
            edit_prompt=edit_prompt,
            generation_prompt=generation_prompt if generation_prompt else "",
            edit_kwargs={
                "true_cfg_scale": true_cfg_scale,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_steps,
                "seed": seed
            }
        )
        
        if result["status"] != "success":
            error_msg = result.get('error', 'Unknown error')
            return None, None, f"✗ Pipeline failed: {error_msg}"
        
        # Get outputs
        edited_path = result["steps"]["image_editing"]["edited_image_path"]
        edited_image = Image.open(edited_path)
        model_path = result["steps"]["3d_generation"]["model_path"]
        
        status = f"""✓ Pipeline Completed Successfully!

═══════════════════════════════════════
📋 Execution ID: {result['execution_id']}
═══════════════════════════════════════

🎨 STEP 1: Image Editing (Qwen)
   • Prompt: {edit_prompt}
   • Output: {Path(edited_path).name}

🎯 STEP 2: 3D Generation (Hunyuan)
   • Prompt: {generation_prompt if generation_prompt else 'None'}
   • Format: GLB
   • Model: {Path(model_path).name}

═══════════════════════════════════════
View the 3D model below. You can rotate, 
zoom, and download it.
═══════════════════════════════════════"""
        
        logger.info("Pipeline completed successfully!")
        
        return edited_image, model_path, status
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        return None, None, f"✗ Error: {str(e)}"


def get_status():
    """Get pipeline status"""
    status = pipeline.get_status()
    
    return f"""📊 Pipeline Status
═══════════════════════════════════════

🔧 Components:
   • Qwen Image Editor: {'✓ Loaded' if status['image_editor_loaded'] else '○ Ready (loads on first use)'}
   • Device: {status['device'].upper()}

📁 Output Directory: {status['output_directory']}
📦 Generated Assets: {len(status['generated_assets'])}

💡 Tips:
   • First edit may take time to load the model
   • Docker with GPU and the Hunyuan3D image are required for 3D generation
   • 3D models are saved in GLB format
"""


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(
    title="Qwen + Hunyuan 3D Pipeline",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown(
        """
        # 🎨 Image Edit → 3D Asset Pipeline
        
        **Complete Workflow:**
        1. Upload your image
        2. Edit it with **Qwen Image Edit Plus** (AI-powered editing)
        3. Generate a **3D asset** from the edited image using **Hunyuan 3D 2.1**
        4. View and download the 3D model (GLB format)
        
        ---
        """
    )
    
    # =================================
    # TAB: Step-by-Step Pipeline
    # =================================
    with gr.Tab("📝 Step-by-Step Pipeline"):
        gr.Markdown("### Follow the steps below to edit an image and generate a 3D asset")
        
        # STEP 1: Image Editing
        gr.Markdown("## Step 1: Edit Image with Qwen")
        
        with gr.Row():
            with gr.Column(scale=1):
                step1_input = gr.Image(
                    label="📷 Upload Original Image",
                    type="pil"
                )
                step1_prompt = gr.Textbox(
                    label="✏️ Edit Prompt",
                    placeholder="Describe how to edit the image...",
                    value="Add sunglasses and a cool hat",
                    lines=2
                )
                step1_negative = gr.Textbox(
                    label="❌ Negative Prompt (Optional)",
                    placeholder="Things to avoid..."
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    step1_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="True CFG Scale")
                    step1_guidance = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Guidance Scale")
                    step1_steps = gr.Slider(10, 100, value=40, step=1, label="Inference Steps")
                    step1_seed = gr.Slider(0, 1000000, value=0, step=1, label="Random Seed")
                
                step1_btn = gr.Button("🎨 Edit Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                step1_output = gr.Image(
                    label="🖼️ Edited Image",
                    type="pil"
                )
                step1_status = gr.Textbox(label="Status", lines=4)
        
        step1_btn.click(
            fn=edit_image_step,
            inputs=[step1_input, step1_prompt, step1_negative, step1_cfg, step1_guidance, step1_steps, step1_seed],
            outputs=[step1_output, step1_status]
        )
        
        gr.Markdown("---")
        
        # STEP 2: 3D Generation
        gr.Markdown("## Step 2: Generate 3D Asset from Edited Image")
        
        with gr.Row():
            with gr.Column(scale=1):
                step2_input = gr.Image(
                    label="🖼️ Edited Image (from Step 1 or upload new)",
                    type="pil"
                )
                step2_prompt = gr.Textbox(
                    label="📝 3D Generation Prompt (Optional)",
                    placeholder="Additional details for 3D generation...",
                    lines=2
                )
                step2_btn = gr.Button("🚀 Generate 3D Asset", variant="primary", size="lg")
                step2_status = gr.Textbox(label="Status", lines=6)
            
            with gr.Column(scale=1):
                step2_model = gr.Model3D(
                    label="🎮 3D Model Viewer (GLB)",
                    clear_color=[0.1, 0.1, 0.1, 1.0]
                )
        
        step2_btn.click(
            fn=generate_3d_step,
            inputs=[step2_input, step2_prompt],
            outputs=[step2_model, step2_status]
        )
        
        # Automatically copy edited image to step 2
        step1_output.change(
            fn=lambda x: x,
            inputs=[step1_output],
            outputs=[step2_input]
        )
    
    # =================================
    # TAB: One-Click Pipeline
    # =================================
    with gr.Tab("⚡ One-Click Pipeline"):
        gr.Markdown(
            """
            ### Complete Pipeline in One Click
            Upload an image, set your prompts, and get both the edited image and 3D model automatically.
            
            **Workflow:** Upload Image → Qwen Edit → Hunyuan 3D → View Result
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                full_input = gr.Image(
                    label="📷 Upload Image",
                    type="pil"
                )
                full_edit_prompt = gr.Textbox(
                    label="✏️ Edit Prompt (for Qwen)",
                    placeholder="Describe how to edit the image...",
                    value="Transform into a fantasy warrior with armor",
                    lines=2
                )
                full_gen_prompt = gr.Textbox(
                    label="🎮 3D Generation Prompt (Optional, for Hunyuan)",
                    placeholder="Additional 3D generation details...",
                    lines=2
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    full_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="True CFG Scale")
                    full_guidance = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Guidance Scale")
                    full_steps = gr.Slider(10, 100, value=40, step=1, label="Inference Steps")
                    full_seed = gr.Slider(0, 1000000, value=0, step=1, label="Random Seed")
                
                full_btn = gr.Button("⚡ Run Complete Pipeline", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                full_status = gr.Textbox(label="Pipeline Status", lines=20)
        
        gr.Markdown("### Results")
        
        with gr.Row():
            with gr.Column(scale=1):
                full_edited = gr.Image(
                    label="🖼️ Edited Image (from Qwen)",
                    type="pil"
                )
            with gr.Column(scale=1):
                full_model = gr.Model3D(
                    label="🎮 3D Model (from Hunyuan)",
                    clear_color=[0.1, 0.1, 0.1, 1.0]
                )
        
        full_btn.click(
            fn=run_complete_pipeline,
            inputs=[full_input, full_edit_prompt, full_gen_prompt, full_cfg, full_guidance, full_steps, full_seed],
            outputs=[full_edited, full_model, full_status]
        )
    
    # =================================
    # TAB: Status & Info
    # =================================
    with gr.Tab("📊 Status"):
        gr.Markdown("### Pipeline Status & Information")
        
        status_display = gr.Textbox(label="Current Status", lines=15)
        refresh_btn = gr.Button("🔄 Refresh Status")
        
        refresh_btn.click(fn=get_status, outputs=[status_display])
        demo.load(fn=get_status, outputs=[status_display])
        
        gr.Markdown(
            """
            ---
            
            ### 📌 Requirements
            
            - **LIGHTNING_API_KEY**: Required for Hunyuan 3D generation
              ```
              # Windows
              set LIGHTNING_API_KEY=your_key_here
              
              # Linux/Mac
              export LIGHTNING_API_KEY=your_key_here
              ```
            
            - **GPU**: Recommended for faster image editing (16GB+ VRAM)
            
            ### 📁 Output Files
            
            - Edited images: `outputs/edited_*.png`
            - 3D models: `outputs/3d_assets/*.glb`
            
            ### 🔄 Pipeline Flow
            
            ```
            ┌─────────────┐     ┌─────────────────┐     ┌───────────────┐
            │ Input Image │ ──▶ │ Qwen Image Edit │ ──▶ │ Edited Image  │
            └─────────────┘     └─────────────────┘     └───────┬───────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────┐
                                                    │   Hunyuan 3D 2.1  │
                                                    └─────────┬─────────┘
                                                              │
                                                              ▼
                                                    ┌───────────────────┐
                                                    │  3D Model (.glb)  │
                                                    └───────────────────┘
            ```
            """
        )
    
    gr.Markdown(
        """
        ---
        <center>
        <b>Integrated Pipeline:</b> Qwen Image Edit Plus → Hunyuan 3D 2.1<br>
        <small>Edit images with AI → Generate 3D assets</small>
        </center>
        """
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=True
    )

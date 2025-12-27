import os
import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Load pipeline once at startup
print("Loading pipeline...")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
print("Pipeline loaded successfully!")

def edit_image(image1, image2, prompt, negative_prompt, true_cfg_scale, 
               guidance_scale, num_steps, seed, num_outputs):
    """
    Edit image(s) based on text prompt
    
    Args:
        image1: First input image (optional)
        image2: Second input image (required if image1 is None)
        prompt: Text description of desired edits
        negative_prompt: Things to avoid in the output
        true_cfg_scale: True CFG scale value
        guidance_scale: Guidance scale for generation
        num_steps: Number of inference steps
        seed: Random seed for reproducibility
        num_outputs: Number of images to generate
    """
    # Validate inputs
    if image1 is None and image2 is None:
        return None, "Please upload at least one image"
    
    if not prompt or prompt.strip() == "":
        return None, "Please provide a prompt"
    
    # Prepare images list
    images = []
    if image1 is not None:
        images.append(Image.fromarray(image1))
    if image2 is not None:
        images.append(Image.fromarray(image2))
    
    # If only one image provided, use it
    if len(images) == 1:
        input_images = [images[0]]
    else:
        input_images = images
    
    try:
        # Set up generator with seed
        generator = torch.manual_seed(seed)
        
        # Prepare inputs
        inputs = {
            "image": input_images,
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt if negative_prompt else " ",
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_outputs,
        }
        
        # Generate
        with torch.inference_mode():
            output = pipeline(**inputs)
            output_images = output.images
            
            # Save the first output
            output_path = "output_image_edit_plus.png"
            output_images[0].save(output_path)
            
            status = f"✓ Generated {len(output_images)} image(s) successfully!\nSaved to: {os.path.abspath(output_path)}"
            
            # Return first image and status (Gradio will display it)
            return output_images[0], status
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Qwen Image Edit Plus") as demo:
    gr.Markdown("""
    # Qwen Image Edit Plus
    Upload one or two images and describe the edits you want to make.
    """)
    
    with gr.Row():
        with gr.Column():
            image1_input = gr.Image(label="Image 1 (Optional)", type="numpy")
            image2_input = gr.Image(label="Image 2 (Primary)", type="numpy")
            
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the edits you want to make...",
                lines=3,
                value="The dog is wearing sunglasses and cap."
            )
            
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="Things to avoid in the output...",
                lines=2,
                value=""
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                true_cfg_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                    label="True CFG Scale"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0, maximum=5.0, value=1.0, step=0.1,
                    label="Guidance Scale"
                )
                num_steps = gr.Slider(
                    minimum=10, maximum=100, value=40, step=1,
                    label="Number of Inference Steps"
                )
                seed = gr.Slider(
                    minimum=0, maximum=1000000, value=0, step=1,
                    label="Random Seed"
                )
                num_outputs = gr.Slider(
                    minimum=1, maximum=4, value=1, step=1,
                    label="Number of Images to Generate"
                )
            
            generate_btn = gr.Button("Generate", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")
            status_text = gr.Textbox(label="Status", lines=2)
    
    # Examples
    gr.Markdown("### Example Prompts")
    gr.Examples(
        examples=[
            ["The dog is wearing sunglasses and cap."],
            ["Make the cat wear a wizard hat and hold a magic wand."],
            ["Add a beautiful sunset background with mountains."],
            ["Turn the scene into a painting in Van Gogh style."],
        ],
        inputs=[prompt_input],
    )
    
    # Connect the button
    generate_btn.click(
        fn=edit_image,
        inputs=[
            image1_input, image2_input, prompt_input, negative_prompt_input,
            true_cfg_scale, guidance_scale, num_steps, seed, num_outputs
        ],
        outputs=[output_image, status_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
#!/usr/bin/env python3
"""
Hunyuan3D-2.1 Docker Pipeline Runner
Generates 3D mesh (.glb) from input image using dockerized Hunyuan3D-2.1
Supports both shape-only and full textured pipeline
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


class Hunyuan3DRunner:
    def __init__(self, docker_image="your_username/hunyuan3d:latest", data_dir="~/hunyuan_data"):
        """
        Initialize the Hunyuan3D pipeline runner
        
        Args:
            docker_image: Docker image name (e.g., 'username/hunyuan3d:latest')
            data_dir: Directory to store input/output files
        """
        self.docker_image = docker_image
        self.data_dir = Path(data_dir).expanduser()
        
    def check_docker_image_exists(self):
        """Check if the Docker image exists locally or can be pulled"""
        print(f"Checking Docker image: {self.docker_image}")
        
        # Check if image exists locally
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                print(f"✓ Docker image found locally: {self.docker_image}")
                return True
            else:
                print(f"⚠ Image not found locally. Attempting to pull from Docker Hub...")
                return self.pull_docker_image()
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Error checking Docker image: {e}")
            return False
    
    def pull_docker_image(self):
        """Pull Docker image from Docker Hub"""
        try:
            print(f"Pulling {self.docker_image} (this may take a while)...")
            result = subprocess.run(
                ["docker", "pull", self.docker_image],
                capture_output=False,
                text=True,
                check=True
            )
            print(f"✓ Successfully pulled {self.docker_image}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to pull Docker image: {e}")
            print(f"Please check:")
            print(f"  1. Image name is correct: {self.docker_image}")
            print(f"  2. Image exists on Docker Hub")
            print(f"  3. You're logged in (run: docker login)")
            return False
        
    def setup_data_directory(self):
        """Create data directory if it doesn't exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Data directory ready: {self.data_dir}")
        
    def copy_input_image(self, input_image_path):
        """Copy input image to data directory"""
        input_path = Path(input_image_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
        target_path = self.data_dir / "input.png"
        
        # Copy file
        import shutil
        shutil.copy2(input_path, target_path)
        print(f"✓ Input image copied to: {target_path}")
        return target_path
    
    def check_docker_available(self):
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Docker available: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Docker not found. Please install Docker first.")
            return False
    
    def check_gpu_available(self):
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ NVIDIA GPU detected")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ Warning: No NVIDIA GPU detected. Pipeline may fail or run very slowly.")
            return False
    
    def generate_shape_only(self, verbose=True):
        """
        Run the Docker container to generate 3D shape only (no texture)
        
        Args:
            verbose: Print detailed output
            
        Returns:
            Path to output .glb file
        """
        output_file = self.data_dir / "output_shape.glb"
        
        # Python script to run inside container
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
        
        # Docker run command
        docker_cmd = [
            "docker", "run",
            "--gpus", "all",
            "--rm",  # Remove container after execution
            "-v", f"{self.data_dir.absolute()}:/data",
            "-w", "/workspace/Hunyuan3D-2.1",
            self.docker_image,
            "python3", "-c", python_script
        ]
        
        print(f"\n{'='*60}")
        print("Starting Hunyuan3D shape generation (no texture)...")
        print(f"{'='*60}\n")
        
        try:
            # Run docker command
            result = subprocess.run(
                docker_cmd,
                capture_output=not verbose,
                text=True,
                check=True
            )
            
            if verbose and result.stdout:
                print(result.stdout)
            
            # Check if output file was created
            if output_file.exists():
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS! Output saved to: {output_file}")
                print(f"{'='*60}\n")
                return output_file
            else:
                raise RuntimeError("Output file was not created")
                
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error running Docker container:")
            if e.stderr:
                print(e.stderr)
            raise
    
    def generate_full_textured(self, verbose=True, max_views=6, resolution=512):
        """
        Run the Docker container to generate full textured 3D model
        
        Args:
            verbose: Print detailed output
            max_views: Number of views for texture generation (default: 6)
            resolution: Texture resolution (default: 512)
            
        Returns:
            Path to output textured .glb file
        """
        output_file = self.data_dir / "output_textured.glb"
        temp_mesh = self.data_dir / "temp_mesh.obj"
        
        # Python script to run inside container
        # Note: paint_pipeline returns a file path string, not a mesh object
        python_script = f"""
import sys
import shutil
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

print('[1/2] Generating shape...')
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image='/data/input.png')[0]
mesh_untextured.export('/data/temp_mesh.obj')

print('[2/2] Generating texture (this takes a while)...')
paint_config = Hunyuan3DPaintConfig(max_num_view={max_views}, resolution={resolution})
paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
output_path = paint_pipeline('/data/temp_mesh.obj', image_path='/data/input.png')

print(f'Texture generation output: {{output_path}}')

# The pipeline outputs to a default location, copy it to our desired location
if isinstance(output_path, str):
    print(f'Copying from {{output_path}} to /data/output_textured.glb')
    shutil.copy2(output_path, '/data/output_textured.glb')
else:
    print('Exporting textured mesh...')
    output_path.export('/data/output_textured.glb')

print('✓ Full textured model generation complete!')
"""
        
        # Docker run command
        docker_cmd = [
            "docker", "run",
            "--gpus", "all",
            "--rm",  # Remove container after execution
            "-v", f"{self.data_dir.absolute()}:/data",
            "-w", "/workspace/Hunyuan3D-2.1",
            self.docker_image,
            "python3", "-c", python_script
        ]
        
        print(f"\n{'='*60}")
        print("Starting Hunyuan3D FULL pipeline (shape + texture)...")
        print(f"Config: {max_views} views, {resolution}px resolution")
        print("⚠ This will take significantly longer than shape-only!")
        print(f"{'='*60}\n")
        
        try:
            # Run docker command
            result = subprocess.run(
                docker_cmd,
                capture_output=not verbose,
                text=True,
                check=True
            )
            
            if verbose and result.stdout:
                print(result.stdout)
            
            # Check if output file was created
            if output_file.exists():
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS! Textured model saved to: {output_file}")
                if temp_mesh.exists():
                    print(f"ℹ Intermediate mesh saved to: {temp_mesh}")
                
                # Check for additional texture files
                texture_files = list(self.data_dir.glob("textured_mesh*"))
                if texture_files:
                    print(f"ℹ Additional texture files generated:")
                    for tf in texture_files:
                        print(f"  - {tf.name}")
                
                print(f"{'='*60}\n")
                return output_file
            else:
                raise RuntimeError("Output file was not created")
                
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error running Docker container:")
            if e.stderr:
                print(e.stderr)
            raise
    
    def run(self, input_image_path, mode="shape", verbose=True, max_views=6, resolution=512, fallback=True):
        """
        Complete pipeline: setup, copy image, generate mesh
        
        Args:
            input_image_path: Path to input image
            mode: 'shape' for shape only, 'textured' for full pipeline
            verbose: Print detailed output
            max_views: Number of views for texture (only for textured mode)
            resolution: Texture resolution (only for textured mode)
            fallback: If True, fall back to shape-only if textured fails
            
        Returns:
            Path to output .glb file
        """
        print("\n🚀 Hunyuan3D-2.1 Pipeline Starting...\n")
        print(f"Mode: {'SHAPE ONLY' if mode == 'shape' else 'FULL TEXTURED'}")
        
        # Pre-flight checks
        print("\nStep 1: Checking Docker...")
        if not self.check_docker_available():
            sys.exit(1)
        
        print("\nStep 2: Checking GPU...")
        self.check_gpu_available()
        
        print("\nStep 3: Checking Docker image...")
        if not self.check_docker_image_exists():
            sys.exit(1)
        
        print("\nStep 4: Setting up data directory...")
        self.setup_data_directory()
        
        print("\nStep 5: Copying input image...")
        self.copy_input_image(input_image_path)
        
        print(f"\nStep 6: Generating 3D {'shape' if mode == 'shape' else 'textured model'}...")
        
        if mode == "shape":
            output_file = self.generate_shape_only(verbose=verbose)
        elif mode == "textured":
            try:
                output_file = self.generate_full_textured(
                    verbose=verbose,
                    max_views=max_views,
                    resolution=resolution
                )
            except Exception as e:
                if fallback:
                    print(f"\n⚠️  Textured pipeline failed: {str(e)[:100]}")
                    print("🔄 Falling back to shape-only generation...")
                    output_file = self.generate_shape_only(verbose=verbose)
                    print("\n⚠️  Note: Texture generation failed. Your Docker image may need fixing.")
                    print("   Fix: Rebuild with updated torchvision (pip install --upgrade torchvision)")
                else:
                    raise
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'shape' or 'textured'")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D mesh from image using Hunyuan3D-2.1 Docker pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate shape only (fast)
  python hunyuan3d_runner.py image.png -d myusername/hunyuan3d:latest
  
  # Generate full textured model (slow but better quality)
  python hunyuan3d_runner.py image.png -d myusername/hunyuan3d:latest --textured
  
  # Textured with custom settings
  python hunyuan3d_runner.py image.png -d user/hunyuan3d:latest --textured --views 8 --resolution 1024
  
  # Custom output directory
  python hunyuan3d_runner.py image.png -d user/hunyuan3d:latest -o ./outputs
  
  # Test mode (check everything without generating)
  python hunyuan3d_runner.py image.png -d user/hunyuan3d:latest --test
        """
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "-d", "--docker-image",
        type=str,
        default="your_username/hunyuan3d:latest",
        help="Docker image name (default: your_username/hunyuan3d:latest)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="~/hunyuan_data",
        help="Output directory for generated files (default: ~/hunyuan_data)"
    )
    parser.add_argument(
        "--textured",
        action="store_true",
        help="Generate full textured model (slower but higher quality)"
    )
    parser.add_argument(
        "--views",
        type=int,
        default=6,
        help="Number of views for texture generation (default: 6, only for --textured)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Texture resolution (default: 512, only for --textured)"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Don't fall back to shape-only if textured mode fails"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run pre-flight checks only (don't generate mesh)"
    )
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = Hunyuan3DRunner(
        docker_image=args.docker_image,
        data_dir=args.output_dir
    )
    
    # Test mode - just run checks
    if args.test:
        print("\n🧪 Running pre-flight checks only...\n")
        print("Step 1: Checking Docker...")
        docker_ok = runner.check_docker_available()
        
        print("\nStep 2: Checking GPU...")
        runner.check_gpu_available()
        
        print("\nStep 3: Checking Docker image...")
        image_ok = runner.check_docker_image_exists()
        
        print("\nStep 4: Setting up data directory...")
        runner.setup_data_directory()
        
        print("\nStep 5: Checking input image...")
        input_path = Path(args.input_image)
        if input_path.exists():
            print(f"✓ Input image found: {input_path}")
        else:
            print(f"✗ Input image not found: {input_path}")
        
        print("\n" + "="*60)
        if docker_ok and image_ok:
            print("✅ All checks passed! Ready to generate.")
        else:
            print("❌ Some checks failed. Please fix the issues above.")
        print("="*60)
        return
    
    # Determine mode
    mode = "textured" if args.textured else "shape"
    
    # Normal mode - run full pipeline
    try:
        output_file = runner.run(
            input_image_path=args.input_image,
            mode=mode,
            verbose=not args.quiet,
            max_views=args.views,
            resolution=args.resolution,
            fallback=not args.no_fallback
        )
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📦 Output: {output_file}")
        
        if mode == "shape":
            print("\nℹ️  Tip: Add --textured flag for full textured model (takes longer)")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
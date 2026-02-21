#!/usr/bin/env python3
"""
Klein Fleet - Hugging Face Repository Setup
Creates and configures the HF model repo for Klein fleet captioning system.
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
REPO_ID = "msrcam/klein-fleet"
REPO_TYPE = "model"

# Model card content
MODEL_CARD = """---
license: apache-2.0
tags:
- flux
- image-captioning
- vision
- klein
- qwen
library_name: diffusers
pipeline_tag: image-to-text
---

# Klein Fleet - Pre-packaged Captioning Model Set

This repository contains a pre-configured model bundle for the Klein fleet captioning system,
designed for rapid deployment on Vast.ai GPU instances.

## What's Inside

This package contains three essential components:

1. **UNET**: `bigLove_klein1_fp8.safetensors` - Fine-tuned Flux UNET for Klein-style image generation
2. **CLIP**: `qwen3_8b_abliterated_v2-fp8mixed.safetensors` - Qwen3 8B text encoder (abliterated, FP8 mixed precision)
3. **VAE**: `flux2-vae.safetensors` - Flux 2 VAE for image encoding/decoding

All models are quantized to FP8 for optimal VRAM usage and performance.

## Purpose

Klein Fleet enables high-quality image captioning using:
- ComfyUI as the inference engine
- Qwen3VL 8B for vision understanding
- Custom-trained Flux models for Klein aesthetic
- Docker containerization for instant deployment

## Usage

### On Vast.ai

1. Launch instance with Docker image: `msrcam/klein-fleet:latest`
2. Models are automatically downloaded from this repo on first run
3. Access ComfyUI at `http://[instance-ip]:8188`
4. Use the Klein captioning workflow

### Manual Installation

```bash
# Download all models
huggingface-cli download msrcam/klein-fleet --local-dir ./models

# Or download individually
huggingface-cli download msrcam/klein-fleet bigLove_klein1_fp8.safetensors
huggingface-cli download msrcam/klein-fleet qwen3_8b_abliterated_v2-fp8mixed.safetensors
huggingface-cli download msrcam/klein-fleet flux2-vae.safetensors
```

## System Requirements

- GPU: 24GB VRAM minimum (RTX 3090, RTX 4090, or A5000)
- RAM: 32GB+ recommended
- Storage: 50GB for models + ComfyUI + dependencies

## Model Details

### bigLove_klein1_fp8.safetensors
- Type: FLUX UNET
- Precision: FP8
- Training: Fine-tuned on Klein aesthetic dataset
- Size: ~8GB

### qwen3_8b_abliterated_v2-fp8mixed.safetensors
- Type: Text Encoder (CLIP replacement)
- Base: Qwen3 8B
- Precision: FP8 Mixed
- Modifications: Abliterated for unrestricted output
- Size: ~8GB

### flux2-vae.safetensors
- Type: VAE
- Compatibility: FLUX 2.0
- Size: ~300MB

## Docker Image

Corresponding Docker image: `msrcam/klein-fleet:latest`

The Docker image includes:
- ComfyUI with all required custom nodes
- Qwen3VL-Toolkit for captioning workflows
- Python environment with all dependencies
- Automatic model download on startup

## License

Apache 2.0 - See LICENSE file for details

## Credits

- FLUX models: Black Forest Labs
- Qwen3: Alibaba Cloud
- Klein training: Custom dataset
- Integration: ComfyUI-Qwen3VL-Toolkit

## Support

Issues and questions: [GitHub Issues](https://github.com/msrcam/ComfyUI-Qwen3VL-Toolkit/issues)
"""


def main():
    """Create and configure the Klein Fleet HF repository."""

    print("=" * 80)
    print("Klein Fleet - Hugging Face Repository Setup")
    print("=" * 80)
    print()

    # Initialize HF API
    api = HfApi()

    # Check if user is logged in
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"✓ Logged in as: {username}")
    except Exception as e:
        print("✗ Not logged in to Hugging Face!")
        print()
        print("Please login first:")
        print("  huggingface-cli login")
        print()
        return

    print()

    # Create repository
    print(f"Creating repository: {REPO_ID}")
    try:
        repo_url = create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            exist_ok=True,
            private=False
        )
        print(f"✓ Repository created/verified: {repo_url}")
    except Exception as e:
        print(f"✗ Error creating repository: {e}")
        return

    print()

    # Upload model card
    print("Uploading README.md (model card)...")
    try:
        # Save model card to temp file
        with open("README_temp.md", "w", encoding="utf-8") as f:
            f.write(MODEL_CARD)

        # Upload to repo
        api.upload_file(
            path_or_fileobj="README_temp.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

        # Clean up temp file
        os.remove("README_temp.md")

        print("✓ Model card uploaded successfully")
    except Exception as e:
        print(f"✗ Error uploading model card: {e}")

    print()
    print("=" * 80)
    print("NEXT STEPS: Upload Your Model Files")
    print("=" * 80)
    print()
    print("You need to upload these three files to the repository:")
    print()
    print("1. bigLove_klein1_fp8.safetensors (UNET)")
    print("2. qwen3_8b_abliterated_v2-fp8mixed.safetensors (CLIP)")
    print("3. flux2-vae.safetensors (VAE)")
    print()
    print("=" * 80)
    print("OPTION 1: Upload Files Individually")
    print("=" * 80)
    print()
    print("# UNET (Klein fine-tuned Flux)")
    print(f"huggingface-cli upload {REPO_ID} \\")
    print("  /path/to/bigLove_klein1_fp8.safetensors \\")
    print("  bigLove_klein1_fp8.safetensors")
    print()
    print("# CLIP (Qwen3 8B abliterated)")
    print(f"huggingface-cli upload {REPO_ID} \\")
    print("  /path/to/qwen3_8b_abliterated_v2-fp8mixed.safetensors \\")
    print("  qwen3_8b_abliterated_v2-fp8mixed.safetensors")
    print()
    print("# VAE (Flux 2)")
    print(f"huggingface-cli upload {REPO_ID} \\")
    print("  /path/to/flux2-vae.safetensors \\")
    print("  flux2-vae.safetensors")
    print()
    print("=" * 80)
    print("OPTION 2: Upload All Files from Directory")
    print("=" * 80)
    print()
    print("If all three files are in a single directory:")
    print()
    print(f"huggingface-cli upload {REPO_ID} /path/to/model/directory --include='*.safetensors'")
    print()
    print("=" * 80)
    print("OPTION 3: Upload Using Python")
    print("=" * 80)
    print()
    print("from huggingface_hub import HfApi")
    print("api = HfApi()")
    print()
    print("# Upload each file")
    print(f"api.upload_file(")
    print(f"    path_or_fileobj='/path/to/bigLove_klein1_fp8.safetensors',")
    print(f"    path_in_repo='bigLove_klein1_fp8.safetensors',")
    print(f"    repo_id='{REPO_ID}',")
    print(f"    repo_type='model'")
    print(f")")
    print()
    print("# Repeat for other two files...")
    print()
    print("=" * 80)
    print("VERIFY UPLOAD")
    print("=" * 80)
    print()
    print(f"Check your repository at: https://huggingface.co/{REPO_ID}")
    print()
    print("You should see all three .safetensors files in the 'Files' tab.")
    print()
    print("Total size should be approximately 16-17 GB")
    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)
    print()
    print("Once files are uploaded, you can:")
    print("1. Build the Docker image with setup_docker.sh")
    print("2. Deploy to Vast.ai instances")
    print("3. Models will auto-download from this repo on first run")
    print()


if __name__ == "__main__":
    main()

#!/bin/bash
#
# Klein Fleet - Docker Build and Push Script
# Builds the Docker image and pushes it to Docker Hub
# Run this from the ComfyUI-Qwen3VL-Toolkit directory
#

set -e  # Exit on any error

# Configuration
IMAGE_NAME="msrcam/klein-fleet"
TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "================================================================================"
echo "Klein Fleet - Docker Build and Push"
echo "================================================================================"
echo ""
echo "Image: ${FULL_IMAGE}"
echo "Directory: $(pwd)"
echo ""

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found in current directory!"
    echo "Please run this script from the ComfyUI-Qwen3VL-Toolkit directory."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Check if logged in to Docker Hub
if ! docker info | grep -q "Username"; then
    echo "WARNING: You may not be logged in to Docker Hub"
    echo ""
    echo "Please login first:"
    echo "  docker login"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to abort..."
fi

echo "================================================================================"
echo "STEP 1: Building Docker Image"
echo "================================================================================"
echo ""
echo "This may take 10-20 minutes depending on your internet connection..."
echo "The image will include:"
echo "  - ComfyUI + custom nodes"
echo "  - Python dependencies"
echo "  - Qwen3VL toolkit"
echo "  - Auto-download scripts for models from HuggingFace"
echo ""

# Build the image
docker build -t "${FULL_IMAGE}" .

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Docker build failed!"
    exit 1
fi

echo ""
echo "✓ Build successful!"
echo ""

# Get image size
IMAGE_SIZE=$(docker images "${IMAGE_NAME}" --format "{{.Size}}" | head -1)
echo "Image size: ${IMAGE_SIZE}"
echo ""

echo "================================================================================"
echo "STEP 2: Tagging Image"
echo "================================================================================"
echo ""

# Tag with 'latest' (already done in build, but being explicit)
docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:latest"

# Optional: Tag with date for versioning
DATE_TAG=$(date +%Y%m%d)
docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:${DATE_TAG}"

echo "✓ Tagged as:"
echo "  - ${IMAGE_NAME}:latest"
echo "  - ${IMAGE_NAME}:${DATE_TAG}"
echo ""

echo "================================================================================"
echo "STEP 3: Pushing to Docker Hub"
echo "================================================================================"
echo ""
echo "Pushing ${FULL_IMAGE}..."
echo "This may take 10-30 minutes depending on upload speed..."
echo ""

# Push latest tag
docker push "${IMAGE_NAME}:latest"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Docker push failed!"
    echo ""
    echo "Common issues:"
    echo "1. Not logged in - run: docker login"
    echo "2. No permission for repo 'msrcam/klein-fleet'"
    echo "3. Network issues - check your connection"
    exit 1
fi

# Push date tag
echo ""
echo "Pushing ${IMAGE_NAME}:${DATE_TAG}..."
docker push "${IMAGE_NAME}:${DATE_TAG}"

echo ""
echo "✓ Push successful!"
echo ""

echo "================================================================================"
echo "DONE! Image Published to Docker Hub"
echo "================================================================================"
echo ""
echo "Your image is now available at:"
echo "  https://hub.docker.com/r/${IMAGE_NAME}"
echo ""
echo "Available tags:"
echo "  - latest"
echo "  - ${DATE_TAG}"
echo ""
echo "================================================================================"
echo "DEPLOYMENT INSTRUCTIONS"
echo "================================================================================"
echo ""
echo "For Vast.ai:"
echo "1. Go to https://cloud.vast.ai/create/"
echo "2. Search for instances (RTX 3090/4090/A5000 recommended)"
echo "3. In 'Docker Image' field, enter: ${IMAGE_NAME}:latest"
echo "4. Set disk space: 60GB minimum"
echo "5. Launch instance"
echo "6. Connect via SSH or web interface"
echo "7. ComfyUI will be available at: http://[instance-ip]:8188"
echo ""
echo "For local testing:"
echo "docker run -it --gpus all -p 8188:8188 ${FULL_IMAGE}"
echo ""
echo "================================================================================"
echo "MODEL DOWNLOAD"
echo "================================================================================"
echo ""
echo "On first run, the container will automatically download models from:"
echo "  https://huggingface.co/msrcam/klein-fleet"
echo ""
echo "Make sure you've uploaded your models to HuggingFace first!"
echo "Run: python setup_hf_repo.py (if you haven't already)"
echo ""
echo "Models that will be downloaded:"
echo "  - bigLove_klein1_fp8.safetensors (~8GB)"
echo "  - qwen3_8b_abliterated_v2-fp8mixed.safetensors (~8GB)"
echo "  - flux2-vae.safetensors (~300MB)"
echo ""
echo "Total download: ~16-17 GB on first launch"
echo "Subsequent launches will use cached models (no re-download)"
echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"
echo ""
echo "Local images:"
docker images | grep "${IMAGE_NAME}" || echo "No images found!"
echo ""
echo "To test locally:"
echo "  docker run -it --rm --gpus all -p 8188:8188 ${FULL_IMAGE}"
echo ""
echo "To view logs:"
echo "  docker logs [container-id]"
echo ""
echo "To enter running container:"
echo "  docker exec -it [container-id] /bin/bash"
echo ""
echo "================================================================================"
echo "SUCCESS! Klein Fleet is ready for deployment."
echo "================================================================================"
echo ""

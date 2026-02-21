# Klein Diffusion Fleet — Vast.ai Container
# Base: CUDA 12.1 + Ubuntu 22.04
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Hugging Face CLI
RUN pip install "huggingface_hub[cli]"

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Clone ComfyUI
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI

# Install ComfyUI requirements
WORKDIR /workspace/ComfyUI
RUN pip install -r requirements.txt

# Create model directories
RUN mkdir -p /workspace/ComfyUI/models/unet \
    /workspace/ComfyUI/models/clip \
    /workspace/ComfyUI/models/vae \
    /workspace/models_cache

# Copy custom nodes toolkit
COPY . /workspace/ComfyUI/custom_nodes/ComfyUI-Qwen3VL-Toolkit/

# Install toolkit requirements
RUN pip install requests imagehash opencv-python-headless Pillow numpy

# Set environment variables for Ollama
ENV OLLAMA_NUM_PARALLEL=4
ENV OLLAMA_HOST=0.0.0.0:11434

# Create entrypoint script
RUN cat > /workspace/start.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Klein Fleet Container Starting ==="

# Download models from Hugging Face if not present
if [ ! -d "/workspace/models_cache/unet" ]; then
    echo "Downloading models from msrcam/klein-fleet..."
    huggingface-cli download msrcam/klein-fleet --local-dir /workspace/models_cache
else
    echo "Models already present, skipping download."
fi

# Symlink or copy models to ComfyUI directories
echo "Setting up model symlinks..."
if [ -d "/workspace/models_cache/unet" ]; then
    ln -sf /workspace/models_cache/unet/* /workspace/ComfyUI/models/unet/ 2>/dev/null || true
fi
if [ -d "/workspace/models_cache/clip" ]; then
    ln -sf /workspace/models_cache/clip/* /workspace/ComfyUI/models/clip/ 2>/dev/null || true
fi
if [ -d "/workspace/models_cache/vae" ]; then
    ln -sf /workspace/models_cache/vae/* /workspace/ComfyUI/models/vae/ 2>/dev/null || true
fi

# Start Ollama in background
echo "Starting Ollama server..."
nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 5

# Pull VLM model
echo "Pulling Qwen2.5-VL model..."
ollama pull huihui_ai/qwen2.5-vl-abliterated:32b

echo "=== Starting ComfyUI on port 18188 ==="
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 18188
EOF

# Make start script executable
RUN chmod +x /workspace/start.sh

# Expose ports
EXPOSE 18188 11434

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/workspace/start.sh"]

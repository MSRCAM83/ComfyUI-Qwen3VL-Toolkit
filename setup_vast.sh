#!/bin/bash
# ============================================================
# ComfyUI-Qwen3VL-Toolkit — Vast.ai Setup Script
# ============================================================
# Run this on a fresh Vast.ai instance to set up everything:
#   1. System deps (ffmpeg)
#   2. ComfyUI (if not present)
#   3. Ollama + Qwen 3 VL models
#   4. This toolkit as a custom node
#
# Usage: bash setup_vast.sh
# ============================================================

set -e

echo "============================================================"
echo "  ComfyUI-Qwen3VL-Toolkit — Vast.ai Setup"
echo "============================================================"

# --- 1. System packages ---
echo ""
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg libgl1-mesa-glx libglib2.0-0 curl git wget > /dev/null

# --- 2. Install ComfyUI (if not present) ---
COMFY_DIR="/workspace/ComfyUI"
if [ ! -d "$COMFY_DIR" ]; then
    echo ""
    echo "[2/5] Installing ComfyUI..."
    cd /workspace
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    pip install -r requirements.txt --quiet
    # Install ComfyUI Manager for easy node management
    cd custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd /workspace
else
    echo ""
    echo "[2/5] ComfyUI already installed at $COMFY_DIR"
fi

# --- 3. Install this toolkit as custom node ---
echo ""
echo "[3/5] Installing Qwen3VL-Toolkit custom nodes..."
TOOLKIT_DIR="$COMFY_DIR/custom_nodes/ComfyUI-Qwen3VL-Toolkit"
if [ -d "$TOOLKIT_DIR" ]; then
    echo "  Updating existing installation..."
    rm -rf "$TOOLKIT_DIR"
fi

# Copy or symlink (copy is safer on Vast.ai)
if [ -d "/workspace/ComfyUI-Qwen3VL-Toolkit" ]; then
    cp -r /workspace/ComfyUI-Qwen3VL-Toolkit "$TOOLKIT_DIR"
else
    # If run from the toolkit directory itself
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cp -r "$SCRIPT_DIR" "$TOOLKIT_DIR"
fi

# Install Python dependencies
pip install -r "$TOOLKIT_DIR/requirements.txt" --quiet
echo "  Toolkit installed at $TOOLKIT_DIR"

# --- 4. Install & start Ollama ---
echo ""
echo "[4/5] Setting up Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  Ollama already installed"
fi

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "  Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# --- 5. Pull Qwen 3 VL models ---
echo ""
echo "[5/5] Pulling Qwen 3 VL models..."
echo "  This may take 5-15 minutes depending on your connection..."
echo ""

# Pull 8B (fast model for filtering + captions)
echo "  Pulling qwen3-vl:8b (~5GB)..."
ollama pull qwen3-vl:8b

echo ""
echo "  Pulling qwen3-vl:37b (~22GB)..."
echo "  (This is the large model — skip with Ctrl+C if you only want 8B)"
ollama pull qwen3-vl:37b

# --- Done ---
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  ComfyUI:  $COMFY_DIR"
echo "  Toolkit:  $TOOLKIT_DIR"
echo "  Ollama:   http://127.0.0.1:11434"
echo "  Models:   qwen3-vl:8b, qwen3-vl:37b"
echo ""
echo "  To start ComfyUI:"
echo "    cd $COMFY_DIR && python main.py --listen 0.0.0.0 --port 8188"
echo ""
echo "  Or with share link (if no direct port access):"
echo "    cd $COMFY_DIR && python main.py --listen 0.0.0.0 --port 8188"
echo ""
echo "  Nodes will appear under 'Qwen3VL' category in the node menu."
echo "============================================================"

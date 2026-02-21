"""
ComfyUI-Qwen3VL-Toolkit
========================
18 nodes for the ultimate LoRA dataset preparation pipeline.
Uses Qwen 3 VL vision models via Ollama or OpenAI-compatible API.

Nodes:
  INPUT:    LoadImages, LoadVideoFrames, ServerImage, FolderPreview
  VLM:      Filter, Analyze, Caption, Detect, CustomQuery
  PROCESS:  Dedup, SmartCrop, AutoCorrect, QualityScore, NudeScore,
            DenseResample, Resize, MetadataRouter
  OUTPUT:   SaveDataset
"""

import os
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .klein_fleet_node import QVL_KleinFleet

# Register Klein Fleet node
NODE_CLASS_MAPPINGS["QVL_KleinFleet"] = QVL_KleinFleet
NODE_DISPLAY_NAME_MAPPINGS["QVL_KleinFleet"] = "Klein Fleet (Multi-Instance)"

# Load server-side API routes (file browser endpoints)
try:
    from . import server_routes
except Exception as e:
    print(f"[QVL] Warning: Failed to load server routes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Tell ComfyUI where our frontend JavaScript lives
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web", "js")

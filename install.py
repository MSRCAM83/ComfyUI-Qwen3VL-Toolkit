"""
Auto-install dependencies for ComfyUI-Qwen3VL-Toolkit.
This runs automatically when ComfyUI loads the custom node package.
"""

import subprocess
import sys

PACKAGES = [
    "requests",
    "imagehash",
    "opencv-python-headless",
    "Pillow",
]

for pkg in PACKAGES:
    try:
        __import__(pkg.replace("-", "_").split("[")[0])
    except ImportError:
        print(f"[QVL-Toolkit] Installing {pkg}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--quiet",
        ])
        print(f"[QVL-Toolkit] Installed {pkg}")

print("[QVL-Toolkit] All dependencies ready.")

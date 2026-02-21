"""
Image processing utilities for ComfyUI-Qwen3VL-Toolkit.
Tensor conversion, quality metrics, dedup, corrections, cropping, resizing.
"""

import os
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


# ============================================================
# TENSOR CONVERSION
# ============================================================

def pil_to_tensor(pil_img):
    """Convert PIL Image to ComfyUI tensor [1, H, W, 3] float32 0-1."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(tensor, index=0):
    """Convert ComfyUI batch tensor to PIL Image at given index."""
    img = tensor[index] if tensor.dim() == 4 else tensor
    arr = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def batch_to_pil_list(tensor):
    """Convert entire batch tensor to list of PIL Images."""
    return [tensor_to_pil(tensor, i) for i in range(tensor.shape[0])]


def pil_list_to_batch(pil_images):
    """Convert list of PIL Images to batch tensor."""
    if not pil_images:
        return empty_batch()
    tensors = [pil_to_tensor(img) for img in pil_images]
    return torch.cat(tensors, dim=0)


def empty_batch():
    """Return 1x1 black tensor as empty result sentinel."""
    return torch.zeros(1, 1, 1, 3, dtype=torch.float32)


def is_empty_batch(tensor):
    """Check if tensor is the empty sentinel."""
    return (tensor.shape[0] == 1 and tensor.shape[1] == 1
            and tensor.shape[2] == 1)


# ============================================================
# FILE LOADING
# ============================================================

def load_images_from_folder(folder, min_size=128, max_images=0, sort_by="name"):
    """Load images from folder. Returns list of (PIL, filename) tuples."""
    if not os.path.isdir(folder):
        return []

    files = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            files.append(f)

    if sort_by == "name":
        files.sort()
    elif sort_by == "date":
        files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    elif sort_by == "size":
        files.sort(key=lambda f: os.path.getsize(os.path.join(folder, f)),
                   reverse=True)

    if max_images > 0:
        files = files[:max_images]

    results = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            img.load()  # Force load to catch corrupt files
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if min(w, h) >= min_size:
                results.append((img, f))
        except Exception:
            continue

    return results


# ============================================================
# QUALITY METRICS (local, no VLM)
# ============================================================

def compute_quality(pil_img, min_size=256):
    """Compute local quality metrics. Returns (score: 0-100, details: dict)."""
    import cv2

    arr = np.array(pil_img)
    if len(arr.shape) == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Sharpness — Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    sharpness_score = min(sharpness / 500.0, 1.0) * 100

    # Brightness — penalize extremes
    brightness = float(gray.mean())
    if brightness < 30:
        bright_score = brightness / 30.0 * 50
    elif brightness > 230:
        bright_score = (255 - brightness) / 25.0 * 50
    else:
        bright_score = 100

    # Contrast — std deviation
    contrast = float(gray.std())
    contrast_score = min(contrast / 60.0, 1.0) * 100

    # Size check
    w, h = pil_img.size
    min_dim = min(w, h)
    size_score = min(min_dim / min_size, 1.0) * 100

    # Combined weighted score
    score = (sharpness_score * 0.35 + bright_score * 0.20
             + contrast_score * 0.20 + size_score * 0.25)

    details = {
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "size": f"{w}x{h}",
        "min_dim": min_dim,
        "subscores": {
            "sharpness": round(sharpness_score, 1),
            "brightness": round(bright_score, 1),
            "contrast": round(contrast_score, 1),
            "size": round(size_score, 1),
        },
    }
    return round(score, 1), details


# ============================================================
# PERCEPTUAL DEDUPLICATION
# ============================================================

def deduplicate(pil_images, threshold=6, hash_size=16):
    """Remove near-duplicate images by perceptual hash.
    Returns (kept_indices, removed_indices)."""
    import imagehash

    hashes = []
    kept = []
    removed = []

    for i, img in enumerate(pil_images):
        h = imagehash.phash(img, hash_size=hash_size)
        is_dupe = False
        for eh in hashes:
            if abs(h - eh) <= threshold:
                is_dupe = True
                break
        if is_dupe:
            removed.append(i)
        else:
            hashes.append(h)
            kept.append(i)

    return kept, removed


# ============================================================
# IMAGE CORRECTIONS
# ============================================================

def apply_corrections(pil_img, corrections):
    """Apply VLM-suggested corrections. corrections is a dict from Analyze node."""
    img = pil_img.copy()

    # Brightness
    b = float(corrections.get("brightness", 1.0))
    if abs(b - 1.0) > 0.01:
        b = max(0.5, min(1.5, b))
        img = ImageEnhance.Brightness(img).enhance(b)

    # Contrast
    c = float(corrections.get("contrast", 1.0))
    if abs(c - 1.0) > 0.01:
        c = max(0.5, min(1.5, c))
        img = ImageEnhance.Contrast(img).enhance(c)

    # Sharpness
    s = float(corrections.get("sharpness", 1.0))
    if abs(s - 1.0) > 0.01:
        if s > 1.0:
            percent = int(min(s, 2.0) - 1.0) * 150
            percent = max(10, min(300, percent))
            img = img.filter(ImageFilter.UnsharpMask(
                radius=2, percent=percent, threshold=3))
        else:
            s = max(0.3, s)
            img = ImageEnhance.Sharpness(img).enhance(s)

    # Edge cropping
    w, h = img.size
    cl = int(w * float(corrections.get("crop_left_pct", 0)) / 100)
    cr = int(w * float(corrections.get("crop_right_pct", 0)) / 100)
    ct = int(h * float(corrections.get("crop_top_pct", 0)) / 100)
    cb = int(h * float(corrections.get("crop_bottom_pct", 0)) / 100)

    if cl + cr + ct + cb > 0:
        new_w = w - cl - cr
        new_h = h - ct - cb
        if new_w > 64 and new_h > 64:
            img = img.crop((cl, ct, w - cr, h - cb))

    return img


def crop_to_bbox(pil_img, bbox_pct, padding_pct=5, square=True):
    """Crop image to percentage-based bounding box.
    bbox_pct = [x1%, y1%, x2%, y2%]"""
    w, h = pil_img.size

    x1 = int(w * float(bbox_pct[0]) / 100)
    y1 = int(h * float(bbox_pct[1]) / 100)
    x2 = int(w * float(bbox_pct[2]) / 100)
    y2 = int(h * float(bbox_pct[3]) / 100)

    # Validate
    if x2 <= x1 or y2 <= y1:
        return pil_img

    # Add padding
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding_pct / 100)
    pad_y = int(bh * padding_pct / 100)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if square:
        cw = x2 - x1
        ch = y2 - y1
        if cw > ch:
            diff = cw - ch
            y1 = max(0, y1 - diff // 2)
            y2 = min(h, y1 + cw)
            if y2 - y1 < cw:
                y1 = max(0, y2 - cw)
        elif ch > cw:
            diff = ch - cw
            x1 = max(0, x1 - diff // 2)
            x2 = min(w, x1 + ch)
            if x2 - x1 < ch:
                x1 = max(0, x2 - ch)

    return pil_img.crop((x1, y1, x2, y2))


# ============================================================
# RESIZE
# ============================================================

def resize_image(pil_img, target_size=1024, mode="resize"):
    """Resize image to target. Modes: resize, pad, crop_center, longest_side."""
    if mode == "resize":
        return pil_img.resize((target_size, target_size), Image.LANCZOS)

    elif mode == "pad":
        ratio = target_size / max(pil_img.size)
        new_w = int(pil_img.width * ratio)
        new_h = int(pil_img.height * ratio)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        x = (target_size - new_w) // 2
        y = (target_size - new_h) // 2
        canvas.paste(resized, (x, y))
        return canvas

    elif mode == "crop_center":
        ratio = target_size / min(pil_img.size)
        new_w = int(pil_img.width * ratio)
        new_h = int(pil_img.height * ratio)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        return resized.crop((left, top, left + target_size, top + target_size))

    elif mode == "longest_side":
        ratio = target_size / max(pil_img.size)
        new_w = int(pil_img.width * ratio)
        new_h = int(pil_img.height * ratio)
        # Round to nearest 64 for stable diffusion compatibility
        new_w = max(64, (new_w // 64) * 64)
        new_h = max(64, (new_h // 64) * 64)
        return pil_img.resize((new_w, new_h), Image.LANCZOS)

    return pil_img

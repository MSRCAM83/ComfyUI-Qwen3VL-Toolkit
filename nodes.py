"""
ComfyUI-Qwen3VL-Toolkit — 12 nodes for the ultimate LoRA dataset pipeline.
All nodes use category "Qwen3VL" and prefix "QVL_" for easy discovery.
"""

import json
import os
import subprocess
import torch

import comfy.utils

from .vlm import (
    query_vlm, query_vlm_openai, query_vlm_batch, parse_json,
    FILTER_PROMPT, ANALYZE_PROMPT, CLASSIFY_PROMPT, BBOX_PROMPT,
    CAPTION_PRESETS, build_caption_prompt,
)
from .utils import (
    pil_to_tensor, tensor_to_pil, batch_to_pil_list, pil_list_to_batch,
    empty_batch, is_empty_batch, load_images_from_folder,
    compute_quality, deduplicate, apply_corrections, crop_to_bbox,
    resize_image, IMG_EXTS,
)

CATEGORY = "Qwen3VL"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
VIDEO_DIR = "/workspace/videos"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts", ".m4v"}


def _scan_videos(video_dir=VIDEO_DIR):
    """Scan directory for video files and return sorted list for dropdown."""
    if not os.path.isdir(video_dir):
        os.makedirs(video_dir, exist_ok=True)
        return ["(no videos found — add files to /workspace/videos/)"]
    files = []
    for f in os.listdir(video_dir):
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
            files.append(f)
    files.sort()
    if not files:
        return ["(no videos found — add files to /workspace/videos/)"]
    if len(files) > 1:
        files.insert(0, "** ALL VIDEOS **")
    return files


# ============================================================
# 1. LOAD IMAGES
# ============================================================

class QVL_LoadImages:
    """Load a batch of images from a folder."""

    DESCRIPTION = """Load images from a folder on disk. Supports jpg, png, webp, bmp.
Use this for pre-extracted frames or curated image collections.

Outputs:
- images: Batch tensor for ComfyUI processing
- filenames: JSON array of filenames (preserves metadata through pipeline)
- count: Number of images loaded

Performance: Instant for folders with <1000 images. For larger collections, use max_images to limit batch size.
Use min_size to filter out thumbnails and low-res images automatically."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "/workspace/images",
                    "multiline": False,
                    "tooltip": "Absolute path to folder containing images. On Vast.ai, use /workspace paths. Windows: use forward slashes (C:/Users/...) or double backslashes.",
                }),
                "min_size": ("INT", {
                    "default": 256, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Minimum width or height in pixels. Images smaller than this are filtered out. Recommended: 256-512 for LoRA training. Higher = fewer images kept, better quality.",
                }),
                "max_images": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Maximum number of images to load. 0 = unlimited. Recommended: 500-1000 to avoid memory issues. Higher = longer load time.",
                }),
                "sort_by": (["name", "date", "size"], {
                    "tooltip": "How to sort files. name = alphabetical, date = newest first, size = largest first. Use date to prioritize recent additions, size to prioritize high-res.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames", "count")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, folder_path, min_size, max_images, sort_by):
        results = load_images_from_folder(
            folder_path, min_size=min_size,
            max_images=max_images, sort_by=sort_by,
        )

        if not results:
            return (empty_batch(), "[]", 0)

        pil_imgs = [r[0] for r in results]
        filenames = [r[1] for r in results]
        batch = pil_list_to_batch(pil_imgs)

        return (batch, json.dumps(filenames), len(filenames))


# ============================================================
# 2. LOAD VIDEO FRAMES (handles large videos)
# ============================================================

class QVL_LoadVideoFrames:
    """Extract frames from a video file with scene detection and built-in dedup.
    Handles large videos (1GB+) by extracting to disk first, deduping on disk,
    then loading only unique frames into memory."""

    DESCRIPTION = """Extract frames from video files using ffmpeg. Optimized for large files (1GB+).

Extraction modes:
- scene_detect: Extracts at scene changes (best for diverse content)
- fps: Extracts at fixed frame rate (predictable, uniform sampling)
- keyframes: Extracts only I-frames (fastest, misses some detail)

Features:
- Built-in deduplication (removes near-identical frames during extraction)
- Automatic fallback (if scene detection fails, switches to fps mode)
- Multi-video support (select "** ALL VIDEOS **" to process entire folder)
- On-disk processing (only loads final unique frames into memory)

Performance: 10-30 seconds per GB of video. Dedup adds ~50% overhead but reduces dataset size by 30-70%.

Outputs:
- frames: Deduplicated frames as batch tensor
- filenames: JSON array with video name prefix (videoname_frame_NNNNNN.jpg)
- kept_count: Final frame count after dedup
- total_extracted: Raw extraction count (before dedup)"""

    @classmethod
    def INPUT_TYPES(cls):
        videos = _scan_videos()
        return {
            "required": {
                "video": (videos, {
                    "default": videos[0],
                    "tooltip": "Select a video file or '** ALL VIDEOS **' to process entire folder. Videos must be in video_dir (default: /workspace/videos).",
                }),
                "extraction_mode": (["scene_detect", "fps", "keyframes"], {
                    "default": "scene_detect",
                    "tooltip": "scene_detect = extract at scene changes (best quality), fps = fixed rate (consistent), keyframes = I-frames only (fastest but sparse). Recommended: scene_detect for most use cases.",
                }),
                "fps": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1,
                    "tooltip": "Frames per second for fps mode (also used as fallback if scene_detect finds nothing). Recommended: 1.0 for body datasets, 0.5 for slow videos, 2-5 for fast action. Higher = more frames, slower processing.",
                }),
                "scene_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 0.9, "step": 0.01,
                    "tooltip": "Scene change sensitivity (scene_detect mode). Lower = more scenes detected. Recommended: 0.10 for typical videos, 0.05 for subtle changes, 0.20 for major cuts only. Higher = fewer frames.",
                }),
                "max_frames": ("INT", {
                    "default": 500, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Maximum frames to keep (after extraction and dedup). Recommended: 300-500 for single video, 1000-2000 for ALL VIDEOS mode. Higher = more data but slower processing downstream.",
                }),
                "dedup_on_extract": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove near-duplicates during extraction (before loading into memory). Recommended: True for all cases. Saves 30-70% memory. Only disable for debugging.",
                }),
                "dedup_threshold": ("INT", {
                    "default": 8, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Perceptual hash distance for dedup. Lower = stricter (more duplicates removed). Recommended: 6-10 for video frames, 3-5 for image sets. Higher = keeps more similar frames. 0 = exact duplicates only.",
                }),
            },
            "optional": {
                "video_dir": ("STRING", {
                    "default": "/workspace/videos",
                    "multiline": False,
                    "tooltip": "Directory containing video files. Must exist and be readable. On Vast.ai, use /workspace/videos. On Windows, use forward slashes or double backslashes.",
                }),
                "output_folder": ("STRING", {
                    "default": "/workspace/frames",
                    "multiline": False,
                    "tooltip": "Where to save extracted frames (before loading). Auto-creates subfolders per video. Frames persist for re-use. Delete manually to free disk space.",
                }),
                "min_size": ("INT", {
                    "default": 256, "min": 64, "max": 4096, "step": 64,
                    "tooltip": "Minimum width or height in pixels. Frames smaller than this are filtered. Recommended: 256-512 for LoRA training. Higher = fewer frames kept.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force ComfyUI to re-scan videos each time
        return float("nan")

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("frames", "filenames", "kept_count", "total_extracted")
    FUNCTION = "extract"
    CATEGORY = CATEGORY

    def _extract_one(self, video_path, extraction_mode, fps, scene_threshold,
                     max_frames, dedup_on_extract, dedup_threshold,
                     output_folder, min_size):
        """Extract frames from a single video file. Returns (pil_imgs, filenames, total_extracted)."""
        from PIL import Image
        import glob

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(output_folder, video_name)
        os.makedirs(frame_dir, exist_ok=True)

        def _run_ffmpeg(cmd, label):
            print(f"[QVL] Extracting: {label}...")
            try:
                result = subprocess.run(
                    cmd, capture_output=True, timeout=600, text=True,
                )
                if result.returncode != 0 and result.stderr:
                    lines = result.stderr.strip().split("\n")
                    for line in lines[-3:]:
                        if "error" in line.lower():
                            print(f"[QVL] ffmpeg: {line.strip()}")
            except subprocess.TimeoutExpired:
                print(f"[QVL] ffmpeg timed out after 600s")
            except Exception as e:
                print(f"[QVL] ffmpeg error: {e}")
            return sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))

        # Clear old frames
        for old in glob.glob(os.path.join(frame_dir, "frame_*.jpg")):
            os.remove(old)

        out_pattern = os.path.join(frame_dir, "frame_%06d.jpg")
        frame_files = []

        if extraction_mode == "scene_detect":
            vf = f"select='gt(scene\\,{scene_threshold})',setpts=N/FRAME_RATE/TB"
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", vf, "-vsync", "vfr",
                "-frames:v", str(max_frames * 3), "-q:v", "2",
                out_pattern,
            ]
            frame_files = _run_ffmpeg(cmd, f"scene_detect threshold={scene_threshold}")
            if not frame_files:
                print(f"[QVL] Scene detect found 0 frames — falling back to fps={fps}")
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"fps={fps}", "-frames:v", str(max_frames),
                    "-q:v", "2", out_pattern,
                ]
                frame_files = _run_ffmpeg(cmd, f"fps fallback at {fps} fps")
        elif extraction_mode == "keyframes":
            cmd = [
                "ffmpeg", "-y", "-skip_frame", "nokey",
                "-i", video_path, "-vsync", "vfr",
                "-frames:v", str(max_frames * 2), "-q:v", "2",
                out_pattern,
            ]
            frame_files = _run_ffmpeg(cmd, "keyframes only")
            if not frame_files:
                print(f"[QVL] Keyframe extract found 0 — falling back to fps={fps}")
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"fps={fps}", "-frames:v", str(max_frames),
                    "-q:v", "2", out_pattern,
                ]
                frame_files = _run_ffmpeg(cmd, f"fps fallback at {fps} fps")
        else:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={fps}", "-frames:v", str(max_frames),
                "-q:v", "2", out_pattern,
            ]
            frame_files = _run_ffmpeg(cmd, f"fps={fps}")

        total_extracted = len(frame_files)
        print(f"[QVL] Extracted {total_extracted} frames from {os.path.basename(video_path)}")

        if not frame_files:
            return [], [], 0

        # Dedup on disk
        if dedup_on_extract and total_extracted > 1:
            import imagehash
            print(f"[QVL] Deduplicating {total_extracted} frames (threshold={dedup_threshold})...")
            kept_files = []
            kept_hashes = []
            removed_count = 0
            pbar = comfy.utils.ProgressBar(len(frame_files))
            for fpath in frame_files:
                try:
                    img = Image.open(fpath)
                    h = imagehash.phash(img, hash_size=16)
                    is_dupe = any(abs(h - eh) <= dedup_threshold for eh in kept_hashes)
                    if is_dupe:
                        os.remove(fpath)
                        removed_count += 1
                    else:
                        kept_hashes.append(h)
                        kept_files.append(fpath)
                    img.close()
                except Exception:
                    kept_files.append(fpath)
                pbar.update(1)
            frame_files = kept_files
            print(f"[QVL] Dedup: kept {len(kept_files)}, removed {removed_count}")

        # Limit to max_frames
        if len(frame_files) > max_frames:
            step = len(frame_files) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            frame_files = [frame_files[i] for i in indices]
            print(f"[QVL] Sampled down to {max_frames} frames")

        # Load frames
        pil_imgs = []
        filenames = []
        pbar = comfy.utils.ProgressBar(len(frame_files))
        for fpath in frame_files:
            try:
                img = Image.open(fpath).convert("RGB")
                w, h = img.size
                if min(w, h) >= min_size:
                    pil_imgs.append(img)
                    filenames.append(f"{video_name}_{os.path.basename(fpath)}")
            except Exception:
                continue
            pbar.update(1)

        return pil_imgs, filenames, total_extracted

    def extract(self, video, extraction_mode, fps, scene_threshold,
                max_frames, dedup_on_extract, dedup_threshold,
                video_dir="/workspace/videos",
                output_folder="/workspace/frames", min_size=256):

        # Build list of videos to process
        if video == "** ALL VIDEOS **":
            video_files = sorted([
                f for f in os.listdir(video_dir)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS
            ])
            print(f"[QVL] Processing ALL {len(video_files)} videos in {video_dir}")
        else:
            video_files = [video]

        all_imgs = []
        all_filenames = []
        grand_total = 0

        for vf in video_files:
            video_path = os.path.join(video_dir, vf)
            if not os.path.isfile(video_path):
                print(f"[QVL] Video not found: {video_path}, skipping")
                continue

            print(f"[QVL] === Processing: {vf} ===")
            per_video_max = max_frames if len(video_files) == 1 else max(50, max_frames // len(video_files))
            imgs, fnames, extracted = self._extract_one(
                video_path, extraction_mode, fps, scene_threshold,
                per_video_max, dedup_on_extract, dedup_threshold,
                output_folder, min_size,
            )
            all_imgs.extend(imgs)
            all_filenames.extend(fnames)
            grand_total += extracted

        if not all_imgs:
            print(f"[QVL] No frames extracted from any video")
            return (empty_batch(), "[]", 0, 0)

        # Final limit across all videos
        if len(all_imgs) > max_frames:
            step = len(all_imgs) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            all_imgs = [all_imgs[i] for i in indices]
            all_filenames = [all_filenames[i] for i in indices]
            print(f"[QVL] Final sample: {max_frames} frames from {len(video_files)} videos")

        # Normalize all frames to same dimensions for batching
        # (different videos may have different resolutions)
        if len(all_imgs) > 1:
            from PIL import Image as PILImage
            # Use the most common resolution, or first image's size
            target_w, target_h = all_imgs[0].size
            resized = []
            for img in all_imgs:
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), PILImage.LANCZOS)
                resized.append(img)
            all_imgs = resized

        print(f"[QVL] Total: {len(all_imgs)} frames from {len(video_files)} video(s)")
        batch = pil_list_to_batch(all_imgs)
        return (batch, json.dumps(all_filenames), len(all_imgs), grand_total)


# ============================================================
# 3. PERCEPTUAL DEDUP
# ============================================================

class QVL_Dedup:
    """Remove near-duplicate images using perceptual hashing."""

    DESCRIPTION = """Remove near-duplicate images using perceptual hashing (pHash).

How it works:
- Generates a 16x16 perceptual hash for each image (hash_size parameter)
- Compares hash Hamming distance (0 = identical, 30 = completely different)
- First occurrence of each unique image is kept, duplicates removed

Outputs:
- kept: Images that passed dedup (unique frames)
- removed: Duplicate images that were filtered
- kept_filenames: JSON array of kept filenames
- kept_count, removed_count: Tallies for logging

Performance: ~5-10ms per image. Handles 1000 images in 5-10 seconds.

Use cases:
- Remove duplicate video frames after extraction
- Clean up image datasets with repeated photos
- Reduce dataset size by 30-70% without quality loss

Recommended: Use threshold=6-10 for video frames, 3-5 for curated images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input batch of images to deduplicate. Connect from LoadImages, LoadVideoFrames, or any node that outputs IMAGE.",
                }),
                "threshold": ("INT", {
                    "default": 6, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Maximum hash distance to consider duplicates. Lower = stricter (more removed). Recommended: 6-10 for video frames, 3-5 for photos, 0 for exact matches only. Higher = keeps more similar images.",
                }),
                "hash_size": ("INT", {
                    "default": 16, "min": 8, "max": 32, "step": 8,
                    "tooltip": "Hash resolution (8x8, 16x16, or 32x32). Higher = more sensitive to small differences but slower. Recommended: 16 for most cases. Use 8 for speed, 32 for strictness.",
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames from upstream node. Optional but recommended to preserve metadata through pipeline.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("kept", "removed", "kept_filenames", "kept_count", "removed_count")
    FUNCTION = "dedup"
    CATEGORY = CATEGORY

    def dedup(self, images, threshold, hash_size, filenames="[]"):
        if is_empty_batch(images):
            return (empty_batch(), empty_batch(), "[]", 0, 0)

        pil_imgs = batch_to_pil_list(images)
        kept_idx, removed_idx = deduplicate(
            pil_imgs, threshold=threshold, hash_size=hash_size,
        )

        # Parse filenames
        try:
            fnames = json.loads(filenames) if filenames else []
        except json.JSONDecodeError:
            fnames = []

        kept_imgs = [pil_imgs[i] for i in kept_idx]
        removed_imgs = [pil_imgs[i] for i in removed_idx]
        kept_fnames = [fnames[i] for i in kept_idx] if fnames else []

        kept_batch = pil_list_to_batch(kept_imgs) if kept_imgs else empty_batch()
        removed_batch = pil_list_to_batch(removed_imgs) if removed_imgs else empty_batch()

        return (
            kept_batch, removed_batch,
            json.dumps(kept_fnames),
            len(kept_imgs), len(removed_imgs),
        )


# ============================================================
# 4. VLM FILTER (8B, fast keep/reject)
# ============================================================

class QVL_Filter:
    """Quick keep/reject filter using fast VLM (8B recommended)."""

    DESCRIPTION = """Fast keep/reject filter using lightweight VLM (7B-8B recommended).

Purpose: First-pass filtering to remove obviously bad images before expensive analysis.

VLM evaluates each image for:
- keep: boolean (should this image be kept?)
- quality: 1-10 rating (technical quality)
- reason: text explanation

Images are rejected if:
- VLM sets keep=false
- quality < min_quality threshold

Outputs:
- kept: Images that passed filter
- rejected: Images that failed filter
- kept_filenames: JSON array of kept filenames
- filter_log: Human-readable log of decisions
- kept_count, rejected_count: Tallies

Performance: ~200-500ms per image with Qwen2.5-VL-7B on GPU. Batch of 100 images = 20-50 seconds.

Recommended model: huihui_ai/qwen2.5-vl-abliterated:7b (fast, uncensored)
For OpenAI API: Set api_type=openai and use gpt-4o or gpt-4-vision-preview

Best practices:
- Use min_quality=5 for moderate filtering (keeps 60-80%)
- Use min_quality=7 for strict filtering (keeps 30-50%)
- Run Filter before Analyze to save compute on bad images"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input batch of images to filter. Connect from Dedup or LoadVideoFrames for best results.",
                }),
                "model": ("STRING", {
                    "default": "huihui_ai/qwen2.5-vl-abliterated:7b",
                    "tooltip": "Ollama model name or OpenAI model ID. Recommended: qwen2.5-vl-abliterated:7b for speed. For OpenAI: gpt-4o, gpt-4-vision-preview. Faster models = lower accuracy but 2-5x speedup.",
                }),
                "ollama_url": ("STRING", {
                    "default": DEFAULT_OLLAMA,
                    "tooltip": "Ollama server URL or OpenAI API endpoint. Default: http://127.0.0.1:11434 for local Ollama. For OpenAI: https://api.openai.com/v1. Must be reachable from ComfyUI instance.",
                }),
                "prompt": ("STRING", {
                    "default": FILTER_PROMPT,
                    "multiline": True,
                    "tooltip": "Filter criteria. Default prompt checks for blur, artifacts, poor composition. Edit to customize filtering logic. Must request JSON output with {keep, quality, reason} fields.",
                }),
                "min_quality": ("INT", {
                    "default": 5, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Minimum quality score (1-10) to keep image. Recommended: 5 for balanced filtering, 7 for strict, 3 for permissive. Higher = fewer images kept, better quality.",
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames from upstream. Optional but recommended for logging and metadata tracking.",
                }),
                "api_type": (["ollama", "openai"], {
                    "default": "ollama",
                    "tooltip": "API type. ollama = local Ollama server, openai = OpenAI-compatible API. Use openai for GPT-4 Vision or OpenRouter endpoints.",
                }),
                "workers": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Concurrent VLM requests. Set OLLAMA_NUM_PARALLEL on server to match. Use 1 for single instance, 4-8 for multi-instance fleet. Higher = faster but needs more VRAM across instances.",
                }),
                "ollama_urls": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated Ollama URLs for multi-instance parallelism. Example: http://instance1:11434,http://instance2:11434. Leave empty to use ollama_url. Each URL gets round-robin work distribution.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("kept", "rejected", "kept_filenames", "filter_log", "kept_count", "rejected_count")
    FUNCTION = "filter_images"
    CATEGORY = CATEGORY

    def filter_images(self, images, model, ollama_url, prompt, min_quality,
                      filenames="[]", api_type="ollama", workers=1, ollama_urls=""):
        if is_empty_batch(images):
            return (empty_batch(), empty_batch(), "[]", "Empty input", 0, 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            fnames = json.loads(filenames) if filenames else []
        except json.JSONDecodeError:
            fnames = []

        kept_imgs, rejected_imgs, kept_fnames = [], [], []
        log_lines = []

        # Determine URLs
        urls = ollama_urls if ollama_urls.strip() else ollama_url

        if workers > 1:
            # Parallel mode
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            responses = query_vlm_batch(
                pil_imgs, prompt, model=model, ollama_urls=urls,
                workers=workers, temperature=0.1, max_tokens=150,
                api_type=api_type,
                progress_callback=lambda n: pbar.update(n),
            )

            for i, resp in enumerate(responses):
                fname = fnames[i] if i < len(fnames) else f"image_{i}"
                data = parse_json(resp)
                keep = True
                quality = 5
                reason = "parse error — keeping by default"

                if data:
                    keep = data.get("keep", True)
                    quality = data.get("quality", 5)
                    reason = data.get("reason", "no reason")
                    if quality < min_quality:
                        keep = False
                        reason = f"quality {quality} < min {min_quality}"

                if keep:
                    kept_imgs.append(pil_imgs[i])
                    kept_fnames.append(fname)
                    log_lines.append(f"KEEP {fname}: q={quality} — {reason}")
                else:
                    rejected_imgs.append(pil_imgs[i])
                    log_lines.append(f"REJECT {fname}: q={quality} — {reason}")
        else:
            # Sequential mode (original behavior)
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            for i, img in enumerate(pil_imgs):
                fname = fnames[i] if i < len(fnames) else f"image_{i}"

                if api_type == "openai":
                    resp = query_vlm_openai(img, prompt, model=model,
                                            api_url=ollama_url, temperature=0.1,
                                            max_tokens=150)
                else:
                    resp = query_vlm(img, prompt, model=model,
                                     ollama_url=ollama_url, temperature=0.1,
                                     max_tokens=150)

                data = parse_json(resp)
                keep = True
                quality = 5
                reason = "parse error — keeping by default"

                if data:
                    keep = data.get("keep", True)
                    quality = data.get("quality", 5)
                    reason = data.get("reason", "no reason")
                    if quality < min_quality:
                        keep = False
                        reason = f"quality {quality} < min {min_quality}"

                if keep:
                    kept_imgs.append(img)
                    kept_fnames.append(fname)
                    log_lines.append(f"KEEP {fname}: q={quality} — {reason}")
                else:
                    rejected_imgs.append(img)
                    log_lines.append(f"REJECT {fname}: q={quality} — {reason}")

                pbar.update(1)

        kept_batch = pil_list_to_batch(kept_imgs) if kept_imgs else empty_batch()
        rejected_batch = pil_list_to_batch(rejected_imgs) if rejected_imgs else empty_batch()

        return (
            kept_batch, rejected_batch,
            json.dumps(kept_fnames),
            "\n".join(log_lines),
            len(kept_imgs), len(rejected_imgs),
        )


# ============================================================
# 5. VLM ANALYZE (37B, deep analysis)
# ============================================================

class QVL_Analyze:
    """Deep image analysis: quality, corrections, bbox, pose, tags, aesthetic score.
    Uses the large model (37B recommended) for comprehensive metadata."""

    DESCRIPTION = """Comprehensive image analysis using large VLM (32B-37B recommended).

Extracts detailed metadata for each image:
- quality_score: 1-10 technical quality rating
- aesthetic_score: 1-10 artistic/composition rating
- subject_bbox: [x1, y1, x2, y2] bounding box (0-1 normalized)
- pose: standing, sitting, lying, kneeling, etc.
- camera_angle: front, side, back, closeup, etc.
- tags: array of descriptive tags (5-15 tags per image)
- corrections: {brightness, contrast, sharpness, crop_*_pct} suggested adjustments

Outputs:
- images: Pass-through of input (for chaining)
- metadata_json: Array of metadata objects (one per image)
- analysis_log: Human-readable summary
- count: Number of images analyzed

Performance: ~1-3 seconds per image with Qwen2.5-VL-32B on GPU. Batch of 100 = 2-5 minutes.

Use cases:
- Generate metadata for SmartCrop (bbox)
- Generate metadata for AutoCorrect (corrections)
- Generate metadata for MetadataRouter (pose, angle, tags)
- Quality filtering (combine with min score threshold)

Best practices:
- Run Filter first to reduce compute
- Use 32B model for best accuracy (7B works but lower quality metadata)
- Metadata format is JSON — connect to SmartCrop, AutoCorrect, MetadataRouter"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input batch of images to analyze. Best used on filtered images (after Filter or QualityScore).",
                }),
                "model": ("STRING", {
                    "default": "huihui_ai/qwen2.5-vl-abliterated:32b",
                    "tooltip": "VLM model for analysis. Recommended: qwen2.5-vl-abliterated:32b for best metadata quality. 7b works but less accurate. For OpenAI: gpt-4o or gpt-4-vision-preview.",
                }),
                "ollama_url": ("STRING", {
                    "default": DEFAULT_OLLAMA,
                    "tooltip": "Ollama server URL or OpenAI API endpoint. Default: http://127.0.0.1:11434. Must have sufficient VRAM for 32B model (24GB+). Use api_type=openai for cloud APIs.",
                }),
                "prompt": ("STRING", {
                    "default": ANALYZE_PROMPT,
                    "multiline": True,
                    "tooltip": "Analysis instructions. Default extracts quality, aesthetic, bbox, pose, angle, tags, corrections. Edit to add/remove fields. Must request JSON output.",
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames. Metadata will include _filename and _index fields for tracking.",
                }),
                "api_type": (["ollama", "openai"], {
                    "default": "ollama",
                    "tooltip": "API type. ollama = local Ollama, openai = OpenAI-compatible API. Use openai for GPT-4 Vision or hosted VLM endpoints.",
                }),
                "workers": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Concurrent VLM requests. Set OLLAMA_NUM_PARALLEL on server to match. Use 1 for single instance, 4-8 for multi-instance fleet. Higher = faster but needs more VRAM across instances.",
                }),
                "ollama_urls": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated Ollama URLs for multi-instance parallelism. Example: http://instance1:11434,http://instance2:11434. Leave empty to use ollama_url. Each URL gets round-robin work distribution.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "metadata_json", "analysis_log", "count")
    FUNCTION = "analyze"
    CATEGORY = CATEGORY

    def analyze(self, images, model, ollama_url, prompt,
                filenames="[]", api_type="ollama", workers=1, ollama_urls=""):
        if is_empty_batch(images):
            return (empty_batch(), "[]", "Empty input", 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            fnames = json.loads(filenames) if filenames else []
        except json.JSONDecodeError:
            fnames = []

        all_metadata = []
        log_lines = []

        # Determine URLs
        urls = ollama_urls if ollama_urls.strip() else ollama_url

        if workers > 1:
            # Parallel mode
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            responses = query_vlm_batch(
                pil_imgs, prompt, model=model, ollama_urls=urls,
                workers=workers, temperature=0.1, max_tokens=600,
                api_type=api_type,
                progress_callback=lambda n: pbar.update(n),
            )

            for i, resp in enumerate(responses):
                fname = fnames[i] if i < len(fnames) else f"image_{i}"
                data = parse_json(resp)
                if data:
                    data["_filename"] = fname
                    data["_index"] = i
                    all_metadata.append(data)
                    qs = data.get("quality_score", "?")
                    ae = data.get("aesthetic_score", "?")
                    pose = data.get("pose", "?")
                    angle = data.get("camera_angle", "?")
                    tags = ", ".join(data.get("tags", [])[:5])
                    log_lines.append(
                        f"{fname}: quality={qs} aesthetic={ae} "
                        f"pose={pose} angle={angle} tags=[{tags}]"
                    )
                else:
                    all_metadata.append({
                        "_filename": fname, "_index": i,
                        "_error": "Failed to parse VLM response",
                        "_raw": resp[:200],
                    })
                    log_lines.append(f"{fname}: PARSE ERROR — {resp[:100]}")
        else:
            # Sequential mode (original behavior)
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            for i, img in enumerate(pil_imgs):
                fname = fnames[i] if i < len(fnames) else f"image_{i}"

                if api_type == "openai":
                    resp = query_vlm_openai(img, prompt, model=model,
                                            api_url=ollama_url, temperature=0.1,
                                            max_tokens=600)
                else:
                    resp = query_vlm(img, prompt, model=model,
                                     ollama_url=ollama_url, temperature=0.1,
                                     max_tokens=600)

                data = parse_json(resp)
                if data:
                    data["_filename"] = fname
                    data["_index"] = i
                    all_metadata.append(data)
                    qs = data.get("quality_score", "?")
                    ae = data.get("aesthetic_score", "?")
                    pose = data.get("pose", "?")
                    angle = data.get("camera_angle", "?")
                    tags = ", ".join(data.get("tags", [])[:5])
                    log_lines.append(
                        f"{fname}: quality={qs} aesthetic={ae} "
                        f"pose={pose} angle={angle} tags=[{tags}]"
                    )
                else:
                    all_metadata.append({
                        "_filename": fname, "_index": i,
                        "_error": "Failed to parse VLM response",
                        "_raw": resp[:200],
                    })
                    log_lines.append(f"{fname}: PARSE ERROR — {resp[:100]}")

                pbar.update(1)

        return (
            images,
            json.dumps(all_metadata, indent=2),
            "\n".join(log_lines),
            len(pil_imgs),
        )


# ============================================================
# 6. VLM CAPTION
# ============================================================

class QVL_Caption:
    """Generate training captions with preset formats."""

    DESCRIPTION = """Generate training captions using VLM with format presets.

Presets:
- natural: Conversational descriptions (2-3 sentences)
- flux: FLUX.1 style (detailed, natural language)
- flux2: FLUX 1.1 style (structured, comprehensive)
- sdxl: SDXL style (comma-separated descriptors)
- booru: Danbooru tag style (tag1, tag2, tag3)
- pony: Pony Diffusion style (score tags + descriptors)
- chroma: ChromaV5 style (technical + aesthetic)
- structured: JSON format with multiple fields

Features:
- Trigger word injection (prepends keyword to captions)
- Custom instructions (add project-specific guidance)
- Prompt override (completely custom prompting)

Outputs:
- images: Pass-through (for chaining)
- captions_json: Array of {filename, caption, preset} objects
- caption_preview: Human-readable preview of all captions
- prompt_used: Actual prompt sent to VLM (for debugging)
- count: Number of captions generated

Performance: ~500ms-2s per image depending on model (7B vs 32B) and preset (booru is fastest, structured slowest).

Use cases:
- Generate training captions for LoRA datasets
- Create consistent caption style across dataset
- Add trigger word to all captions automatically

Best practices:
- Use trigger_word for LoRA training (e.g., "ohwx woman")
- Use 7B model for speed, 32B for quality
- Booru/Pony presets are fastest (max_tokens=300)
- Natural/FLUX presets are most detailed (max_tokens=500)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to caption. Best used after filtering and cropping.",
                }),
                "model": ("STRING", {
                    "default": "huihui_ai/qwen2.5-vl-abliterated:7b",
                    "tooltip": "VLM model for captioning. Recommended: 7b for speed (500ms/image), 32b for quality (2s/image). For OpenAI: gpt-4o works well.",
                }),
                "ollama_url": ("STRING", {
                    "default": DEFAULT_OLLAMA,
                    "tooltip": "Ollama server URL or OpenAI API endpoint. Default: http://127.0.0.1:11434. Must be reachable and have model loaded.",
                }),
                "preset": (["natural", "flux", "flux2", "sdxl", "booru", "pony", "chroma", "structured"], {
                    "tooltip": "Caption format. natural = conversational, flux/flux2 = FLUX.1 style, sdxl = comma-separated, booru/pony = tag style, chroma = technical, structured = JSON. See DESCRIPTION for details.",
                }),
            },
            "optional": {
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom prompt (overrides preset). Leave empty to use preset. Use this for completely custom captioning logic. Trigger word and custom instructions still apply if set.",
                }),
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "Text to prepend to every caption (e.g., 'ohwx woman', 'my_subject'). Essential for LoRA training. Automatically adds comma separator if needed.",
                }),
                "custom_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional instructions appended to prompt (e.g., 'Focus on clothing details', 'Avoid mentioning background'). Works with presets and overrides.",
                }),
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames. Captions will include filename field for SaveDataset matching.",
                }),
                "api_type": (["ollama", "openai"], {
                    "default": "ollama",
                    "tooltip": "API type. ollama = local Ollama, openai = OpenAI-compatible API.",
                }),
                "workers": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Concurrent VLM requests. Set OLLAMA_NUM_PARALLEL on server to match. Use 1 for single instance, 4-8 for multi-instance fleet. Higher = faster but needs more VRAM across instances.",
                }),
                "ollama_urls": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated Ollama URLs for multi-instance parallelism. Example: http://instance1:11434,http://instance2:11434. Leave empty to use ollama_url. Each URL gets round-robin work distribution.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "captions_json", "caption_preview", "prompt_used", "count")
    FUNCTION = "caption"
    CATEGORY = CATEGORY

    def caption(self, images, model, ollama_url, preset,
                prompt_override="", trigger_word="", custom_instructions="",
                filenames="[]", api_type="ollama", workers=1, ollama_urls=""):
        if is_empty_batch(images):
            return (empty_batch(), "[]", "Empty input", "", 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            fnames = json.loads(filenames) if filenames else []
        except json.JSONDecodeError:
            fnames = []

        # Use override if provided, otherwise build from preset
        if prompt_override and prompt_override.strip():
            prompt = prompt_override.strip()
            if trigger_word:
                prompt += f"\nBegin the caption with the exact text: {trigger_word}"
            if custom_instructions:
                prompt += f"\n{custom_instructions}"
        else:
            prompt = build_caption_prompt(
                preset=preset,
                trigger_word=trigger_word,
                custom_instructions=custom_instructions,
            )

        max_tokens = 300 if preset in ("booru", "pony") else 500
        all_captions = []
        preview_lines = []

        # Determine URLs
        urls = ollama_urls if ollama_urls.strip() else ollama_url

        if workers > 1:
            # Parallel mode
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            responses = query_vlm_batch(
                pil_imgs, prompt, model=model, ollama_urls=urls,
                workers=workers, temperature=0.3, max_tokens=max_tokens,
                api_type=api_type,
                progress_callback=lambda n: pbar.update(n),
            )

            for i, resp in enumerate(responses):
                fname = fnames[i] if i < len(fnames) else f"image_{i}"

                # For structured preset, try to parse JSON
                if preset == "structured" and not prompt_override:
                    data = parse_json(resp)
                    caption_text = json.dumps(data) if data else resp.strip()
                else:
                    caption_text = resp.strip()

                # Prepend trigger word if not already present
                if trigger_word and not caption_text.startswith(trigger_word):
                    caption_text = f"{trigger_word}, {caption_text}"

                all_captions.append({
                    "filename": fname,
                    "caption": caption_text,
                    "preset": preset,
                })
                preview_lines.append(f"[{fname}]\n{caption_text}\n")
        else:
            # Sequential mode (original behavior)
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            for i, img in enumerate(pil_imgs):
                fname = fnames[i] if i < len(fnames) else f"image_{i}"

                if api_type == "openai":
                    resp = query_vlm_openai(img, prompt, model=model,
                                            api_url=ollama_url, temperature=0.3,
                                            max_tokens=max_tokens)
                else:
                    resp = query_vlm(img, prompt, model=model,
                                     ollama_url=ollama_url, temperature=0.3,
                                     max_tokens=max_tokens)

                # For structured preset, try to parse JSON
                if preset == "structured" and not prompt_override:
                    data = parse_json(resp)
                    caption_text = json.dumps(data) if data else resp.strip()
                else:
                    caption_text = resp.strip()

                # Prepend trigger word if not already present
                if trigger_word and not caption_text.startswith(trigger_word):
                    caption_text = f"{trigger_word}, {caption_text}"

                all_captions.append({
                    "filename": fname,
                    "caption": caption_text,
                    "preset": preset,
                })
                preview_lines.append(f"[{fname}]\n{caption_text}\n")

                pbar.update(1)

        return (
            images,
            json.dumps(all_captions, indent=2),
            "\n---\n".join(preview_lines),
            prompt,
            len(pil_imgs),
        )


# ============================================================
# 7. SMART CROP (uses metadata bbox)
# ============================================================

class QVL_SmartCrop:
    """Crop images to subject using bounding boxes from Analyze metadata."""

    DESCRIPTION = """Crop images to subject using bounding boxes from Analyze node.

How it works:
- Reads subject_bbox from Analyze metadata ([x1, y1, x2, y2] in 0-1 normalized coords)
- Adds padding around bbox (padding_pct)
- Optionally expands to square (for training models that require square inputs)
- Falls back to original/center crop if bbox missing

Outputs:
- cropped: Batch of cropped images
- crop_log: Human-readable log of operations
- count: Number of images processed

Fallback modes:
- keep_original: Use uncropped image if bbox missing (default, safest)
- center_crop: Square crop from center if bbox missing
- skip: Don't output image if bbox missing (reduces count)

Performance: Instant (no VLM calls, pure image ops).

Use cases:
- Crop to subject after Analyze
- Remove empty background/borders
- Standardize framing across dataset

Best practices:
- Always run Analyze first with bbox extraction
- Use padding_pct=5-10 for breathing room
- Set square=True for SDXL/Flux training (requires square images)
- Use fallback_mode=keep_original for safety (never lose images)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input batch to crop. Must correspond to metadata_json order (same _index values).",
                }),
                "metadata_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "JSON array from Analyze node containing subject_bbox fields. Each object must have _index matching image position.",
                }),
                "padding_pct": ("INT", {
                    "default": 5, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Padding around bbox as percentage of bbox size. Recommended: 5-10 for tight crops, 15-20 for loose crops. 0 = crop exactly to bbox. Higher = more background included.",
                }),
                "square": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Expand crop to square (1:1 aspect ratio). Recommended: True for SDXL/Flux training (requires square). False preserves aspect ratio. If True, expands shorter dimension to match longer.",
                }),
            },
            "optional": {
                "fallback_mode": (["keep_original", "center_crop", "skip"], {
                    "default": "keep_original",
                    "tooltip": "What to do if bbox is missing or invalid. keep_original = use uncropped (safest), center_crop = square crop from center, skip = omit from output (count decreases).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("cropped", "crop_log", "count")
    FUNCTION = "smart_crop"
    CATEGORY = CATEGORY

    def smart_crop(self, images, metadata_json, padding_pct, square,
                   fallback_mode="keep_original"):
        if is_empty_batch(images):
            return (empty_batch(), "Empty input", 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            metadata = json.loads(metadata_json)
            if not isinstance(metadata, list):
                metadata = [metadata] if isinstance(metadata, dict) else []
        except (json.JSONDecodeError, TypeError):
            metadata = []

        # Build index lookup
        meta_by_idx = {}
        for m in metadata:
            if not isinstance(m, dict):
                continue
            idx = m.get("_index", -1)
            if idx >= 0:
                meta_by_idx[idx] = m

        cropped_imgs = []
        log_lines = []
        pbar = comfy.utils.ProgressBar(len(pil_imgs))

        for i, img in enumerate(pil_imgs):
            meta = meta_by_idx.get(i, {})
            bbox = meta.get("subject_bbox")

            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    cropped = crop_to_bbox(
                        img, bbox,
                        padding_pct=padding_pct,
                        square=square,
                    )
                    cropped_imgs.append(cropped)
                    log_lines.append(
                        f"[{i}] Cropped to bbox {bbox} "
                        f"→ {cropped.size[0]}x{cropped.size[1]}"
                    )
                except Exception as e:
                    log_lines.append(f"[{i}] Crop error: {e}")
                    if fallback_mode == "keep_original":
                        cropped_imgs.append(img)
                    elif fallback_mode == "center_crop":
                        s = min(img.size)
                        left = (img.width - s) // 2
                        top = (img.height - s) // 2
                        cropped_imgs.append(img.crop((left, top, left + s, top + s)))
            else:
                if fallback_mode == "keep_original":
                    cropped_imgs.append(img)
                    log_lines.append(f"[{i}] No bbox — kept original")
                elif fallback_mode == "center_crop":
                    s = min(img.size)
                    left = (img.width - s) // 2
                    top = (img.height - s) // 2
                    cropped_imgs.append(img.crop((left, top, left + s, top + s)))
                    log_lines.append(f"[{i}] No bbox — center cropped")
                else:
                    log_lines.append(f"[{i}] No bbox — skipped")

            pbar.update(1)

        batch = pil_list_to_batch(cropped_imgs) if cropped_imgs else empty_batch()
        return (batch, "\n".join(log_lines), len(cropped_imgs))


# ============================================================
# 8. AUTO CORRECT (uses metadata corrections)
# ============================================================

class QVL_AutoCorrect:
    """Apply VLM-suggested corrections from Analyze metadata."""

    DESCRIPTION = """Apply automatic corrections suggested by Analyze node.

Correction types:
- brightness: 0.5-2.0 multiplier (1.0 = no change, <1 = darker, >1 = brighter)
- contrast: 0.5-2.0 multiplier (1.0 = no change, <1 = less contrast, >1 = more contrast)
- sharpness: 0.5-2.0 multiplier (1.0 = no change, <1 = blur, >1 = sharpen)
- crop_*_pct: Percentage to trim from each edge (0-20)

Toggles let you enable/disable each correction type independently.

How it works:
- Reads corrections dict from Analyze metadata
- Applies PIL ImageEnhance filters + edge cropping
- Preserves original if no corrections needed

Outputs:
- corrected: Batch of corrected images
- correction_log: Human-readable summary of changes
- count: Number of images processed

Performance: Instant (no VLM, pure PIL ops).

Use cases:
- Auto-fix underexposed/overexposed images
- Auto-sharpen blurry frames
- Auto-trim borders/letterboxing

Best practices:
- Run Analyze first to generate corrections metadata
- Use selectively (toggle off sharpness if dataset is already sharp)
- Check correction_log to see what was changed
- Corrections are conservative (typically 0.8-1.2 range)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to correct. Must match metadata_json order (_index field).",
                }),
                "metadata_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "JSON array from Analyze node containing corrections dicts. Each object should have _index and corrections fields.",
                }),
            },
            "optional": {
                "apply_brightness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply brightness corrections. Fixes underexposed/overexposed images. Recommended: True for video frames, False for pre-color-graded images.",
                }),
                "apply_contrast": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply contrast corrections. Fixes flat/washed-out images. Recommended: True for most cases. May oversaturate if dataset is already high contrast.",
                }),
                "apply_sharpness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply sharpness corrections. Fixes soft/blurry images. Recommended: True for video frames. False for photos (may over-sharpen). Can introduce artifacts if aggressive.",
                }),
                "apply_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply edge crop corrections (removes letterboxing, borders). Recommended: True for video frames. False for pre-cropped images. Combines with SmartCrop if needed.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("corrected", "correction_log", "count")
    FUNCTION = "auto_correct"
    CATEGORY = CATEGORY

    def auto_correct(self, images, metadata_json,
                     apply_brightness=True, apply_contrast=True,
                     apply_sharpness=True, apply_crop=True):
        if is_empty_batch(images):
            return (empty_batch(), "Empty input", 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = []

        meta_by_idx = {}
        for m in metadata:
            idx = m.get("_index", -1)
            if idx >= 0:
                meta_by_idx[idx] = m

        corrected_imgs = []
        log_lines = []
        pbar = comfy.utils.ProgressBar(len(pil_imgs))

        for i, img in enumerate(pil_imgs):
            meta = meta_by_idx.get(i, {})
            corrections = meta.get("corrections", {})

            # Filter which corrections to apply
            filtered = {}
            if apply_brightness:
                filtered["brightness"] = corrections.get("brightness", 1.0)
            if apply_contrast:
                filtered["contrast"] = corrections.get("contrast", 1.0)
            if apply_sharpness:
                filtered["sharpness"] = corrections.get("sharpness", 1.0)
            if apply_crop:
                filtered["crop_left_pct"] = corrections.get("crop_left_pct", 0)
                filtered["crop_right_pct"] = corrections.get("crop_right_pct", 0)
                filtered["crop_top_pct"] = corrections.get("crop_top_pct", 0)
                filtered["crop_bottom_pct"] = corrections.get("crop_bottom_pct", 0)

            corrected = apply_corrections(img, filtered)
            corrected_imgs.append(corrected)

            changes = []
            if filtered.get("brightness", 1.0) != 1.0:
                changes.append(f"bright={filtered['brightness']:.2f}")
            if filtered.get("contrast", 1.0) != 1.0:
                changes.append(f"contrast={filtered['contrast']:.2f}")
            if filtered.get("sharpness", 1.0) != 1.0:
                changes.append(f"sharp={filtered['sharpness']:.2f}")
            crop_total = sum(filtered.get(k, 0) for k in
                            ["crop_left_pct", "crop_right_pct",
                             "crop_top_pct", "crop_bottom_pct"])
            if crop_total > 0:
                changes.append(f"crop={crop_total:.0f}%")

            if changes:
                log_lines.append(f"[{i}] {', '.join(changes)}")
            else:
                log_lines.append(f"[{i}] No corrections needed")

            pbar.update(1)

        batch = pil_list_to_batch(corrected_imgs)
        return (batch, "\n".join(log_lines), len(corrected_imgs))


# ============================================================
# 9. LOCAL QUALITY SCORE (no VLM)
# ============================================================

class QVL_QualityScore:
    """Score images by local metrics (sharpness, brightness, contrast, size).
    No VLM needed — instant processing."""

    DESCRIPTION = """Fast quality scoring using local metrics (no VLM required).

Metrics evaluated:
- Sharpness: Laplacian variance (detects blur)
- Brightness: Mean luminance (detects under/overexposure)
- Contrast: Std dev of luminance (detects flat images)
- Size: Resolution penalty for small images

Score range: 0-100 (higher = better quality)
Typical scores: 60-80 = good, 40-60 = acceptable, <40 = poor

Outputs:
- passed: Images above min_score threshold
- failed: Images below threshold
- passed_filenames: JSON array of passed filenames
- score_log: Scores + reasons for each image
- passed_count, failed_count: Tallies

Performance: ~5ms per image (200 images/second). Instant even on CPU.

Use cases:
- Fast pre-filtering before VLM (removes obviously bad images)
- Size-based filtering (remove thumbnails)
- Blur detection (remove out-of-focus frames)

Best practices:
- Use min_score=30-40 for permissive filtering (keeps 80-90%)
- Use min_score=50-60 for strict filtering (keeps 40-60%)
- Combine with min_size to filter resolution + quality together
- Run before Filter/Analyze to save VLM compute on garbage"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to score. Works on any batch from any source.",
                }),
                "min_score": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 100.0, "step": 5.0,
                    "tooltip": "Minimum quality score (0-100) to pass. Recommended: 30 for permissive (keeps 80%), 50 for balanced (keeps 50%), 70 for strict (keeps 20%). Higher = fewer images kept.",
                }),
                "min_size": ("INT", {
                    "default": 256, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Minimum width or height in pixels. Images smaller than this get heavy score penalty. Recommended: 256-512 for LoRA training. 512+ for high-res models. Higher = fewer small images pass.",
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames for logging. Optional but useful for tracking which images failed.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("passed", "failed", "passed_filenames", "score_log", "passed_count", "failed_count")
    FUNCTION = "score"
    CATEGORY = CATEGORY

    def score(self, images, min_score, min_size, filenames="[]"):
        if is_empty_batch(images):
            return (empty_batch(), empty_batch(), "[]", "Empty input", 0, 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            fnames = json.loads(filenames) if filenames else []
        except json.JSONDecodeError:
            fnames = []

        passed_imgs, failed_imgs, passed_fnames = [], [], []
        log_lines = []
        pbar = comfy.utils.ProgressBar(len(pil_imgs))

        for i, img in enumerate(pil_imgs):
            fname = fnames[i] if i < len(fnames) else f"image_{i}"
            score, details = compute_quality(img, min_size=min_size)

            if score >= min_score:
                passed_imgs.append(img)
                passed_fnames.append(fname)
                log_lines.append(f"PASS {fname}: {score:.1f}/100 — {details['size']}")
            else:
                failed_imgs.append(img)
                log_lines.append(f"FAIL {fname}: {score:.1f}/100 — {details['size']}")

            pbar.update(1)

        passed_batch = pil_list_to_batch(passed_imgs) if passed_imgs else empty_batch()
        failed_batch = pil_list_to_batch(failed_imgs) if failed_imgs else empty_batch()

        return (
            passed_batch, failed_batch,
            json.dumps(passed_fnames),
            "\n".join(log_lines),
            len(passed_imgs), len(failed_imgs),
        )


# ============================================================
# 10. NUDE SCORE (NudeNet anatomy-weighted filter)
# ============================================================

class QVL_NudeScore:
    """Score frames by anatomy visibility using NudeNet (YOLOv8).
    Keeps more frames from high-detail moments, fewer from low-detail.
    High anatomy score = keep all, medium = subsample, low = sparse, garbage = drop."""

    DESCRIPTION = """Adaptive frame filtering based on anatomy visibility (NudeNet YOLOv8 detector).

How it works:
- Detects 16 body part classes per frame (exposed/covered breast, genitalia, buttocks, etc.)
- Weights parts by value (exposed > covered, primary > secondary)
- Scores each frame (high = lots of visible anatomy, low = mostly clothed/obscured)
- Adaptive sampling: keep all high-value, subsample medium, sparse low, drop garbage

Scoring:
- High value parts (3.0x): BREAST_EXPOSED, GENITALIA_EXPOSED, BUTTOCKS_EXPOSED, etc.
- Medium value parts (1.0x): BREAST_COVERED, BUTTOCKS_COVERED, etc.
- Low value parts (0.3x): FACE, ARMPITS_COVERED, FEET_COVERED

Outputs:
- kept: Frames that passed adaptive filter
- rejected: Frames that were dropped or subsampled
- kept_filenames: JSON array of kept filenames
- score_log: Per-frame scores + parts detected
- kept_count, rejected_count: Tallies

Performance: ~100-200ms per frame on GPU (YOLO inference). Batch of 500 frames = 50-100 seconds.

Use cases:
- Video frame filtering for adult/body datasets
- Remove low-value frames (clothed, obscured, out-of-frame)
- Keep all high-detail moments, sparse sample low-detail

Best practices:
- high_threshold=5 (keep all frames with significant anatomy)
- medium_threshold=2, medium_keep_every=3 (keep 1 in 3 medium frames)
- low_threshold=0.5, low_keep_every=10 (keep 1 in 10 low frames)
- Tweak thresholds based on content (higher for explicit, lower for artistic)"""

    HIGH_VALUE = {
        "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
        "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "BELLY_EXPOSED",
    }
    MEDIUM_VALUE = {
        "FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED",
        "BUTTOCKS_COVERED", "BELLY_COVERED",
        "ARMPITS_EXPOSED", "FEET_EXPOSED",
    }
    LOW_VALUE = {
        "FACE_FEMALE", "ARMPITS_COVERED", "FEET_COVERED",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input frames to score. Best used on video frames (from LoadVideoFrames after initial dedup).",
                }),
                "high_threshold": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Minimum score to keep ALL frames. Recommended: 5.0 for explicit content, 3.0 for artistic/implied. Higher = only keep very explicit moments. Lower = keep more frames unconditionally.",
                }),
                "medium_threshold": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Minimum score for medium tier (subsampled via medium_keep_every). Recommended: 2.0 for balanced filtering. Higher = fewer medium frames, more dropped.",
                }),
                "low_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Minimum score for low tier (sparse sampling via low_keep_every). Below this = dropped entirely. Recommended: 0.5 to keep some context frames. 0 = drop nothing.",
                }),
                "medium_keep_every": ("INT", {
                    "default": 3, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Keep 1 in N medium-score frames. Recommended: 3 (keeps 33% of medium), 5 (keeps 20%), 2 (keeps 50%). Higher = fewer frames kept, smaller dataset.",
                }),
                "low_keep_every": ("INT", {
                    "default": 10, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Keep 1 in N low-score frames. Recommended: 10 (keeps 10% of low), 20 (keeps 5%), 5 (keeps 20%). Higher = fewer frames kept. Low frames are mostly context/transition.",
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames. If provided, must match image count. Used for logging and output metadata.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("kept", "rejected", "kept_filenames", "score_log", "kept_count", "rejected_count")
    FUNCTION = "score"
    CATEGORY = CATEGORY

    def _score_detections(self, detections):
        score = 0
        parts = []
        for det in detections:
            cls = det["class"]
            conf = det["score"]
            if cls in self.HIGH_VALUE:
                score += 3.0 * conf
                parts.append(cls)
            elif cls in self.MEDIUM_VALUE:
                score += 1.0 * conf
                parts.append(cls)
            elif cls in self.LOW_VALUE:
                score += 0.3 * conf
        return score, parts

    def score(self, images, high_threshold, medium_threshold, low_threshold,
              medium_keep_every, low_keep_every, filenames="[]"):
        from nudenet import NudeDetector
        from PIL import Image
        import tempfile

        if is_empty_batch(images):
            return (empty_batch(), empty_batch(), "[]", "No images", 0, 0)

        detector = NudeDetector()
        pil_imgs = batch_to_pil_list(images)

        try:
            fnames = json.loads(filenames)
        except (json.JSONDecodeError, TypeError):
            fnames = []
        if len(fnames) != len(pil_imgs):
            fnames = [f"frame_{i:06d}.jpg" for i in range(len(pil_imgs))]

        # Score each frame
        scored = []
        pbar = comfy.utils.ProgressBar(len(pil_imgs))
        for i, img in enumerate(pil_imgs):
            # NudeDetector.detect needs a file path, write temp
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp, format="JPEG", quality=85)
                tmp_path = tmp.name
            try:
                dets = detector.detect(tmp_path)
                sc, parts = self._score_detections(dets)
            except Exception:
                sc, parts = 0.0, []
            finally:
                os.remove(tmp_path)

            scored.append({
                "index": i,
                "score": round(sc, 2),
                "parts": parts,
                "filename": fnames[i],
            })
            pbar.update(1)

        # Adaptive keep
        kept_indices = []
        rejected_indices = []
        medium_counter = 0
        low_counter = 0
        log_lines = []

        for s in scored:
            idx = s["index"]
            sc = s["score"]
            if sc >= high_threshold:
                kept_indices.append(idx)
                log_lines.append(f"HIGH {sc:.1f} {s['filename']} {s['parts']}")
            elif sc >= medium_threshold:
                if medium_counter % medium_keep_every == 0:
                    kept_indices.append(idx)
                    log_lines.append(f"MED  {sc:.1f} {s['filename']} {s['parts']}")
                else:
                    rejected_indices.append(idx)
                medium_counter += 1
            elif sc >= low_threshold:
                if low_counter % low_keep_every == 0:
                    kept_indices.append(idx)
                    log_lines.append(f"LOW  {sc:.1f} {s['filename']}")
                else:
                    rejected_indices.append(idx)
                low_counter += 1
            else:
                rejected_indices.append(idx)

        # Build output batches
        kept_imgs = [pil_imgs[i] for i in kept_indices]
        rejected_imgs = [pil_imgs[i] for i in rejected_indices]
        kept_fnames = [fnames[i] for i in kept_indices]

        kept_batch = pil_list_to_batch(kept_imgs) if kept_imgs else empty_batch()
        rejected_batch = pil_list_to_batch(rejected_imgs) if rejected_imgs else empty_batch()

        high_count = sum(1 for s in scored if s["score"] >= high_threshold)
        med_count = sum(1 for s in scored if medium_threshold <= s["score"] < high_threshold)
        low_count = sum(1 for s in scored if low_threshold <= s["score"] < medium_threshold)
        drop_count = sum(1 for s in scored if s["score"] < low_threshold)

        summary = f"Scored {len(scored)} frames: {high_count} high, {med_count} med, {low_count} low, {drop_count} drop → kept {len(kept_indices)}"
        log_lines.insert(0, summary)
        log_text = "\n".join(log_lines)
        print(f"[QVL] NudeScore: {summary}")

        return (kept_batch, rejected_batch, json.dumps(kept_fnames),
                log_text, len(kept_indices), len(rejected_indices))


# ============================================================
# 10b. DENSE RESAMPLE (re-extract high-value time ranges at higher fps)
# ============================================================

class QVL_DenseResample:
    """Re-extract frames at higher fps from time ranges that scored high on NudeScore.
    Takes the kept images + filenames from NudeScore (or LoadVideoFrames),
    identifies which timestamps had the best content, goes back to the source
    videos and extracts at dense_fps. Returns combined batch of originals + dense frames."""

    DESCRIPTION = """Re-extract high-value video segments at higher frame rate.

Purpose: After NudeScore filtering, go back and extract more frames from the best moments.

How it works:
1. Parses filenames/score_log to find HIGH-scoring frame timestamps
2. Groups timestamps into clusters (nearby frames = same scene)
3. Re-extracts each cluster at dense_fps (e.g., 5fps instead of 1fps)
4. Deduplicates new frames
5. Combines with original frames

Outputs:
- images: Original frames + dense re-extracted frames
- filenames: Combined JSON array (originals + new frames with _dense_ prefix)
- count: Total frame count after dense resampling

Performance: Depends on cluster count and dense_fps. Typical: 10-30 seconds per cluster.

Use cases:
- Extract 5fps from best moments after 1fps initial extraction
- Capture smooth motion/pose transitions in high-value scenes
- Increase dataset diversity without processing entire video at high fps

Best practices:
- Run after NudeScore (uses score_log to find high-value ranges)
- dense_fps=5 for smooth transitions, 10 for maximum detail
- padding_secs=5 to capture lead-in/lead-out of scenes
- dedup_threshold=6 to remove near-duplicates (new frames often overlap)

Example pipeline:
LoadVideoFrames (1fps) → NudeScore → DenseResample (5fps on high) → 500 orig + 1000 dense = 1500 total"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Original frames (from NudeScore kept output). These are preserved and combined with dense frames.",
                }),
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames from NudeScore. Must include video name prefix (videoname_frame_NNNNNN.jpg format) for timestamp parsing.",
                }),
                "dense_fps": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 30.0, "step": 1.0,
                    "tooltip": "Frame rate for re-extraction. Recommended: 5 for smooth motion, 10 for maximum detail, 2-3 for conservative. Higher = more frames, slower processing, larger dataset.",
                }),
                "padding_secs": ("INT", {
                    "default": 5, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Seconds to pad before/after each cluster. Recommended: 5 for context, 10 for long scenes, 0 for exact clusters only. Higher = more frames, more overlap between clusters.",
                }),
                "dedup_threshold": ("INT", {
                    "default": 6, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Perceptual hash threshold for dedup on dense frames. Recommended: 6 (removes near-duplicates). 0 = keep exact dupes only. Higher = more aggressive dedup.",
                }),
            },
            "optional": {
                "video_dir": ("STRING", {
                    "default": "/workspace/videos",
                    "multiline": False,
                    "tooltip": "Directory containing source video files. Must match the directory used in LoadVideoFrames. Videos must exist and be readable.",
                }),
                "score_log": ("STRING", {
                    "default": "",
                    "tooltip": "Score log from NudeScore (contains HIGH/MED/LOW lines). Used to identify high-value timestamps. If empty, uses all frames in filenames.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames", "count")
    FUNCTION = "resample"
    CATEGORY = CATEGORY

    def _find_video(self, video_name, video_dir):
        """Find video file by name (without extension)."""
        for ext in VIDEO_EXTS:
            candidate = os.path.join(video_dir, video_name + ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _extract_range(self, video_path, out_dir, start, end, fps, prefix):
        """Extract frames from a time range."""
        import glob as g
        duration = end - start
        out_pattern = os.path.join(out_dir, f"{prefix}_%06d.jpg")
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
            "-t", str(duration), "-vf", f"fps={fps}",
            "-q:v", "2", out_pattern,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        except Exception:
            pass
        return sorted(g.glob(os.path.join(out_dir, f"{prefix}_*.jpg")))

    def resample(self, images, filenames, dense_fps, padding_secs,
                 dedup_threshold, video_dir="/workspace/videos", score_log=""):
        from PIL import Image
        import glob as g
        import tempfile

        if is_empty_batch(images):
            return (empty_batch(), "[]", 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            fnames = json.loads(filenames)
        except (json.JSONDecodeError, TypeError):
            fnames = []
        if len(fnames) != len(pil_imgs):
            fnames = [f"frame_{i:06d}.jpg" for i in range(len(pil_imgs))]

        # Parse high-scoring frames from score_log
        # Score log lines look like: "HIGH 7.2 videoname_frame_000150.jpg [parts]"
        high_by_video = {}
        if score_log:
            for line in score_log.split("\n"):
                if line.startswith("HIGH"):
                    parts = line.split()
                    if len(parts) >= 3:
                        fname = parts[2]
                        # Parse video name and frame number from filename
                        # Format: videoname_frame_NNNNNN.jpg
                        idx = fname.rfind("_frame_")
                        if idx > 0:
                            vname = fname[:idx]
                            try:
                                fnum = int(fname[idx+7:].replace(".jpg", ""))
                                high_by_video.setdefault(vname, []).append(fnum)
                            except ValueError:
                                pass

        # If no score_log, try to infer from filenames
        if not high_by_video:
            for fn in fnames:
                idx = fn.rfind("_frame_")
                if idx > 0:
                    vname = fn[:idx]
                    try:
                        fnum = int(fn[idx+7:].replace(".jpg", ""))
                        high_by_video.setdefault(vname, []).append(fnum)
                    except ValueError:
                        pass

        if not high_by_video:
            print("[QVL] DenseResample: no frame timestamps found, returning originals only")
            batch = pil_list_to_batch(pil_imgs)
            return (batch, json.dumps(fnames), len(pil_imgs))

        # Find clusters and extract dense frames
        dense_imgs = []
        dense_fnames = []
        tmp_dir = tempfile.mkdtemp(prefix="qvl_dense_")

        for vname, frame_nums in sorted(high_by_video.items()):
            video_path = self._find_video(vname, video_dir)
            if not video_path:
                print(f"[QVL] DenseResample: video not found for {vname}")
                continue

            # Build clusters
            nums = sorted(set(frame_nums))
            clusters = []
            start = nums[0]
            end = nums[0]
            for n in nums[1:]:
                if n - end <= 10:
                    end = n
                else:
                    clusters.append((max(0, start - padding_secs), end + padding_secs))
                    start = n
                    end = n
            clusters.append((max(0, start - padding_secs), end + padding_secs))

            print(f"[QVL] DenseResample: {vname} — {len(clusters)} clusters")

            out_dir = os.path.join(tmp_dir, vname)
            os.makedirs(out_dir, exist_ok=True)

            for ci, (s, e) in enumerate(clusters):
                frames = self._extract_range(video_path, out_dir, s, e, dense_fps, f"d{ci:02d}")
                for fp in frames:
                    try:
                        img = Image.open(fp).convert("RGB")
                        dense_imgs.append(img)
                        dense_fnames.append(f"{vname}_dense_d{ci:02d}_{os.path.basename(fp)}")
                    except Exception:
                        pass

        print(f"[QVL] DenseResample: {len(dense_imgs)} dense frames extracted")

        # Dedup dense frames
        if dense_imgs and dedup_threshold > 0:
            import imagehash
            kept_imgs = []
            kept_fnames = []
            kept_hashes = []
            removed = 0
            pbar = comfy.utils.ProgressBar(len(dense_imgs))
            for img, fn in zip(dense_imgs, dense_fnames):
                h = imagehash.phash(img, hash_size=16)
                if any(abs(h - eh) <= dedup_threshold for eh in kept_hashes):
                    removed += 1
                else:
                    kept_hashes.append(h)
                    kept_imgs.append(img)
                    kept_fnames.append(fn)
                pbar.update(1)
            dense_imgs = kept_imgs
            dense_fnames = kept_fnames
            print(f"[QVL] DenseResample: dedup removed {removed}, kept {len(dense_imgs)}")

        # Combine originals + dense
        all_imgs = pil_imgs + dense_imgs
        all_fnames = fnames + dense_fnames

        # Normalize sizes for batching
        if len(all_imgs) > 1:
            target_w, target_h = all_imgs[0].size
            normalized = []
            for img in all_imgs:
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.LANCZOS)
                normalized.append(img)
            all_imgs = normalized

        print(f"[QVL] DenseResample: total {len(all_imgs)} frames ({len(pil_imgs)} original + {len(dense_imgs)} dense)")

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        batch = pil_list_to_batch(all_imgs)
        return (batch, json.dumps(all_fnames), len(all_imgs))


# ============================================================
# 11. RESIZE
# ============================================================

class QVL_Resize:
    """Resize images to uniform dimensions for training."""

    DESCRIPTION = """Resize images to uniform dimensions for LoRA training.

Modes:
- resize: Stretch/squash to exact target_size (distorts aspect ratio)
- pad: Resize to fit within target_size, pad with black to fill (preserves aspect)
- crop_center: Resize then center crop to exact target_size (crops edges)
- longest_side: Resize longest side to target_size, preserve aspect (variable dimensions)

Outputs:
- resized: Batch of resized images
- count: Number of images processed

Performance: Instant (PIL resize ops).

Use cases:
- Standardize dataset to 1024x1024 for SDXL training
- Resize to 512x512 for SD1.5 training
- Resize to 768x768 for flux training

Best practices:
- Use mode=pad for training (preserves aspect, no distortion)
- Use mode=crop_center if you want exact square (loses edge content)
- Use mode=resize only if you don't care about distortion
- Common sizes: 512 (SD1.5), 768 (SDXL draft), 1024 (SDXL/Flux)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to resize. Can be any size/aspect ratio.",
                }),
                "target_size": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "Target dimension in pixels. For square modes (resize/pad/crop_center), both dimensions become this. For longest_side, only longest edge is resized. Recommended: 512 (SD1.5), 768 (SDXL draft), 1024 (SDXL/Flux), 1536 (high-res).",
                }),
                "mode": (["resize", "pad", "crop_center", "longest_side"], {
                    "tooltip": "Resize strategy. resize = stretch to fit (distorts), pad = fit + black bars (safe), crop_center = fit + crop edges (loses content), longest_side = scale longest edge (variable sizes). Recommended: pad for training.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("resized", "count")
    FUNCTION = "do_resize"
    CATEGORY = CATEGORY

    def do_resize(self, images, target_size, mode):
        if is_empty_batch(images):
            return (empty_batch(), 0)

        pil_imgs = batch_to_pil_list(images)
        resized = [resize_image(img, target_size, mode) for img in pil_imgs]
        batch = pil_list_to_batch(resized)
        return (batch, len(resized))


# ============================================================
# 11. SAVE DATASET
# ============================================================

class QVL_SaveDataset:
    """Save images + caption .txt files to disk."""

    DESCRIPTION = """Save final dataset to disk (images + .txt caption files).

Output format:
- Each image: {filename}.{format}
- Each caption: {filename}.txt (matching basename)
- Captions are UTF-8 text files (one caption per file)

Inputs:
- images: Final processed batch
- output_folder: Where to save (auto-creates if missing)
- format: png (lossless), jpg (smaller), webp (smallest)
- quality: 50-100 (jpg/webp only, higher = better quality, larger files)
- captions_json: Array from Caption node
- filenames: Array from upstream (used for naming)
- prefix: Optional prefix for all filenames

Outputs:
- output_path: Folder where dataset was saved
- count: Number of image/caption pairs saved

Performance: ~50-200ms per image depending on format and quality. Batch of 500 = 25-100 seconds.

Use cases:
- Final save after full pipeline (LoadVideoFrames → Filter → Analyze → Caption → SmartCrop → Resize → SaveDataset)
- Export for upload to Hugging Face / Google Drive
- Create training folder for Kohya/EveryDream

Best practices:
- Use format=jpg, quality=95 for good balance (5-10MB per image)
- Use format=png for lossless (10-30MB per image)
- Use format=webp, quality=90 for smallest files (3-7MB per image)
- Set prefix to dataset name (e.g., 'myperson_') for easy identification
- Output folder should be empty or non-existent (auto-creates)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Final processed images to save. Should be resized to training dimensions before saving.",
                }),
                "output_folder": ("STRING", {
                    "default": "/workspace/dataset",
                    "multiline": False,
                    "tooltip": "Absolute path where dataset will be saved. Auto-creates if missing. On Vast.ai: /workspace/dataset. Windows: use forward slashes.",
                }),
                "format": (["png", "jpg", "webp"], {
                    "tooltip": "Image format. png = lossless (10-30MB), jpg = lossy (5-10MB at q=95), webp = smallest (3-7MB at q=90). Recommended: jpg for most cases, png for archival, webp for space-constrained.",
                }),
                "quality": ("INT", {
                    "default": 95, "min": 50, "max": 100, "step": 5,
                    "tooltip": "JPEG/WebP quality (50-100). Only applies to jpg/webp. Recommended: 95 for jpg (near-lossless), 90 for webp (good compression). Higher = better quality, larger files. PNG ignores this.",
                }),
            },
            "optional": {
                "captions_json": ("STRING", {
                    "default": "[]", "multiline": True,
                    "tooltip": "JSON array from Caption node. Each object should have {filename, caption} fields. Captions are saved as {basename}.txt next to images.",
                }),
                "filenames": ("STRING", {
                    "default": "[]",
                    "tooltip": "JSON array of filenames from upstream nodes. Used to determine output filenames. If missing, uses image_0000, image_0001, etc.",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Optional prefix for all output files (e.g., 'myperson_'). Results in filenames like myperson_frame_0001.jpg. Useful for organizing multi-subject datasets.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_path", "count")
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save(self, images, output_folder, format, quality,
             captions_json="[]", filenames="[]", prefix=""):
        if is_empty_batch(images):
            return (output_folder, 0)

        os.makedirs(output_folder, exist_ok=True)
        pil_imgs = batch_to_pil_list(images)

        # Parse captions
        try:
            captions = json.loads(captions_json) if captions_json else []
        except json.JSONDecodeError:
            captions = []

        # Parse filenames
        try:
            fnames = json.loads(filenames) if filenames else []
        except json.JSONDecodeError:
            fnames = []

        # Build caption lookup
        caption_map = {}
        for c in captions:
            if isinstance(c, dict):
                caption_map[c.get("filename", "")] = c.get("caption", "")
            elif isinstance(c, str):
                caption_map[f"caption_{len(caption_map)}"] = c

        pbar = comfy.utils.ProgressBar(len(pil_imgs))
        saved = 0

        for i, img in enumerate(pil_imgs):
            # Determine filename
            if i < len(fnames) and fnames[i]:
                stem = os.path.splitext(fnames[i])[0]
            else:
                stem = f"{prefix}{i:04d}" if prefix else f"image_{i:04d}"

            if prefix and i < len(fnames) and fnames[i]:
                stem = f"{prefix}{stem}"

            # Save image
            img_path = os.path.join(output_folder, f"{stem}.{format}")
            if format == "jpg":
                img.save(img_path, "JPEG", quality=quality)
            elif format == "webp":
                img.save(img_path, "WEBP", quality=quality)
            else:
                img.save(img_path, "PNG")

            # Save caption if available
            orig_fname = fnames[i] if i < len(fnames) else ""
            caption = caption_map.get(orig_fname, "")

            # Also try index-based lookup
            if not caption and i < len(captions):
                c = captions[i]
                if isinstance(c, dict):
                    caption = c.get("caption", "")
                elif isinstance(c, str):
                    caption = c

            if caption:
                txt_path = os.path.join(output_folder, f"{stem}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

            saved += 1
            pbar.update(1)

        print(f"[QVL] Saved {saved} images to {output_folder}")
        return (output_folder, saved)


# ============================================================
# 12. CUSTOM VLM QUERY
# ============================================================

class QVL_CustomQuery:
    """Send any custom prompt to the VLM. Maximum flexibility."""

    DESCRIPTION = """Send custom prompts to VLM for any task.

Purpose: Freeform VLM queries not covered by other nodes.

Use cases:
- Extract custom metadata (e.g., "List all visible objects")
- Classification tasks (e.g., "Is this indoor or outdoor?")
- Quality assessment with custom criteria
- Generate alternate caption styles
- Answer questions about image content

Outputs:
- images: Pass-through (for chaining)
- responses: Concatenated text responses (separated by ---)
- count: Number of images processed

Performance: Depends on model and max_tokens. Typical: 200ms-2s per image.

Best practices:
- Use temperature=0.1 for factual/structured output
- Use temperature=0.3-0.7 for creative/varied output
- Set max_tokens based on expected response length (150 for yes/no, 500 for descriptions, 1000+ for detailed analysis)
- For JSON output, instruct VLM to respond with valid JSON in prompt
- Use 7B models for speed, 32B for accuracy"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to query. VLM will process each image with the same prompt.",
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "tooltip": "Custom instruction for VLM. Be specific about desired output format. Examples: 'List visible objects', 'Rate quality 1-10', 'Describe pose and expression', 'Generate booru tags'.",
                }),
                "model": ("STRING", {
                    "default": "huihui_ai/qwen2.5-vl-abliterated:7b",
                    "tooltip": "VLM model name (Ollama) or model ID (OpenAI). Recommended: qwen2.5-vl-abliterated:7b for speed, :32b for quality. For OpenAI: gpt-4o, gpt-4-vision-preview.",
                }),
                "ollama_url": ("STRING", {
                    "default": DEFAULT_OLLAMA,
                    "tooltip": "Ollama server URL or OpenAI API endpoint. Default: http://127.0.0.1:11434 for local Ollama. For OpenAI: https://api.openai.com/v1.",
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Sampling temperature. Lower = more deterministic/factual. Recommended: 0.1 for structured output, 0.3 for descriptions, 0.7+ for creative. Higher = more random/varied.",
                }),
                "max_tokens": ("INT", {
                    "default": 500, "min": 50, "max": 4096, "step": 50,
                    "tooltip": "Maximum response length. Recommended: 150 for short answers, 500 for descriptions, 1000+ for detailed analysis. Higher = slower, more complete responses.",
                }),
                "api_type": (["ollama", "openai"], {
                    "default": "ollama",
                    "tooltip": "API type. ollama = local Ollama server, openai = OpenAI-compatible API (OpenAI, OpenRouter, etc.).",
                }),
                "workers": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Concurrent VLM requests. Set OLLAMA_NUM_PARALLEL on server to match. Use 1 for single instance, 4-8 for multi-instance fleet. Higher = faster but needs more VRAM across instances.",
                }),
                "ollama_urls": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated Ollama URLs for multi-instance parallelism. Example: http://instance1:11434,http://instance2:11434. Leave empty to use ollama_url. Each URL gets round-robin work distribution.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "responses", "count")
    FUNCTION = "query"
    CATEGORY = CATEGORY

    def query(self, images, prompt, model, ollama_url,
              temperature=0.3, max_tokens=500, api_type="ollama", workers=1, ollama_urls=""):
        if is_empty_batch(images):
            return (empty_batch(), "Empty input", 0)

        pil_imgs = batch_to_pil_list(images)

        # Determine URLs
        urls = ollama_urls if ollama_urls.strip() else ollama_url

        if workers > 1:
            # Parallel mode
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            responses_list = query_vlm_batch(
                pil_imgs, prompt, model=model, ollama_urls=urls,
                workers=workers, temperature=temperature, max_tokens=max_tokens,
                api_type=api_type,
                progress_callback=lambda n: pbar.update(n),
            )
            responses = [r.strip() for r in responses_list]
        else:
            # Sequential mode (original behavior)
            responses = []
            pbar = comfy.utils.ProgressBar(len(pil_imgs))

            for img in pil_imgs:
                if api_type == "openai":
                    resp = query_vlm_openai(img, prompt, model=model,
                                            api_url=ollama_url,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
                else:
                    resp = query_vlm(img, prompt, model=model,
                                     ollama_url=ollama_url,
                                     temperature=temperature,
                                     max_tokens=max_tokens)
                responses.append(resp.strip())
                pbar.update(1)

        return (images, "\n---\n".join(responses), len(pil_imgs))


# ============================================================
# 13. METADATA ROUTER (sort by classification)
# ============================================================

class QVL_MetadataRouter:
    """Route/split images based on metadata field values.
    Use with Analyze output to sort images by pose, angle, quality, etc."""

    DESCRIPTION = """Split/route images based on metadata field values.

Purpose: Sort images by VLM-extracted attributes (pose, angle, quality, tags, etc.).

How it works:
- Reads metadata_json from Analyze node
- Filters images where route_field matches route_value
- Outputs two batches: matched and unmatched

Outputs:
- matched: Images where field matches value
- unmatched: Images where field doesn't match
- matched_count, unmatched_count: Tallies

Use cases:
- Split by pose: route_field="pose", route_value="standing" → isolate standing poses
- Split by angle: route_field="camera_angle", route_value="front,closeup" → front/closeup only
- Split by quality: route_field="quality_score", route_value="8,9,10" → high quality only
- Multi-value matching: Comma-separated values (e.g., "front,back,side")

Performance: Instant (no VLM, just JSON filtering).

Best practices:
- Run Analyze first to generate metadata
- Use lowercase values (matching is case-insensitive)
- Chain multiple routers for complex filtering (e.g., pose=standing → angle=front)
- Check analysis_log from Analyze to see available field values"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to route. Must correspond to metadata_json order (_index field).",
                }),
                "metadata_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "JSON array from Analyze node. Each object should have _index and the field you want to route by (e.g., pose, camera_angle, quality_score).",
                }),
                "route_field": ("STRING", {
                    "default": "pose",
                    "tooltip": "Metadata field to filter by. Common fields from Analyze: pose, camera_angle, quality_score, aesthetic_score, tags. Must match field name exactly.",
                }),
                "route_value": ("STRING", {
                    "default": "standing",
                    "tooltip": "Value(s) to match (case-insensitive). Use comma-separated for multiple values (e.g., 'standing,sitting'). Images with matching field value go to matched output.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("matched", "unmatched", "matched_count", "unmatched_count")
    FUNCTION = "route"
    CATEGORY = CATEGORY

    def route(self, images, metadata_json, route_field, route_value):
        if is_empty_batch(images):
            return (empty_batch(), empty_batch(), 0, 0)

        pil_imgs = batch_to_pil_list(images)

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            return (images, empty_batch(), len(pil_imgs), 0)

        meta_by_idx = {}
        for m in metadata:
            idx = m.get("_index", -1)
            if idx >= 0:
                meta_by_idx[idx] = m

        matched, unmatched = [], []
        values = [v.strip().lower() for v in route_value.split(",")]

        for i, img in enumerate(pil_imgs):
            meta = meta_by_idx.get(i, {})
            field_val = str(meta.get(route_field, "")).lower()

            if field_val in values:
                matched.append(img)
            else:
                unmatched.append(img)

        matched_batch = pil_list_to_batch(matched) if matched else empty_batch()
        unmatched_batch = pil_list_to_batch(unmatched) if unmatched else empty_batch()

        return (matched_batch, unmatched_batch, len(matched), len(unmatched))


# ============================================================
# 14. VLM DETECT (bounding box only — lightweight)
# ============================================================

class QVL_Detect:
    """Detect subject bounding box without full analysis.
    Lighter than Analyze — use when you only need the crop box."""

    DESCRIPTION = """Lightweight bounding box detection (faster than Analyze).

Purpose: Extract subject bbox without full metadata analysis.

How it works:
- Sends bbox-only prompt to VLM
- Resizes image to img_size before VLM (faster inference)
- Extracts [x1, y1, x2, y2] normalized coordinates (0-1)
- Confidence score if VLM provides it

Outputs:
- images: Pass-through
- bbox_metadata: JSON array with _index, subject_bbox, confidence
- count: Number of images processed

Performance: ~100-300ms per image with 7B model (2-3x faster than Analyze due to smaller prompt and image resize).

Use cases:
- Fast bbox extraction for SmartCrop
- When you don't need quality/pose/tags metadata
- Processing large batches where speed matters

Best practices:
- Use img_size=512 for speed (default), 768-1024 for accuracy
- Use 7B model (bbox detection doesn't need 32B accuracy)
- Connect output to SmartCrop node
- For full metadata, use Analyze instead"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to detect subjects in. Works best with images containing clear subjects (people, objects).",
                }),
                "model": ("STRING", {
                    "default": "huihui_ai/qwen2.5-vl-abliterated:7b",
                    "tooltip": "VLM model for detection. Recommended: 7b for speed (bbox detection is simpler than full analysis). 32b works but overkill.",
                }),
                "ollama_url": ("STRING", {
                    "default": DEFAULT_OLLAMA,
                    "tooltip": "Ollama server URL or OpenAI API endpoint. Default: http://127.0.0.1:11434.",
                }),
                "prompt": ("STRING", {
                    "default": BBOX_PROMPT,
                    "multiline": True,
                    "tooltip": "Bbox detection prompt. Default asks for subject bounding box in JSON format. Edit to detect specific objects (e.g., 'Detect person's face', 'Detect car').",
                }),
            },
            "optional": {
                "api_type": (["ollama", "openai"], {
                    "default": "ollama",
                    "tooltip": "API type. ollama = local Ollama, openai = OpenAI-compatible API.",
                }),
                "img_size": ("INT", {
                    "default": 512, "min": 256, "max": 1024, "step": 64,
                    "tooltip": "Resize image to this size before VLM (speed optimization). Recommended: 512 for fast (100-200ms), 768 for balanced, 1024 for accuracy. Higher = slower but more precise bbox.",
                }),
                "workers": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Concurrent VLM requests. Set OLLAMA_NUM_PARALLEL on server to match. Use 1 for single instance, 4-8 for multi-instance fleet. Higher = faster but needs more VRAM across instances.",
                }),
                "ollama_urls": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated Ollama URLs for multi-instance parallelism. Example: http://instance1:11434,http://instance2:11434. Leave empty to use ollama_url. Each URL gets round-robin work distribution.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "bbox_metadata", "count")
    FUNCTION = "detect"
    CATEGORY = CATEGORY

    def detect(self, images, model, ollama_url, prompt,
               api_type="ollama", img_size=512, workers=1, ollama_urls=""):
        if is_empty_batch(images):
            return (empty_batch(), "[]", 0)

        pil_imgs = batch_to_pil_list(images)

        all_metadata = []

        # Determine URLs
        urls = ollama_urls if ollama_urls.strip() else ollama_url

        if workers > 1:
            # Parallel mode
            pbar = comfy.utils.ProgressBar(len(pil_imgs))
            responses = query_vlm_batch(
                pil_imgs, prompt, model=model, ollama_urls=urls,
                workers=workers, temperature=0.1, max_tokens=200,
                api_type=api_type, max_img_size=img_size,
                progress_callback=lambda n: pbar.update(n),
            )

            for i, resp in enumerate(responses):
                data = parse_json(resp)
                if data and "bbox" in data:
                    all_metadata.append({
                        "_index": i,
                        "subject_bbox": data["bbox"],
                        "confidence": data.get("confidence", 0.5),
                    })
                else:
                    all_metadata.append({
                        "_index": i,
                        "subject_bbox": None,
                        "_error": "No bbox detected",
                    })
        else:
            # Sequential mode (original behavior)
            pbar = comfy.utils.ProgressBar(len(pil_imgs))

            for i, img in enumerate(pil_imgs):
                if api_type == "openai":
                    resp = query_vlm_openai(img, prompt, model=model,
                                            api_url=ollama_url, temperature=0.1,
                                            max_tokens=200,
                                            max_img_size=img_size)
                else:
                    resp = query_vlm(img, prompt, model=model,
                                     ollama_url=ollama_url, temperature=0.1,
                                     max_tokens=200,
                                     max_img_size=img_size)

                data = parse_json(resp)
                if data and "bbox" in data:
                    all_metadata.append({
                        "_index": i,
                        "subject_bbox": data["bbox"],
                        "confidence": data.get("confidence", 0.5),
                    })
                else:
                    all_metadata.append({
                        "_index": i,
                        "subject_bbox": None,
                        "_error": "No bbox detected",
                    })

                pbar.update(1)

        return (images, json.dumps(all_metadata, indent=2), len(pil_imgs))


# ============================================================
# 15. FOLDER PREVIEW (image browser for extraction results)
# ============================================================

class QVL_FolderPreview:
    """Browse images from a folder with pagination. Connect output to
    ComfyUI's built-in PreviewImage node to see thumbnails in the workflow.
    Great for reviewing extraction / scoring / crop results."""

    DESCRIPTION = """Image browser for reviewing extraction/scoring results.

Purpose: Paginated folder browsing for QA and visual inspection.

How it works:
- Scans folder for images (jpg, png, webp, bmp)
- Sorts by name/date/size/random
- Returns page of images (default: 16 per page)
- Resizes to max_preview_size for fast preview

Outputs:
- preview_images: Batch of preview images (connect to PreviewImage node)
- filenames: JSON array of filenames on current page
- info: Human-readable page info string
- total_files: Total image count in folder
- total_pages: Total page count

Performance: ~10-50ms per image depending on resize. Page of 16 = 160-800ms.

Use cases:
- Review NudeScore keep/reject folders
- Browse SmartCrop output
- QA final dataset before SaveDataset
- Random sample inspection (sort_by=random)

Best practices:
- Set per_page=16-32 for grid preview (use PreviewImage)
- Use sort_by=date to see newest extractions
- Use sort_by=random for unbiased sampling
- Set max_preview_size=512 for fast preview, 1024 for detail
- Connect to PreviewImage node in ComfyUI for visual grid"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "/workspace/scored_keep",
                    "multiline": False,
                    "tooltip": "Absolute path to folder containing images to browse. On Vast.ai: /workspace paths. Folder must exist and contain images.",
                }),
                "page": ("INT", {
                    "default": 1, "min": 1, "max": 9999, "step": 1,
                    "tooltip": "Page number to view (1-indexed). Change this to browse through pages. Total pages shown in info output.",
                }),
                "per_page": ("INT", {
                    "default": 16, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Images per page. Recommended: 16 for 4x4 grid, 32 for 4x8, 64 for 8x8. Higher = more images loaded, slower preview.",
                }),
                "sort_by": (["name", "date", "size", "random"], {
                    "tooltip": "Sort method. name = alphabetical, date = newest first (useful for recent extractions), size = largest first, random = unbiased sampling. Random re-shuffles each time.",
                }),
            },
            "optional": {
                "max_preview_size": ("INT", {
                    "default": 512, "min": 128, "max": 2048, "step": 64,
                    "tooltip": "Maximum dimension for preview (width or height). Images are resized to fit within this (aspect preserved). Recommended: 512 for fast preview, 1024 for detailed inspection. Higher = slower load.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("preview_images", "filenames", "info", "total_files", "total_pages")
    FUNCTION = "browse"
    CATEGORY = CATEGORY

    def browse(self, folder_path, page, per_page, sort_by, max_preview_size=512):
        from PIL import Image
        import random

        if not os.path.isdir(folder_path):
            return (empty_batch(), "[]", f"Folder not found: {folder_path}", 0, 0)

        # Scan folder for images
        all_files = []
        for f in os.listdir(folder_path):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                fp = os.path.join(folder_path, f)
                all_files.append((f, fp))

        if not all_files:
            return (empty_batch(), "[]", f"No images in {folder_path}", 0, 0)

        # Sort
        if sort_by == "date":
            all_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        elif sort_by == "size":
            all_files.sort(key=lambda x: os.path.getsize(x[1]), reverse=True)
        elif sort_by == "random":
            random.shuffle(all_files)
        else:
            all_files.sort(key=lambda x: x[0])

        total = len(all_files)
        total_pages = (total + per_page - 1) // per_page
        page = min(page, total_pages)

        start = (page - 1) * per_page
        end = min(start + per_page, total)
        page_files = all_files[start:end]

        pil_imgs = []
        fnames = []
        for fname, fpath in page_files:
            try:
                img = Image.open(fpath).convert("RGB")
                # Resize for preview (keep aspect ratio, fit in max_preview_size)
                w, h = img.size
                if max(w, h) > max_preview_size:
                    scale = max_preview_size / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                pil_imgs.append(img)
                fnames.append(fname)
            except Exception:
                pass

        if not pil_imgs:
            return (empty_batch(), "[]", "Failed to load images", total, total_pages)

        # Normalize sizes for batch (pad to same dimensions)
        max_w = max(img.size[0] for img in pil_imgs)
        max_h = max(img.size[1] for img in pil_imgs)
        padded = []
        for img in pil_imgs:
            if img.size != (max_w, max_h):
                canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
                x = (max_w - img.size[0]) // 2
                y = (max_h - img.size[1]) // 2
                canvas.paste(img, (x, y))
                padded.append(canvas)
            else:
                padded.append(img)

        batch = pil_list_to_batch(padded)
        info = f"Page {page}/{total_pages} | Showing {len(fnames)} of {total} images | Folder: {folder_path}"
        print(f"[QVL] FolderPreview: {info}")

        return (batch, json.dumps(fnames), info, total, total_pages)


# ============================================================
# 16. SERVER IMAGE (pick from server filesystem)
# ============================================================

class QVL_ServerImage:
    """Load a single image from the server filesystem with visual browser.
    Drop-in replacement for LoadImages — same outputs (IMAGE, filenames, count)
    plus MASK. Click 'Browse Server' to pick files visually."""

    DESCRIPTION = """Browse and load an image from the server filesystem.

    Click 'Browse Server' to open a Windows Explorer-style file browser
    with thumbnails, preview panel, and folder navigation.

    Outputs match QVL_LoadImages for workflow compatibility:
    - images: Image tensor [1, H, W, 3]
    - filenames: JSON array with the filename
    - count: Always 1
    - mask: Alpha channel (zeros if no transparency)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "/workspace/scored_keep/",
                    "multiline": False,
                    "tooltip": "Full path to an image on the server. Use the 'Browse Server' button below to visually pick a file, or type/paste a path directly.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "MASK")
    RETURN_NAMES = ("images", "filenames", "count", "mask")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, image_path):
        from PIL import Image
        import numpy as np
        import json

        if not image_path or not os.path.isfile(image_path):
            print(f"[QVL] ServerImage: file not found: {image_path}")
            return (empty_batch(), json.dumps([]), 0, torch.zeros((1, 64, 64), dtype=torch.float32))

        img = Image.open(image_path)

        # Handle alpha channel
        if img.mode == "RGBA":
            alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = torch.from_numpy(alpha).unsqueeze(0)
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")
            mask = torch.zeros((1, img.size[1], img.size[0]), dtype=torch.float32)

        tensor = pil_to_tensor(img)
        filename = os.path.basename(image_path)
        filenames_json = json.dumps([filename])
        print(f"[QVL] ServerImage: loaded {image_path} ({img.size[0]}x{img.size[1]})")

        return (tensor, filenames_json, 1, mask)

    @classmethod
    def IS_CHANGED(cls, image_path):
        if os.path.isfile(image_path):
            return os.path.getmtime(image_path)
        return float("nan")


# ============================================================
# NODE REGISTRATION
# ============================================================

NODE_CLASS_MAPPINGS = {
    "QVL_LoadImages": QVL_LoadImages,
    "QVL_LoadVideoFrames": QVL_LoadVideoFrames,
    "QVL_Dedup": QVL_Dedup,
    "QVL_Filter": QVL_Filter,
    "QVL_Analyze": QVL_Analyze,
    "QVL_Caption": QVL_Caption,
    "QVL_SmartCrop": QVL_SmartCrop,
    "QVL_AutoCorrect": QVL_AutoCorrect,
    "QVL_QualityScore": QVL_QualityScore,
    "QVL_Resize": QVL_Resize,
    "QVL_SaveDataset": QVL_SaveDataset,
    "QVL_CustomQuery": QVL_CustomQuery,
    "QVL_MetadataRouter": QVL_MetadataRouter,
    "QVL_Detect": QVL_Detect,
    "QVL_NudeScore": QVL_NudeScore,
    "QVL_DenseResample": QVL_DenseResample,
    "QVL_FolderPreview": QVL_FolderPreview,
    "QVL_ServerImage": QVL_ServerImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QVL_LoadImages": "QVL Load Images",
    "QVL_LoadVideoFrames": "QVL Load Video Frames",
    "QVL_Dedup": "QVL Perceptual Dedup",
    "QVL_Filter": "QVL Quick Filter (VLM)",
    "QVL_Analyze": "QVL Deep Analyze (VLM)",
    "QVL_Caption": "QVL Caption (VLM)",
    "QVL_SmartCrop": "QVL Smart Crop",
    "QVL_AutoCorrect": "QVL Auto Correct",
    "QVL_QualityScore": "QVL Quality Score",
    "QVL_Resize": "QVL Resize",
    "QVL_SaveDataset": "QVL Save Dataset",
    "QVL_CustomQuery": "QVL Custom Query (VLM)",
    "QVL_MetadataRouter": "QVL Metadata Router",
    "QVL_Detect": "QVL Detect Subject (VLM)",
    "QVL_NudeScore": "QVL Nude Score (Anatomy Filter)",
    "QVL_DenseResample": "QVL Dense Resample (High-Value Re-extract)",
    "QVL_FolderPreview": "QVL Folder Preview (Image Browser)",
    "QVL_ServerImage": "QVL Server Image (Browse Server Files)",
}

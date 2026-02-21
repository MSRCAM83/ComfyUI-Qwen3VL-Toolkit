#!/usr/bin/env python3
"""
Run Pipeline — End-to-end LoRA dataset creation with fleet parallelism.

Launches Klein + VLM fleets, generates workflow, queues it, monitors completion.
One command to go from raw video → finished training dataset.

Usage:
    python run_pipeline.py --input /workspace/videos/ --output /workspace/dataset/
    python run_pipeline.py --input /workspace/images/ --output /workspace/dataset/ --klein-instances 20 --vlm-instances 20
    python run_pipeline.py --help

Cost estimate (20+20 instances, ~$0.15/hr each):
    Fleet: ~$6.00/hr total
    15,000 images: ~45 min = ~$4.50 total
"""

import argparse
import json
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error

def run_fleet(args_list):
    """Run fleet.py command."""
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "fleet.py")] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)  # 15 min max
    return result.stdout, result.stderr, result.returncode

def get_fleet_state():
    """Load fleet state."""
    state_file = os.path.join(os.path.dirname(__file__), ".fleet_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return None

def wait_comfyui_ready(url, timeout=60):
    """Wait for main ComfyUI instance to be responsive."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{url}/system_stats", timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def generate_workflow_api(comfyui_url, klein_urls, vlm_urls, input_path, output_path, prompt, caption_preset, vlm_model, workers):
    """Generate the API-format workflow with fleet URLs baked in.

    This builds the workflow directly in API format (dict of node_id -> {class_type, inputs}).
    It uses the fleet URLs so the workflow runs distributed across all instances.
    """

    # Determine input type based on path
    is_video = any(input_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm'])
    input_is_dir = not is_video  # directory of images vs single video

    prompt_dict = {}

    # --- SECTION 1: INPUT ---
    if is_video or input_is_dir:
        # Use LoadVideoFrames for video, ServerImage for directory
        if is_video:
            prompt_dict["1"] = {
                "class_type": "QVL_LoadVideoFrames",
                "inputs": {
                    "video": input_path,
                    "extraction_mode": "scene_detect",
                    "fps": 1.0,
                    "scene_threshold": 0.1,
                    "max_frames": 5000,
                    "dedup_on_extract": True,
                    "dedup_threshold": 2,
                }
            }
        else:
            prompt_dict["1"] = {
                "class_type": "QVL_ServerImage",
                "inputs": {
                    "image_path": input_path,
                }
            }

    # --- SECTION 2: QUALITY GATE ---
    prompt_dict["10"] = {
        "class_type": "QVL_QualityScore",
        "inputs": {
            "images": ["1", 0],
            "min_score": 25.0,
        }
    }

    prompt_dict["11"] = {
        "class_type": "QVL_Filter",
        "inputs": {
            "images": ["10", 0],
            "filenames": ["10", 1] if not is_video else ["1", 1],
            "model": vlm_model,
            "ollama_url": vlm_urls.split(",")[0] if vlm_urls else "http://localhost:11434",
            "api_type": "ollama",
            "threshold": 0.5,
            "workers": workers,
            "ollama_urls": vlm_urls,
        }
    }

    # --- SECTION 3: DEDUP ---
    prompt_dict["20"] = {
        "class_type": "QVL_Dedup",
        "inputs": {
            "images": ["11", 0],
            "filenames": ["11", 1],
            "threshold": 4,
            "method": "phash",
        }
    }

    # --- SECTION 4: KLEIN FLEET ---
    if klein_urls:
        prompt_dict["30"] = {
            "class_type": "QVL_KleinFleet",
            "inputs": {
                "images": ["20", 0],
                "filenames": ["20", 1],
                "comfyui_urls": klein_urls,
                "prompt": prompt,
                "workers": workers,
                "unet_name": "bigLove_klein1_fp8.safetensors",
                "clip_name": "qwen3_8b_abliterated_v2-fp8mixed.safetensors",
                "vae_name": "flux2-vae.safetensors",
                "steps": 3,
                "cfg": 1.0,
                "width": 1024,
                "height": 1024,
                "megapixels": 1.0,
            }
        }
        caption_source = "30"
    else:
        caption_source = "20"

    # --- SECTION 5: CAPTION ---
    prompt_dict["40"] = {
        "class_type": "QVL_Caption",
        "inputs": {
            "images": [caption_source, 0],
            "filenames": [caption_source, 1],
            "model": vlm_model,
            "ollama_url": vlm_urls.split(",")[0] if vlm_urls else "http://localhost:11434",
            "preset": caption_preset,
            "max_resolution": 512,
            "max_tokens": 150,
            "temperature": 0.3,
            "batch_size": 1,
            "api_type": "ollama",
            "workers": workers,
            "ollama_urls": vlm_urls,
        }
    }

    # --- SECTION 6: RESIZE + SAVE ---
    prompt_dict["50"] = {
        "class_type": "QVL_Resize",
        "inputs": {
            "images": ["40", 0],
            "target_size": 1024,
            "mode": "cover",
            "keep_aspect": True,
        }
    }

    prompt_dict["60"] = {
        "class_type": "QVL_SaveDataset",
        "inputs": {
            "images": ["50", 0],
            "captions": ["40", 1],
            "filenames": ["40", 2],
            "output_dir": output_path,
            "format": "kohya",
            "save_captions": True,
            "overwrite": True,
        }
    }

    return prompt_dict

def queue_workflow(comfyui_url, prompt_dict):
    """Queue workflow on ComfyUI and return prompt_id."""
    data = json.dumps({"prompt": prompt_dict}).encode()
    req = urllib.request.Request(
        f"{comfyui_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    return result.get("prompt_id")

def monitor_progress(comfyui_url, prompt_id, poll_interval=5):
    """Monitor workflow execution until complete."""
    print(f"Monitoring prompt {prompt_id}...")
    start = time.time()

    while True:
        try:
            resp = urllib.request.urlopen(f"{comfyui_url}/history/{prompt_id}", timeout=10)
            history = json.loads(resp.read())

            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False) or status.get("status_str") == "success":
                    elapsed = time.time() - start
                    print(f"\nPipeline COMPLETE in {elapsed/60:.1f} minutes")

                    # Print output summary
                    outputs = entry.get("outputs", {})
                    for node_id, node_out in outputs.items():
                        if "images" in node_out:
                            count = len(node_out["images"])
                            print(f"  Node {node_id}: {count} output images")
                    return True

                if status.get("status_str") == "error":
                    print(f"\nPipeline FAILED:")
                    messages = status.get("messages", [])
                    for msg in messages:
                        print(f"  {msg}")
                    return False
        except urllib.error.HTTPError:
            pass  # Not in history yet
        except Exception as e:
            pass

        elapsed = int(time.time() - start)
        # Try to get queue position
        try:
            resp = urllib.request.urlopen(f"{comfyui_url}/queue", timeout=5)
            queue = json.loads(resp.read())
            running = len(queue.get("queue_running", []))
            pending = len(queue.get("queue_pending", []))
            print(f"  [{elapsed}s] Running: {running}, Pending: {pending}", end="\r")
        except Exception:
            print(f"  [{elapsed}s] Waiting...", end="\r")

        time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser(
        description="Run Pipeline — End-to-end LoRA dataset creation with fleet parallelism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with 10+10 fleet
  python run_pipeline.py --input /workspace/videos/ --output /workspace/dataset/

  # Large scale with auto-cleanup
  python run_pipeline.py --input /workspace/images/ --output /workspace/dataset/ \\
    --klein-instances 20 --vlm-instances 20 --auto-destroy

  # Skip Klein (caption only)
  python run_pipeline.py --input /workspace/images/ --output /workspace/dataset/ \\
    --no-klein --vlm-instances 15

  # Use existing fleet (already launched)
  python run_pipeline.py --input /workspace/images/ --output /workspace/dataset/ \\
    --use-existing-fleet
        """
    )

    parser.add_argument("--input", required=True, help="Input path (video file or image directory)")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--comfyui", default="http://localhost:18188", help="Main ComfyUI URL")
    parser.add_argument("--klein-instances", type=int, default=10, help="Number of Klein instances (default: 10)")
    parser.add_argument("--vlm-instances", type=int, default=10, help="Number of VLM instances (default: 10)")
    parser.add_argument("--vlm-model", default="huihui_ai/qwen2.5-vl-abliterated:32b", help="VLM model name")
    parser.add_argument("--model-size", choices=["7b", "32b"], default="32b", help="VLM model size for fleet")
    parser.add_argument("--prompt", default="solo woman, sharp focus, professional studio photograph, soft studio lighting, clean neutral background, high resolution DSLR photograph, pristine image quality, no artifacts, no noise", help="Klein edit prompt")
    parser.add_argument("--caption-preset", default="flux2", help="Caption preset (flux, flux2, sdxl, booru, pony, natural, chroma, structured)")
    parser.add_argument("--workers", type=int, default=4, help="Workers per instance (default: 4)")
    parser.add_argument("--no-klein", action="store_true", help="Skip Klein editing stage")
    parser.add_argument("--auto-destroy", action="store_true", help="Destroy fleet when pipeline completes")
    parser.add_argument("--use-existing-fleet", action="store_true", help="Use already-running fleet instead of launching new")
    parser.add_argument("--dry-run", action="store_true", help="Generate workflow but don't queue it")

    args = parser.parse_args()

    print("=" * 60)
    print("LoRA DATASET PIPELINE — Fleet Mode")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Klein:  {'DISABLED' if args.no_klein else f'{args.klein_instances} instances'}")
    print(f"VLM:    {args.vlm_instances} instances ({args.model_size})")
    print(f"Preset: {args.caption_preset}")
    print()

    # Step 1: Launch or verify fleet
    klein_urls = ""
    vlm_urls = ""

    if args.use_existing_fleet:
        print("Using existing fleet...")
        state = get_fleet_state()
        if not state or state.get("status") != "running":
            print("ERROR: No running fleet found. Launch one first or remove --use-existing-fleet")
            sys.exit(1)
        klein_urls = ",".join(i["url"] for i in state.get("klein_instances", []) if i.get("url"))
        vlm_urls = ",".join(i["url"] for i in state.get("vlm_instances", []) if i.get("url"))
    else:
        # Launch fleet
        launch_args = []
        if args.no_klein:
            launch_args += ["launch", "--vlm-only", "--vlm", str(args.vlm_instances), "--model", args.model_size]
        else:
            launch_args += ["launch", "--klein", str(args.klein_instances), "--vlm", str(args.vlm_instances), "--model", args.model_size]

        print("Launching fleet...")
        stdout, stderr, rc = run_fleet(launch_args)
        print(stdout)
        if rc != 0:
            print(f"Fleet launch failed: {stderr}")
            sys.exit(1)

        state = get_fleet_state()
        if not state:
            print("ERROR: Fleet state not found after launch")
            sys.exit(1)

        klein_urls = ",".join(i["url"] for i in state.get("klein_instances", []) if i.get("url"))
        vlm_urls = ",".join(i["url"] for i in state.get("vlm_instances", []) if i.get("url"))

    if args.no_klein:
        klein_urls = ""

    klein_count = len(klein_urls.split(",")) if klein_urls else 0
    vlm_count = len(vlm_urls.split(",")) if vlm_urls else 0
    print(f"\nFleet ready: {klein_count} Klein + {vlm_count} VLM instances")

    # Step 2: Verify main ComfyUI
    print(f"\nChecking main ComfyUI at {args.comfyui}...")
    if not wait_comfyui_ready(args.comfyui):
        print("ERROR: Main ComfyUI not responding. Is it running?")
        sys.exit(1)
    print("  ComfyUI OK")

    # Step 3: Generate workflow
    print("\nGenerating pipeline workflow...")
    prompt_dict = generate_workflow_api(
        comfyui_url=args.comfyui,
        klein_urls=klein_urls,
        vlm_urls=vlm_urls,
        input_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        caption_preset=args.caption_preset,
        vlm_model=args.vlm_model,
        workers=args.workers,
    )

    print(f"  Generated {len(prompt_dict)} nodes")

    # Save for inspection
    workflow_path = os.path.join(os.path.dirname(__file__), "pipeline_api_workflow.json")
    with open(workflow_path, "w") as f:
        json.dump(prompt_dict, f, indent=2)
    print(f"  Saved to {workflow_path}")

    if args.dry_run:
        print("\nDRY RUN — workflow saved but not queued")
        print(json.dumps(prompt_dict, indent=2))
        return

    # Step 4: Queue workflow
    print(f"\nQueuing workflow on {args.comfyui}...")
    try:
        prompt_id = queue_workflow(args.comfyui, prompt_dict)
        print(f"  Queued: prompt_id={prompt_id}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  Queue FAILED (HTTP {e.code}):")
        try:
            err = json.loads(body)
            if "node_errors" in err:
                for nid, nerr in err["node_errors"].items():
                    ct = prompt_dict.get(nid, {}).get("class_type", "?")
                    print(f"    Node {nid} ({ct}):")
                    for msg in nerr.get("errors", []):
                        print(f"      - {msg.get('message', str(msg))}")
            if "error" in err:
                print(f"    Error: {err['error']}")
        except Exception:
            print(f"    Raw: {body[:500]}")
        sys.exit(1)

    # Step 5: Monitor
    success = monitor_progress(args.comfyui, prompt_id)

    # Step 6: Cleanup
    if args.auto_destroy:
        print("\nDestroying fleet (--auto-destroy)...")
        stdout, stderr, rc = run_fleet(["destroy"])
        print(stdout)
    else:
        print("\nFleet still running. Run 'python fleet.py destroy' when done.")

    if success:
        print(f"\nDataset saved to: {args.output}")
        print("Done!")
    else:
        print("\nPipeline failed. Check ComfyUI logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()

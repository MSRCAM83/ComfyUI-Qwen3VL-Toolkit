#!/usr/bin/env python3
"""
Fleet Klein — Spin up N Vast.ai instances for parallel image editing with Klein.

Usage:
    python fleet_klein.py launch --count 4
    python fleet_klein.py status
    python fleet_klein.py urls
    python fleet_klein.py process --input_dir /path/to/images --output_dir /path/to/output --prompt "your prompt" --workers 4
    python fleet_klein.py destroy

Workflow:
    1. Run 'launch' to rent N instances with Klein pre-installed (Docker image)
    2. Run 'process' to upload images and run Klein editing across the fleet
    3. Run 'destroy' when done to stop billing

Cost estimate (Vast.ai RTX 3090, ~$0.15-0.30/hr each):
    4 instances = ~$0.60-1.20/hr
    1000 images at 3 steps ≈ 20-40 min = ~$0.40-0.80 total
"""

import json
import os
import sys
import time
import subprocess
import argparse
import requests
import random
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

FLEET_STATE = os.path.join(os.path.dirname(__file__), ".fleet_klein_state.json")

# Vast.ai search criteria for Klein instances
# Need RTX 3090+ for Klein (fp8 inference needs 22GB+ VRAM)
VAST_SEARCH = {
    "gpu_ram": ">=22",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=40",
    "dph_total": "<=0.30",  # Max $0.30/hr
    "gpu_name": ["RTX_3090", "RTX_4090", "RTX_A6000", "A100_SXM4"],
}

# Docker image with Klein pre-installed
DOCKER_IMAGE = "msrcam/klein-fleet:latest"

# Onstart command — everything is in the Docker image
ONSTART_CMD = "/workspace/start.sh"


def load_state():
    if os.path.exists(FLEET_STATE):
        with open(FLEET_STATE) as f:
            return json.load(f)
    return {"instances": [], "status": "idle"}


def save_state(state):
    with open(FLEET_STATE, "w") as f:
        json.dump(state, f, indent=2)


def run_vast(args_list):
    """Run vastai CLI command and return output."""
    cmd = ["vastai"] + args_list
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout.strip(), result.returncode
    except FileNotFoundError:
        print("ERROR: 'vastai' CLI not found. Install with: pip install vastai")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        return "timeout", 1


def search_offers():
    """Search Vast.ai for suitable instances."""
    query_parts = []
    for key, val in VAST_SEARCH.items():
        if isinstance(val, list):
            query_parts.append(f"{key} in [{','.join(val)}]")
        else:
            query_parts.append(f"{key} {val}")

    query = " ".join(query_parts)
    out, rc = run_vast(["search", "offers", "--raw", query])
    if rc != 0:
        print(f"Search failed: {out}")
        return []

    try:
        offers = json.loads(out)
        # Sort by price
        offers.sort(key=lambda x: x.get("dph_total", 999))
        return offers
    except json.JSONDecodeError:
        print(f"Failed to parse offers: {out[:200]}")
        return []


def _parse_instance_id(out):
    """Parse instance ID from vastai create output."""
    import re
    m = re.search(r"'new_contract':\s*(\d+)", out)
    if m:
        return int(m.group(1))
    m = re.search(r'"new_contract":\s*(\d+)', out)
    if m:
        return int(m.group(1))
    for word in out.split():
        if word.isdigit():
            return int(word)
    return None


def launch_instance(offer_id):
    """Rent a single instance and set it up."""
    out, rc = run_vast(["create", "instance", str(offer_id),
                        "--image", DOCKER_IMAGE,
                        "--disk", "40",
                        "--onstart-cmd", ONSTART_CMD])
    if rc != 0:
        return None, f"Create failed: {out}"

    iid = _parse_instance_id(out)
    if iid:
        return iid, "ok"
    return None, f"Could not parse instance ID from: {out}"


def get_instance_info(instance_id):
    """Get instance status and IP."""
    out, rc = run_vast(["show", "instance", str(instance_id), "--raw"])
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def check_comfyui_ready(url):
    """Check if ComfyUI is ready by hitting /system_stats."""
    try:
        resp = requests.get(f"{url}/system_stats", timeout=5)
        return resp.status_code == 200
    except:
        return False


def wait_for_ready(instance_ids, timeout=900):
    """Wait for all instances to be running and ComfyUI ready."""
    start = time.time()
    ready = set()
    urls = {}

    print("Waiting for instances to boot and ComfyUI to start...")
    print("(This takes 3-10 minutes for Docker pull + startup)")

    while time.time() - start < timeout and len(ready) < len(instance_ids):
        for iid in instance_ids:
            if iid in ready:
                continue

            info = get_instance_info(iid)
            if not info:
                continue

            status = info.get("actual_status", info.get("status_msg", ""))
            if status == "running":
                # Get the public IP and port
                ip = info.get("public_ipaddr", "")
                ports = info.get("ports", {})

                # Find the ComfyUI port mapping (18188)
                comfy_port = None
                if "18188/tcp" in ports:
                    port_info = ports["18188/tcp"]
                    if isinstance(port_info, list) and port_info:
                        comfy_port = port_info[0].get("HostPort")
                    elif isinstance(port_info, dict):
                        comfy_port = port_info.get("HostPort")

                if ip and comfy_port:
                    url = f"http://{ip}:{comfy_port}"
                    # Check if ComfyUI is actually responding
                    if check_comfyui_ready(url):
                        urls[iid] = url
                        ready.add(iid)
                        print(f"  Instance {iid}: READY at {url}")

        if len(ready) < len(instance_ids):
            remaining = len(instance_ids) - len(ready)
            elapsed = int(time.time() - start)
            print(f"  Waiting... {len(ready)}/{len(instance_ids)} ready ({elapsed}s elapsed)")
            time.sleep(10)

    return urls


def cmd_launch(args):
    """Launch N instances."""
    count = args.count

    print(f"Searching for {count} instances for Klein...")
    offers = search_offers()

    if len(offers) < count:
        print(f"WARNING: Only {len(offers)} offers found, need {count}")
        count = min(count, len(offers))

    if count == 0:
        print("No suitable instances found. Try relaxing search criteria.")
        return

    # Show cost estimate
    total_cost = sum(o.get("dph_total", 0) for o in offers[:count])
    print(f"Renting {count} instances at ~${total_cost:.2f}/hr total")
    print(f"Docker image: {DOCKER_IMAGE}")
    print()

    # Launch instances
    instance_ids = []
    for i, offer in enumerate(offers[:count]):
        offer_id = offer["id"]
        gpu = offer.get("gpu_name", "?")
        price = offer.get("dph_total", 0)
        print(f"  [{i+1}/{count}] Launching on {gpu} (${price:.3f}/hr)...")

        iid, msg = launch_instance(offer_id)
        if iid:
            instance_ids.append(iid)
            print(f"    Instance {iid}: created")
        else:
            print(f"    FAILED: {msg}")

    if not instance_ids:
        print("No instances launched successfully.")
        return

    print(f"\nWaiting for {len(instance_ids)} instances to boot + start ComfyUI...")

    urls = wait_for_ready(instance_ids)

    # Save state
    state = {
        "instances": [{"id": iid, "url": urls.get(iid, "")} for iid in instance_ids],
        "status": "running",
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_state(state)

    print(f"\n{'='*60}")
    print(f"FLEET READY — {len(urls)}/{len(instance_ids)} instances online")
    print(f"{'='*60}")
    url_list = ",".join(urls.values())
    print(f"\nComfyUI URLs:")
    print(f"  {url_list}")
    print(f"\nRun 'python fleet_klein.py process' to start editing images!")
    print(f"\nRun 'python fleet_klein.py destroy' when done!")


def cmd_status(args):
    """Show fleet status."""
    state = load_state()
    if state["status"] == "idle":
        print("No fleet running.")
        return

    print(f"Fleet status: {state['status']}")
    print(f"Launched: {state.get('launched_at', '?')}")
    print(f"Instances: {len(state['instances'])}")

    for inst in state["instances"]:
        iid = inst["id"]
        info = get_instance_info(iid)
        status = "unknown"
        if info:
            status = info.get("actual_status", info.get("status_msg", "?"))
        print(f"  {iid}: {status} — {inst.get('url', 'no url')}")


def cmd_urls(args):
    """Print comma-separated URLs for ComfyUI."""
    state = load_state()
    urls = [inst["url"] for inst in state.get("instances", []) if inst.get("url")]
    if urls:
        print(",".join(urls))
    else:
        print("No URLs available. Run 'launch' first.")


def cmd_destroy(args):
    """Destroy all fleet instances."""
    state = load_state()
    instances = state.get("instances", [])

    if not instances:
        print("No instances to destroy.")
        return

    print(f"Destroying {len(instances)} instances...")
    for inst in instances:
        iid = inst["id"]
        out, rc = run_vast(["destroy", "instance", str(iid)])
        if rc == 0:
            print(f"  {iid}: destroyed")
        else:
            print(f"  {iid}: destroy failed — {out}")

    state["status"] = "idle"
    state["instances"] = []
    save_state(state)
    print("Fleet destroyed. Billing stopped.")


def build_klein_workflow(prompt, input_image_name, seed=None):
    """Build Klein workflow API payload."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # ComfyUI API format workflow
    workflow = {
        "1": {
            "inputs": {
                "unet_name": "bigLove_klein1_fp8.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            },
            "class_type": "UNETLoader"
        },
        "2": {
            "inputs": {
                "clip_name": "qwen3_8b_abliterated_v2-fp8mixed.safetensors",
                "type": "flux2"
            },
            "class_type": "CLIPLoader"
        },
        "3": {
            "inputs": {
                "vae_name": "flux2-vae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "4": {
            "inputs": {
                "text": prompt,
                "clip": ["2", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {
                "conditioning": ["4", 0]
            },
            "class_type": "ConditioningZeroOut"
        },
        "6": {
            "inputs": {
                "image": input_image_name,
                "upload": "image"
            },
            "class_type": "LoadImage"
        },
        "7": {
            "inputs": {
                "image": ["6", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0
            },
            "class_type": "ImageScaleToTotalPixels"
        },
        "8": {
            "inputs": {
                "pixels": ["7", 0],
                "vae": ["3", 0]
            },
            "class_type": "VAEEncode"
        },
        "9": {
            "inputs": {
                "conditioning": ["4", 0],
                "latent": ["8", 0]
            },
            "class_type": "ReferenceLatent"
        },
        "10": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyFlux2LatentImage"
        },
        "11": {
            "inputs": {
                "noise_seed": seed
            },
            "class_type": "RandomNoise"
        },
        "12": {
            "inputs": {
                "sampler_name": "euler"
            },
            "class_type": "KSamplerSelect"
        },
        "13": {
            "inputs": {
                "steps": 3,
                "width": 1024,
                "height": 1024
            },
            "class_type": "Flux2Scheduler"
        },
        "14": {
            "inputs": {
                "model": ["1", 0],
                "positive": ["9", 0],
                "negative": ["5", 0],
                "cfg": 1.0
            },
            "class_type": "CFGGuider"
        },
        "15": {
            "inputs": {
                "noise": ["11", 0],
                "guider": ["14", 0],
                "sampler": ["12", 0],
                "sigmas": ["13", 0],
                "latent_image": ["10", 0]
            },
            "class_type": "SamplerCustomAdvanced"
        },
        "16": {
            "inputs": {
                "samples": ["15", 0],
                "vae": ["3", 0]
            },
            "class_type": "VAEDecode"
        },
        "17": {
            "inputs": {
                "images": ["16", 0],
                "filename_prefix": "klein_output"
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def upload_image(url, image_path):
    """Upload image to ComfyUI instance."""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/png')}
            resp = requests.post(f"{url}/upload/image", files=files, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                return result.get('name', os.path.basename(image_path))
            else:
                print(f"Upload failed for {image_path}: {resp.status_code}")
                return None
    except Exception as e:
        print(f"Upload error for {image_path}: {e}")
        return None


def queue_prompt(url, workflow):
    """Queue a prompt on ComfyUI instance."""
    try:
        payload = {"prompt": workflow}
        resp = requests.post(f"{url}/prompt", json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result.get('prompt_id')
        else:
            print(f"Queue failed: {resp.status_code}")
            return None
    except Exception as e:
        print(f"Queue error: {e}")
        return None


def wait_for_completion(url, prompt_id, timeout=300):
    """Wait for prompt to complete and return output info."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/history/{prompt_id}", timeout=10)
            if resp.status_code == 200:
                history = resp.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get('outputs', {})
                    # Look for SaveImage node output (node 17)
                    if '17' in outputs:
                        images = outputs['17'].get('images', [])
                        if images:
                            return images[0]  # {filename, subfolder, type}
            time.sleep(2)
        except Exception as e:
            print(f"History check error: {e}")
            time.sleep(2)

    return None


def download_output(url, image_info, output_path):
    """Download output image from ComfyUI."""
    try:
        params = {
            'filename': image_info['filename'],
            'subfolder': image_info.get('subfolder', ''),
            'type': image_info.get('type', 'output')
        }
        resp = requests.get(f"{url}/view", params=params, timeout=30)
        if resp.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            return True
        else:
            print(f"Download failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Download error: {e}")
        return False


def process_image_on_instance(url, image_path, prompt, output_dir):
    """Process a single image on a single instance."""
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"klein_{image_name}")

    # Upload image
    uploaded_name = upload_image(url, image_path)
    if not uploaded_name:
        return False, f"Upload failed: {image_name}"

    # Build and queue workflow
    workflow = build_klein_workflow(prompt, uploaded_name)
    prompt_id = queue_prompt(url, workflow)
    if not prompt_id:
        return False, f"Queue failed: {image_name}"

    # Wait for completion
    output_info = wait_for_completion(url, prompt_id)
    if not output_info:
        return False, f"Timeout: {image_name}"

    # Download result
    if download_output(url, output_info, output_path):
        return True, output_path
    else:
        return False, f"Download failed: {image_name}"


def cmd_process(args):
    """Process images across the fleet."""
    state = load_state()
    urls = [inst["url"] for inst in state.get("instances", []) if inst.get("url")]

    if not urls:
        print("No fleet URLs available. Run 'launch' first.")
        return

    # Limit workers to available URLs
    workers = min(args.workers, len(urls))

    # Find all images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in image_exts]

    if not images:
        print(f"No images found in {input_dir}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(images)} images across {workers} instances")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output_dir}")
    print()

    # Distribute images across workers
    url_cycle = [urls[i % len(urls)] for i in range(len(images))]

    # Process with progress tracking
    completed = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for img_path, url in zip(images, url_cycle):
            future = executor.submit(process_image_on_instance, url, str(img_path), args.prompt, str(output_dir))
            futures[future] = img_path.name

        for future in as_completed(futures):
            img_name = futures[future]
            try:
                success, result = future.result()
                if success:
                    completed += 1
                    print(f"  [{completed}/{len(images)}] ✓ {img_name}")
                else:
                    failed += 1
                    print(f"  [{completed+failed}/{len(images)}] ✗ {img_name} — {result}")
            except Exception as e:
                failed += 1
                print(f"  [{completed+failed}/{len(images)}] ✗ {img_name} — Exception: {e}")

            # Show ETA
            if completed + failed > 0:
                elapsed = time.time() - start_time
                per_image = elapsed / (completed + failed)
                remaining = len(images) - (completed + failed)
                eta_sec = per_image * remaining
                eta_min = int(eta_sec / 60)
                print(f"    Progress: {completed+failed}/{len(images)} | ETA: {eta_min}m | Speed: {per_image:.1f}s/img")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE — {completed} succeeded, {failed} failed")
    print(f"Total time: {int(total_time/60)}m {int(total_time%60)}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Fleet Klein — Multi-instance parallel image editing")
    sub = parser.add_subparsers(dest="command")

    launch_p = sub.add_parser("launch", help="Launch N instances")
    launch_p.add_argument("--count", "-n", type=int, default=4, help="Number of instances (default: 4)")

    sub.add_parser("status", help="Show fleet status")
    sub.add_parser("urls", help="Print URLs for ComfyUI")
    sub.add_parser("destroy", help="Destroy all instances")

    process_p = sub.add_parser("process", help="Process images across fleet")
    process_p.add_argument("--input_dir", "-i", required=True, help="Input directory with images")
    process_p.add_argument("--output_dir", "-o", required=True, help="Output directory for edited images")
    process_p.add_argument("--prompt", "-p", required=True, help="Positive prompt for Klein")
    process_p.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)")

    args = parser.parse_args()

    commands = {
        "launch": cmd_launch,
        "status": cmd_status,
        "urls": cmd_urls,
        "destroy": cmd_destroy,
        "process": cmd_process,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

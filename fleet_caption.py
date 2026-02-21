#!/usr/bin/env python3
"""
Fleet Caption — Spin up N cheap Vast.ai instances for parallel VLM captioning.

Usage:
    python fleet_caption.py launch --count 10 --model 32b
    python fleet_caption.py status
    python fleet_caption.py urls
    python fleet_caption.py destroy

Workflow:
    1. Run 'launch' to rent N instances and install Ollama + model
    2. Run 'urls' to get comma-separated URLs for ComfyUI
    3. Paste URLs into the ollama_urls field, set workers=N
    4. Run your workflow — captioning runs across all instances
    5. Run 'destroy' when done to stop billing

Cost estimate (Vast.ai RTX 3090, ~$0.15/hr each):
    10 instances = ~$1.50/hr
    500 images at 32B ≈ 25-40 min = ~$1.00 total
    1000 images at 32B ≈ 50-80 min = ~$2.00 total
"""

import json
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FLEET_STATE = os.path.join(os.path.dirname(__file__), ".fleet_state.json")

MODELS = {
    "7b": "huihui_ai/qwen2.5-vl-abliterated:7b",
    "32b": "huihui_ai/qwen2.5-vl-abliterated:32b",
}

# Vast.ai search criteria for cheap caption instances
# RTX 3090 (24GB) is cheapest for 32B model (~20GB VRAM)
# RTX 3060 (12GB) works for 7B model (~8GB VRAM)
VAST_SEARCH_32B = {
    "gpu_ram": ">=22",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=40",
    "dph_total": "<=0.30",  # Max $0.30/hr
    "gpu_name": ["RTX_3090", "RTX_4090", "RTX_A6000", "A100_SXM4"],
}

VAST_SEARCH_7B = {
    "gpu_ram": ">=10",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=20",
    "dph_total": "<=0.15",
    "gpu_name": ["RTX_3060", "RTX_3070", "RTX_3080", "RTX_3090", "RTX_4060", "RTX_4070"],
}

SETUP_SCRIPT = """#!/bin/bash
set -e
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Start Ollama with parallel inference
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_HOST=0.0.0.0:11434
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
# Pull model
ollama pull {model}
echo "READY"
"""


def load_state():
    if os.path.exists(FLEET_STATE):
        with open(FLEET_STATE) as f:
            return json.load(f)
    return {"instances": [], "model": "", "status": "idle"}


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


def search_offers(model_size):
    """Search Vast.ai for suitable instances."""
    criteria = VAST_SEARCH_32B if model_size == "32b" else VAST_SEARCH_7B

    query_parts = []
    for key, val in criteria.items():
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


def launch_instance(offer_id, model):
    """Rent a single instance and set it up."""
    out, rc = run_vast(["create", "instance", str(offer_id),
                        "--image", "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
                        "--disk", "40",
                        "--onstart-cmd", SETUP_SCRIPT.format(model=model)])
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


def wait_for_ready(instance_ids, timeout=600):
    """Wait for all instances to be running and Ollama ready."""
    start = time.time()
    ready = set()
    urls = {}

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

                # Find the Ollama port mapping (11434)
                ollama_port = None
                if "11434/tcp" in ports:
                    port_info = ports["11434/tcp"]
                    if isinstance(port_info, list) and port_info:
                        ollama_port = port_info[0].get("HostPort")
                    elif isinstance(port_info, dict):
                        ollama_port = port_info.get("HostPort")

                if ip and ollama_port:
                    url = f"http://{ip}:{ollama_port}"
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
    model_size = args.model
    model = MODELS.get(model_size, MODELS["32b"])

    print(f"Searching for {count} instances for {model_size} model...")
    offers = search_offers(model_size)

    if len(offers) < count:
        print(f"WARNING: Only {len(offers)} offers found, need {count}")
        count = min(count, len(offers))

    if count == 0:
        print("No suitable instances found. Try relaxing search criteria.")
        return

    # Show cost estimate
    total_cost = sum(o.get("dph_total", 0) for o in offers[:count])
    print(f"Renting {count} instances at ~${total_cost:.2f}/hr total")
    print(f"Model: {model}")
    print()

    # Launch instances
    instance_ids = []
    for i, offer in enumerate(offers[:count]):
        offer_id = offer["id"]
        gpu = offer.get("gpu_name", "?")
        price = offer.get("dph_total", 0)
        print(f"  [{i+1}/{count}] Launching on {gpu} (${price:.3f}/hr)...")

        iid, msg = launch_instance(offer_id, model)
        if iid:
            instance_ids.append(iid)
            print(f"    Instance {iid}: created")
        else:
            print(f"    FAILED: {msg}")

    if not instance_ids:
        print("No instances launched successfully.")
        return

    print(f"\nWaiting for {len(instance_ids)} instances to boot + install Ollama + pull model...")
    print("(This takes 3-10 minutes depending on model size and network speed)")

    urls = wait_for_ready(instance_ids)

    # Save state
    state = {
        "instances": [{"id": iid, "url": urls.get(iid, "")} for iid in instance_ids],
        "model": model,
        "model_size": model_size,
        "status": "running",
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_state(state)

    print(f"\n{'='*60}")
    print(f"FLEET READY — {len(urls)}/{len(instance_ids)} instances online")
    print(f"{'='*60}")
    url_list = ",".join(urls.values())
    print(f"\nPaste this into ComfyUI 'ollama_urls' field:")
    print(f"  {url_list}")
    print(f"\nSet workers = {len(urls)}")
    print(f"\nRun 'python fleet_caption.py destroy' when done!")


def cmd_status(args):
    """Show fleet status."""
    state = load_state()
    if state["status"] == "idle":
        print("No fleet running.")
        return

    print(f"Fleet status: {state['status']}")
    print(f"Model: {state.get('model', '?')}")
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


def main():
    parser = argparse.ArgumentParser(description="Fleet Caption — Multi-instance VLM parallelism")
    sub = parser.add_subparsers(dest="command")

    launch_p = sub.add_parser("launch", help="Launch N instances")
    launch_p.add_argument("--count", "-n", type=int, default=10, help="Number of instances (default: 10)")
    launch_p.add_argument("--model", "-m", choices=["7b", "32b"], default="32b", help="Model size (default: 32b)")

    sub.add_parser("status", help="Show fleet status")
    sub.add_parser("urls", help="Print URLs for ComfyUI")
    sub.add_parser("destroy", help="Destroy all instances")

    args = parser.parse_args()

    commands = {
        "launch": cmd_launch,
        "status": cmd_status,
        "urls": cmd_urls,
        "destroy": cmd_destroy,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

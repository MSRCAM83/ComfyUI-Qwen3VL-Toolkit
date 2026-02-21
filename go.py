#!/usr/bin/env python3
"""
GO — One-Command LoRA Dataset Pipeline Launcher

Launch EVERYTHING with one command from your Windows PC:
    python go.py

This launches:
- 1 main ComfyUI instance (coordinator that runs the workflow)
- 10 Klein fleet instances (ComfyUI workers for image editing)
- 10 VLM fleet instances (Ollama for vision captioning)

The main ComfyUI instance opens in your browser automatically.
All fleet URLs are printed ready to paste into ComfyUI nodes.

Options:
    python go.py --klein 20 --vlm 15     # Scale fleets
    python go.py --model 7b              # Use Qwen 7B instead of 32B
    python go.py --no-klein              # Skip Klein fleet
    python go.py --race                  # Race mode: launch 2x, keep fastest N
    python go.py --race --race-ratio 3   # Race mode: launch 3x, keep fastest N
    python go.py --status                # Check current system status
    python go.py --destroy               # Destroy everything (main + fleets)

Race Mode:
    Launches 2x (or race-ratio x) the requested instances, waits for the fastest N
    to come online, then immediately destroys the slow ones. Saves money by cutting
    instances that are slow to boot or have network issues.

Status and URLs persist in .go_state.json
"""

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered output so progress shows when piped/redirected
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _abort_handler(signum, frame):
    """Handle Ctrl+C — destroy everything and exit."""
    print("\n\n  ABORT — Ctrl+C detected. Destroying all instances...")
    try:
        state_file = os.path.join(os.path.dirname(__file__), ".go_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)

            all_ids = []
            main = state.get("main_instance")
            if main and main.get("id"):
                all_ids.append(main["id"])
            for inst in state.get("klein_instances", []):
                if inst.get("id"):
                    all_ids.append(inst["id"])
            for inst in state.get("vlm_instances", []):
                if inst.get("id"):
                    all_ids.append(inst["id"])

            if all_ids:
                print(f"  Destroying {len(all_ids)} instances...")
                for iid in all_ids:
                    subprocess.run(["vastai", "destroy", "instance", str(iid)],
                                   capture_output=True, timeout=15)
                    print(f"    Instance {iid}: destroyed")

                state["status"] = "idle"
                state["main_instance"] = None
                state["klein_instances"] = []
                state["vlm_instances"] = []
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)

            print("  All instances destroyed. Billing stopped.")
        else:
            print("  No state file found. Check Vast.ai dashboard for orphans.")
    except Exception as e:
        print(f"  Cleanup error: {e}")
        print("  CHECK VAST.AI DASHBOARD — instances may still be running!")
    sys.exit(1)


signal.signal(signal.SIGINT, _abort_handler)
signal.signal(signal.SIGTERM, _abort_handler)


# Constants
STATE_FILE = os.path.join(os.path.dirname(__file__), ".go_state.json")

# Pre-built ComfyUI image (ComfyUI already installed, boots in ~2 min)
COMFY_IMAGE = "vastai/comfy:v0.13.0-cuda-12.9-py312"
# Base CUDA image for VLM-only instances (no ComfyUI needed)
BASE_IMAGE = "nvidia/cuda:12.1.0-runtime-ubuntu22.04"

# Search criteria
VAST_SEARCH_MAIN = "gpu_name in [RTX_3090,RTX_4090] gpu_ram>=22 num_gpus=1 disk_space>=60 inet_down>=200 reliability>=0.95 dph_total<=0.30"

VAST_SEARCH_KLEIN = "gpu_name in [RTX_3090,RTX_4090] gpu_ram>=22 num_gpus=1 disk_space>=40 inet_down>=200 reliability>=0.95 dph_total<=0.30"

VAST_SEARCH_VLM = "gpu_name in [RTX_3090,RTX_4090] gpu_ram>=22 num_gpus=1 disk_space>=40 inet_down>=200 reliability>=0.95 dph_total<=0.30"

VLM_MODELS = {
    "7b": "huihui_ai/qwen2.5-vl-abliterated:7b",
    "32b": "huihui_ai/qwen2.5-vl-abliterated:32b",
}


def _b64_script(script_text):
    """Base64-encode a bash script for safe --onstart-cmd passing."""
    encoded = base64.b64encode(script_text.encode()).decode()
    return f"bash -c 'echo {encoded} | base64 -d | bash'"


# Main instance setup: install toolkit + Ollama into pre-built ComfyUI image
# ComfyUI is already running on port 18188 from image entrypoint.
# We install the toolkit, then restart ComfyUI on 18188 (portal on 8188 proxies to it).
COMFYUI_SETUP_SCRIPT = """#!/bin/bash
set -e
# Wait for ComfyUI to boot from image entrypoint
sleep 15
# Find the ComfyUI custom_nodes directory
COMFY_DIR=$(find /opt /workspace /root -name "custom_nodes" -path "*/ComfyUI/*" 2>/dev/null | head -1)
if [ -z "$COMFY_DIR" ]; then
    echo "ERROR: Could not find ComfyUI custom_nodes directory"
    exit 1
fi
echo "Found custom_nodes at: $COMFY_DIR"
cd "$COMFY_DIR"
if [ ! -d ComfyUI-Qwen3VL-Toolkit ]; then
    git clone https://github.com/MSRCAM83/ComfyUI-Qwen3VL-Toolkit.git
fi
cd ComfyUI-Qwen3VL-Toolkit
pip install -r requirements.txt 2>&1
# Install Ollama
apt-get update && apt-get install -y zstd >/dev/null 2>&1
curl -fsSL https://ollama.com/install.sh | sh
OLLAMA_NUM_PARALLEL=4 OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
nohup ollama pull huihui_ai/qwen2.5-vl-abliterated:32b > /tmp/ollama_pull.log 2>&1 &
# Restart ComfyUI on port 18188 (portal at 8188 proxies to it)
pkill -f "python.*main.py" || true
sleep 3
COMFY_MAIN=$(find /opt /workspace /root -name "main.py" -path "*/ComfyUI/*" 2>/dev/null | head -1)
cd "$(dirname "$COMFY_MAIN")" && nohup python3 main.py --listen 0.0.0.0 --port 18188 > /tmp/comfyui.log 2>&1 &
echo "ComfyUI restarted on 18188"
"""

# Klein setup: just install toolkit into pre-built ComfyUI image (no Ollama needed)
KLEIN_SETUP_SCRIPT = """#!/bin/bash
set -e
sleep 15
COMFY_DIR=$(find /opt /workspace /root -name "custom_nodes" -path "*/ComfyUI/*" 2>/dev/null | head -1)
if [ -z "$COMFY_DIR" ]; then
    echo "ERROR: Could not find ComfyUI custom_nodes directory"
    exit 1
fi
cd "$COMFY_DIR"
if [ ! -d ComfyUI-Qwen3VL-Toolkit ]; then
    git clone https://github.com/MSRCAM83/ComfyUI-Qwen3VL-Toolkit.git
fi
cd ComfyUI-Qwen3VL-Toolkit
pip install -r requirements.txt 2>&1
# Restart ComfyUI on port 18188 (portal at 8188 proxies to it)
pkill -f "python.*main.py" || true
sleep 3
COMFY_MAIN=$(find /opt /workspace /root -name "main.py" -path "*/ComfyUI/*" 2>/dev/null | head -1)
cd "$(dirname "$COMFY_MAIN")" && nohup python3 main.py --listen 0.0.0.0 --port 18188 > /tmp/comfyui.log 2>&1 &
echo "ComfyUI restarted on 18188"
"""

# VLM-only setup: just Ollama (uses base CUDA image, no ComfyUI)
VLM_SETUP_SCRIPT_TEMPLATE = """#!/bin/bash
set -e
apt-get update && apt-get install -y zstd curl >/dev/null 2>&1
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_HOST=0.0.0.0:11434
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
ollama pull {model}
echo "READY"
"""


def load_state():
    """Load state from file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "main_instance": None,
        "klein_instances": [],
        "vlm_instances": [],
        "status": "idle",
        "launched_at": "",
        "total_cost_per_hour": 0.0,
        "model": ""
    }


def _parse_instance_id(out):
    """Parse instance ID from vastai create output."""
    # Try JSON/dict format: 'new_contract': 12345
    m = re.search(r"'new_contract':\s*(\d+)", out)
    if m:
        return int(m.group(1))
    # Try "new_contract": 12345 (proper JSON)
    m = re.search(r'"new_contract":\s*(\d+)', out)
    if m:
        return int(m.group(1))
    # No fallback — only trust new_contract field
    return None


def save_state(state):
    """Save state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def run_vast(args_list, timeout=60):
    """Run vastai CLI command and return output."""
    cmd = ["vastai"] + args_list
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip(), result.returncode
    except FileNotFoundError:
        print("ERROR: 'vastai' CLI not found. Install with: pip install vastai")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        return "timeout", 1


def search_offers(query):
    """Search Vast.ai for instances matching query string."""
    out, rc = run_vast(["search", "offers", "--raw", query])
    if rc != 0:
        print(f"Search failed: {out}")
        return []

    try:
        offers = json.loads(out)
        offers.sort(key=lambda x: x.get("dph_total", 999))
        return offers
    except json.JSONDecodeError:
        print(f"Failed to parse offers: {out[:200]}")
        return []


def get_instance_info(instance_id):
    """Get instance status and connection info."""
    out, rc = run_vast(["show", "instance", str(instance_id), "--raw"])
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def launch_main_instance():
    """Launch 4 main ComfyUI candidates, keep the fastest, destroy the rest."""
    race_count = 4
    print(f"\nStep 1/4: Launching main ComfyUI instance (racing {race_count}, keeping fastest)...")

    offers = search_offers(VAST_SEARCH_MAIN)
    if not offers:
        print("ERROR: No suitable instances found for main ComfyUI")
        return None

    launch_count = min(race_count, len(offers))
    if launch_count < race_count:
        print(f"  WARNING: Only {launch_count} offers available (wanted {race_count})")

    # Launch candidates in parallel
    instance_ids = []
    instance_prices = {}
    with ThreadPoolExecutor(max_workers=launch_count) as executor:
        futures = {}
        for i, offer in enumerate(offers[:launch_count]):
            gpu = offer.get("gpu_name", "?")
            price = offer.get("dph_total", 0)
            print(f"  [{i+1}/{launch_count}] {gpu} at ${price:.2f}/hr — launching...")
            future = executor.submit(run_vast, [
                "create", "instance", str(offer["id"]),
                "--image", COMFY_IMAGE,
                "--disk", "60",
                "--env", "-p 8188:8188 -p 11434:11434",
                "--onstart-cmd", _b64_script(COMFYUI_SETUP_SCRIPT)
            ])
            futures[future] = (offer["id"], price)

        for future in as_completed(futures):
            offer_id, price = futures[future]
            try:
                out, rc = future.result()
                if rc == 0:
                    iid = _parse_instance_id(out)
                    if iid:
                        instance_ids.append(iid)
                        instance_prices[iid] = price
                        print(f"    Instance {iid}: created")
                    else:
                        print(f"    FAILED to parse ID: {out[:100]}")
                else:
                    print(f"    FAILED: {out[:100]}")
            except Exception as e:
                print(f"    FAILED: {e}")

    if not instance_ids:
        print("ERROR: No instances launched successfully")
        return None

    print(f"  {len(instance_ids)} candidates racing...")

    # Race — first one to pass health check wins

    max_wait = 600
    start = time.time()
    winner_id = None
    winner_url = None

    while time.time() - start < max_wait and not winner_id:
        for iid in instance_ids:
            info = get_instance_info(iid)
            if not info:
                continue

            status = info.get("actual_status", info.get("status_msg", ""))
            if status == "running":
                ip = info.get("public_ipaddr", "")
                ports = info.get("ports", {})

                comfy_port = None
                if "8188/tcp" in ports:
                    port_info = ports["8188/tcp"]
                    if isinstance(port_info, list) and port_info:
                        comfy_port = port_info[0].get("HostPort")
                    elif isinstance(port_info, dict):
                        comfy_port = port_info.get("HostPort")

                if ip and comfy_port:
                    test_url = f"http://{ip}:{comfy_port}"
                    try:
                        resp = urllib.request.urlopen(f"{test_url}/system_stats", timeout=5)
                        if resp.status == 200:
                            elapsed = int(time.time() - start)
                            winner_id = iid
                            winner_url = test_url
                            print(f"  WINNER: Instance {iid} ready in {elapsed}s at {test_url}")
                            break
                    except Exception:
                        pass

        if not winner_id:
            elapsed = int(time.time() - start)
            print(f"  Racing... {elapsed}s elapsed", end="\r")
            time.sleep(5)

    # Destroy losers
    losers = [iid for iid in instance_ids if iid != winner_id]
    if losers:
        print(f"  Destroying {len(losers)} slower candidates...")
        with ThreadPoolExecutor(max_workers=len(losers)) as executor:
            for iid in losers:
                executor.submit(destroy_instance, iid)
                print(f"    Instance {iid}: DESTROYED")

    if not winner_id:
        print("ERROR: No instance became ready in time. All destroyed.")
        return None

    return {
        "id": winner_id,
        "url": winner_url,
        "dph_total": instance_prices.get(winner_id, 0)
    }


def launch_klein_instance(offer_id):
    """Launch a single Klein instance."""
    out, rc = run_vast([
        "create", "instance", str(offer_id),
        "--image", COMFY_IMAGE,
        "--disk", "40",
        "--env", "-p 8188:8188",
        "--onstart-cmd", _b64_script(KLEIN_SETUP_SCRIPT)
    ])  # Klein only needs ComfyUI port (8188)

    if rc != 0:
        return None, f"Create failed: {out}"

    iid = _parse_instance_id(out)
    if iid:
        return iid, "ok"

    return None, f"Could not parse instance ID from: {out}"


def launch_vlm_instance(offer_id, model):
    """Launch a single VLM instance."""
    out, rc = run_vast([
        "create", "instance", str(offer_id),
        "--image", BASE_IMAGE,
        "--disk", "40",
        "--env", "-p 11434:11434",
        "--onstart-cmd", _b64_script(VLM_SETUP_SCRIPT_TEMPLATE.format(model=model))
    ])  # VLM only needs Ollama port (11434)

    if rc != 0:
        return None, f"Create failed: {out}"

    # Parse instance ID
    iid = _parse_instance_id(out)
    if iid:
        return iid, "ok"

    return None, f"Could not parse instance ID from: {out}"


def wait_for_klein_ready(instance_ids, timeout=900, target_count=None):
    """Wait for Klein instances to be ready.

    Args:
        instance_ids: List of instance IDs to wait for
        timeout: Max time to wait in seconds
        target_count: If set (race mode), return as soon as this many are ready

    Returns:
        If target_count is None: dict of {instance_id: url}
        If target_count is set: tuple of (winners_dict, losers_list)
            winners_dict: {instance_id: url} for first N ready
            losers_list: [instance_id] for instances that didn't make it
    """
    start = time.time()
    ready_order = []  # Track order instances became ready
    urls = {}

    # If no target, we need all instances
    if target_count is None:
        target_count = len(instance_ids)

    while time.time() - start < timeout and len(ready_order) < target_count:
        for iid in instance_ids:
            if iid in urls:  # Already ready
                continue

            info = get_instance_info(iid)
            if not info:
                continue

            status = info.get("actual_status", info.get("status_msg", ""))
            if status == "running":
                ip = info.get("public_ipaddr", "")
                ports = info.get("ports", {})

                # Find ComfyUI port (18188)
                comfy_port = None
                if "8188/tcp" in ports:
                    port_info = ports["8188/tcp"]
                    if isinstance(port_info, list) and port_info:
                        comfy_port = port_info[0].get("HostPort")
                    elif isinstance(port_info, dict):
                        comfy_port = port_info.get("HostPort")

                if ip and comfy_port:
                    url = f"http://{ip}:{comfy_port}"
                    # Verify ComfyUI is responding
                    try:

                        resp = urllib.request.urlopen(f"{url}/system_stats", timeout=5)
                        if resp.status == 200:
                            elapsed = int(time.time() - start)
                            urls[iid] = url
                            ready_order.append((iid, elapsed))
                            print(f"    Instance {iid}: READY in {elapsed}s")
                    except Exception:
                        pass

        if len(ready_order) < target_count:
            elapsed = int(time.time() - start)
            print(f"  Klein: {len(ready_order)}/{target_count} ready ({elapsed}s elapsed)")
            time.sleep(10)

    # If this was a race, return winners and losers separately
    if target_count < len(instance_ids):
        winners = {iid: urls[iid] for iid, _ in ready_order[:target_count]}
        losers = [iid for iid in instance_ids if iid not in winners]
        return winners, losers

    # Destroy any instances that never came online (prevent orphans)
    unready = [iid for iid in instance_ids if iid not in urls]
    if unready:
        print(f"  Destroying {len(unready)} Klein instance(s) that didn't come online...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            for iid in unready:
                executor.submit(destroy_instance, iid)
                print(f"    Instance {iid}: timed out — DESTROYED")

    return urls


def wait_for_vlm_ready(instance_ids, timeout=900, target_count=None):
    """Wait for VLM instances to be ready.

    Args:
        instance_ids: List of instance IDs to wait for
        timeout: Max time to wait in seconds
        target_count: If set (race mode), return as soon as this many are ready

    Returns:
        If target_count is None: dict of {instance_id: url}
        If target_count is set: tuple of (winners_dict, losers_list)
            winners_dict: {instance_id: url} for first N ready
            losers_list: [instance_id] for instances that didn't make it
    """
    start = time.time()
    ready_order = []  # Track order instances became ready
    urls = {}

    # If no target, we need all instances
    if target_count is None:
        target_count = len(instance_ids)

    while time.time() - start < timeout and len(ready_order) < target_count:
        for iid in instance_ids:
            if iid in urls:  # Already ready
                continue

            info = get_instance_info(iid)
            if not info:
                continue

            status = info.get("actual_status", info.get("status_msg", ""))
            if status == "running":
                ip = info.get("public_ipaddr", "")
                ports = info.get("ports", {})

                # Find Ollama port (11434)
                ollama_port = None
                if "11434/tcp" in ports:
                    port_info = ports["11434/tcp"]
                    if isinstance(port_info, list) and port_info:
                        ollama_port = port_info[0].get("HostPort")
                    elif isinstance(port_info, dict):
                        ollama_port = port_info.get("HostPort")

                if ip and ollama_port:
                    url = f"http://{ip}:{ollama_port}"
                    # Verify Ollama is responding
                    try:

                        resp = urllib.request.urlopen(f"{url}/api/tags", timeout=5)
                        if resp.status == 200:
                            elapsed = int(time.time() - start)
                            urls[iid] = url
                            ready_order.append((iid, elapsed))
                            print(f"    Instance {iid}: READY in {elapsed}s")
                    except Exception:
                        pass

        if len(ready_order) < target_count:
            elapsed = int(time.time() - start)
            print(f"  VLM: {len(ready_order)}/{target_count} ready ({elapsed}s elapsed)")
            time.sleep(10)

    # If this was a race, return winners and losers separately
    if target_count < len(instance_ids):
        winners = {iid: urls[iid] for iid, _ in ready_order[:target_count]}
        losers = [iid for iid in instance_ids if iid not in winners]
        return winners, losers

    # Destroy any instances that never came online (prevent orphans)
    unready = [iid for iid in instance_ids if iid not in urls]
    if unready:
        print(f"  Destroying {len(unready)} VLM instance(s) that didn't come online...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            for iid in unready:
                executor.submit(destroy_instance, iid)
                print(f"    Instance {iid}: timed out — DESTROYED")

    return urls


def launch_klein_fleet(count, race_mode=False, race_ratio=2.0):
    """Launch Klein fleet in parallel.

    Args:
        count: Target number of instances to keep
        race_mode: If True, launch count*race_ratio instances and keep fastest count
        race_ratio: Multiplier for race mode (default 2.0)
    """
    launch_count = int(count * race_ratio) if race_mode else count

    if race_mode:
        print(f"\nStep 2/4: Launching Klein fleet in RACE MODE ({launch_count} launched, keeping fastest {count})...")
    else:
        print(f"\nStep 2/4: Launching Klein fleet ({count} instances)...")

    offers = search_offers(VAST_SEARCH_KLEIN)
    if len(offers) < launch_count:
        print(f"WARNING: Only {len(offers)} Klein offers found, need {launch_count}")
        launch_count = min(launch_count, len(offers))

    if launch_count == 0:
        print("No suitable Klein instances found.")
        return []

    # Launch in parallel
    instance_ids = []
    instance_prices = {}  # Track price per instance ID
    with ThreadPoolExecutor(max_workers=min(launch_count, 10)) as executor:
        futures = {}
        for i, offer in enumerate(offers[:launch_count]):
            future = executor.submit(launch_klein_instance, offer["id"])
            futures[future] = (i, offer)

        for future in as_completed(futures):
            i, offer = futures[future]
            gpu = offer.get("gpu_name", "?")
            price = offer.get("dph_total", 0)
            try:
                iid, msg = future.result()
                if iid:
                    instance_ids.append(iid)
                    instance_prices[iid] = price
                    print(f"  [{i+1}/{launch_count}] RTX 3090 (${price:.2f}/hr) — created")
                else:
                    print(f"  [{i+1}/{launch_count}] FAILED: {msg}")
            except Exception as e:
                print(f"  [{i+1}/{launch_count}] FAILED: {e}")

    if not instance_ids:
        print("No Klein instances launched successfully.")
        return []

    print(f"  {len(instance_ids)}/{launch_count} instances created")

    # Wait for instances to be ready
    if race_mode and len(instance_ids) > count:
        print(f"  Racing for fastest {count} instances...")
        urls, losers = wait_for_klein_ready(instance_ids, target_count=count)

        # Destroy losers
        if losers:
            print(f"  --- Cutting slowest {len(losers)} instances ---")
            destroyed_count = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(destroy_instance, iid): iid for iid in losers}
                for future in as_completed(futures):
                    iid = futures[future]
                    try:
                        success = future.result()
                        if success:
                            destroyed_count += 1
                            print(f"    Instance {iid}: still booting — DESTROYED")
                        else:
                            print(f"    Instance {iid}: destroy FAILED")
                    except Exception as e:
                        print(f"    Instance {iid}: destroy ERROR {e}")

            # Calculate cost savings
            avg_price = sum(instance_prices.values()) / len(instance_prices) if instance_prices else 0
            savings = destroyed_count * avg_price
            print(f"  Saved ~${savings:.2f}/hr by cutting slow instances")

        # Build result from winners only
        instance_ids = list(urls.keys())
    else:
        urls = wait_for_klein_ready(instance_ids)

    return [{"id": iid, "url": urls.get(iid, ""), "dph_total": instance_prices.get(iid, 0)}
            for iid in instance_ids if iid in urls]


def launch_vlm_fleet(count, model_name, race_mode=False, race_ratio=2.0):
    """Launch VLM fleet in parallel.

    Args:
        count: Target number of instances to keep
        model_name: Ollama model to pull
        race_mode: If True, launch count*race_ratio instances and keep fastest count
        race_ratio: Multiplier for race mode (default 2.0)
    """
    launch_count = int(count * race_ratio) if race_mode else count

    if race_mode:
        print(f"\nStep 3/4: Launching VLM fleet in RACE MODE ({launch_count} launched, keeping fastest {count})...")
    else:
        print(f"\nStep 3/4: Launching VLM fleet ({count} instances)...")

    offers = search_offers(VAST_SEARCH_VLM)
    if len(offers) < launch_count:
        print(f"WARNING: Only {len(offers)} VLM offers found, need {launch_count}")
        launch_count = min(launch_count, len(offers))

    if launch_count == 0:
        print("No suitable VLM instances found.")
        return []

    # Launch in parallel
    instance_ids = []
    instance_prices = {}  # Track price per instance ID
    with ThreadPoolExecutor(max_workers=min(launch_count, 10)) as executor:
        futures = {}
        for i, offer in enumerate(offers[:launch_count]):
            future = executor.submit(launch_vlm_instance, offer["id"], model_name)
            futures[future] = (i, offer)

        for future in as_completed(futures):
            i, offer = futures[future]
            gpu = offer.get("gpu_name", "?")
            price = offer.get("dph_total", 0)
            try:
                iid, msg = future.result()
                if iid:
                    instance_ids.append(iid)
                    instance_prices[iid] = price
                    print(f"  [{i+1}/{launch_count}] RTX 3090 (${price:.2f}/hr) — created")
                else:
                    print(f"  [{i+1}/{launch_count}] FAILED: {msg}")
            except Exception as e:
                print(f"  [{i+1}/{launch_count}] FAILED: {e}")

    if not instance_ids:
        print("No VLM instances launched successfully.")
        return []

    print(f"  {len(instance_ids)}/{launch_count} instances created")

    # Wait for instances to be ready
    if race_mode and len(instance_ids) > count:
        print(f"  Racing for fastest {count} instances...")
        urls, losers = wait_for_vlm_ready(instance_ids, target_count=count)

        # Destroy losers
        if losers:
            print(f"  --- Cutting slowest {len(losers)} instances ---")
            destroyed_count = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(destroy_instance, iid): iid for iid in losers}
                for future in as_completed(futures):
                    iid = futures[future]
                    try:
                        success = future.result()
                        if success:
                            destroyed_count += 1
                            print(f"    Instance {iid}: still booting — DESTROYED")
                        else:
                            print(f"    Instance {iid}: destroy FAILED")
                    except Exception as e:
                        print(f"    Instance {iid}: destroy ERROR {e}")

            # Calculate cost savings
            avg_price = sum(instance_prices.values()) / len(instance_prices) if instance_prices else 0
            savings = destroyed_count * avg_price
            print(f"  Saved ~${savings:.2f}/hr by cutting slow instances")

        # Build result from winners only
        instance_ids = list(urls.keys())
    else:
        urls = wait_for_vlm_ready(instance_ids)

    return [{"id": iid, "url": urls.get(iid, ""), "dph_total": instance_prices.get(iid, 0)}
            for iid in instance_ids if iid in urls]


def destroy_instance(instance_id):
    """Destroy a single instance."""
    out, rc = run_vast(["destroy", "instance", str(instance_id)])
    return rc == 0


def destroy_all():
    """Destroy all instances (main + fleets)."""
    state = load_state()

    all_ids = []
    if state.get("main_instance"):
        all_ids.append(state["main_instance"]["id"])

    for inst in state.get("klein_instances", []):
        all_ids.append(inst["id"])

    for inst in state.get("vlm_instances", []):
        all_ids.append(inst["id"])

    if not all_ids:
        print("No instances to destroy.")
        return

    print(f"Destroying {len(all_ids)} instances...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(destroy_instance, iid): iid for iid in all_ids}

        for future in as_completed(futures):
            iid = futures[future]
            try:
                success = future.result()
                status = "destroyed" if success else "FAILED"
                print(f"  Instance {iid}: {status}")
            except Exception as e:
                print(f"  Instance {iid}: ERROR {e}")

    # Clear state
    save_state({
        "main_instance": None,
        "klein_instances": [],
        "vlm_instances": [],
        "status": "idle",
        "launched_at": "",
        "total_cost_per_hour": 0.0,
        "model": ""
    })
    print("\nAll instances destroyed.")


def show_status():
    """Show current system status."""
    state = load_state()

    print("=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)

    if state.get("status") == "idle":
        print("Status: Idle (no instances running)")
        print("\nRun 'python go.py' to launch everything.")
        return

    print(f"Status: {state.get('status')}")
    print(f"Launched: {state.get('launched_at')}")
    print(f"Total cost: ~${state.get('total_cost_per_hour', 0):.2f}/hr")

    if state.get("main_instance"):
        main = state["main_instance"]
        print(f"\nMain ComfyUI: {main.get('url', 'N/A')}")
        print(f"  Instance ID: {main.get('id')}")

    klein_instances = state.get("klein_instances", [])
    if klein_instances:
        online = len([i for i in klein_instances if i.get("url")])
        print(f"\nKlein Fleet: {online}/{len(klein_instances)} online")

    vlm_instances = state.get("vlm_instances", [])
    if vlm_instances:
        online = len([i for i in vlm_instances if i.get("url")])
        print(f"VLM Fleet: {online}/{len(vlm_instances)} online")
        print(f"  Model: {state.get('model', 'N/A')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GO — One-Command LoRA Dataset Pipeline Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--klein", type=int, default=10, choices=range(2, 51), metavar="N", help="Number of Klein instances, 2-50 (default: 10)")
    parser.add_argument("--vlm", type=int, default=10, choices=range(2, 51), metavar="N", help="Number of VLM instances, 2-50 (default: 10)")
    parser.add_argument("--model", choices=["7b", "32b"], default="32b", help="VLM model size (default: 32b)")
    parser.add_argument("--no-klein", action="store_true", help="Skip Klein fleet")
    parser.add_argument("--race", action="store_true", help="Race mode: launch 2x instances, keep fastest N, destroy the rest")
    parser.add_argument("--race-ratio", type=float, default=2.0, help="Race mode multiplier (default: 2.0)")
    parser.add_argument("--destroy", action="store_true", help="Destroy all instances")
    parser.add_argument("--status", action="store_true", help="Show current status")

    args = parser.parse_args()

    # Handle special commands
    if args.destroy:
        destroy_all()
        return

    if args.status:
        show_status()
        return

    # Launch everything
    print("=" * 60)
    print("LORA DATASET PIPELINE — One-Click Launch")
    print("=" * 60)

    # Step 1: Launch main ComfyUI
    main_instance = launch_main_instance()
    if not main_instance:
        print("\nERROR: Failed to launch main instance. Aborting.")
        return

    # Step 2 & 3: Launch fleets in parallel
    klein_instances = []
    vlm_instances = []

    klein_count = 0 if args.no_klein else args.klein
    vlm_count = args.vlm
    model_name = VLM_MODELS[args.model]
    race_mode = args.race
    race_ratio = args.race_ratio

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []

        if klein_count > 0:
            futures.append(("klein", executor.submit(launch_klein_fleet, klein_count, race_mode, race_ratio)))

        if vlm_count > 0:
            futures.append(("vlm", executor.submit(launch_vlm_fleet, vlm_count, model_name, race_mode, race_ratio)))

        for fleet_type, future in futures:
            try:
                instances = future.result()
                if fleet_type == "klein":
                    klein_instances = instances
                else:
                    vlm_instances = instances
            except Exception as e:
                print(f"ERROR launching {fleet_type} fleet: {e}")

    # Step 4: Open browser
    print("\nStep 4/4: Opening ComfyUI...")
    if main_instance.get("url"):
        webbrowser.open(main_instance["url"])

    # Calculate total cost
    total_cost = main_instance.get("dph_total", 0)
    for inst in klein_instances:
        total_cost += inst.get("dph_total", 0)
    for inst in vlm_instances:
        total_cost += inst.get("dph_total", 0)

    # Save state
    state = {
        "main_instance": main_instance,
        "klein_instances": klein_instances,
        "vlm_instances": vlm_instances,
        "status": "running",
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cost_per_hour": total_cost,
        "model": args.model
    }
    save_state(state)

    # Print summary
    print("\n" + "=" * 60)
    print("ALL SYSTEMS GO")
    print("=" * 60)
    print(f"Main ComfyUI: {main_instance.get('url', 'N/A')} (opened in browser)")
    print(f"Total cost:   ~${total_cost:.2f}/hr ({1 + len(klein_instances) + len(vlm_instances)} instances)")

    # Print Klein URLs
    if klein_instances:
        klein_urls = [i["url"] for i in klein_instances if i.get("url")]
        if klein_urls:
            print(f"\nKlein URLs (paste into KleinFleet node 'comfyui_urls'):")
            print(",".join(klein_urls))

    # Print VLM URLs
    if vlm_instances:
        vlm_urls = [i["url"] for i in vlm_instances if i.get("url")]
        if vlm_urls:
            print(f"\nVLM URLs (paste into Caption node 'ollama_urls'):")
            print(",".join(vlm_urls))

    # Upload instructions
    if main_instance.get("id"):
        print(f"\nUpload your videos/images to the main instance:")
        print(f"  vastai copy {main_instance['id']} /path/to/files/ /workspace/input/")

    print(f"\nWhen done: python go.py --destroy")
    print("=" * 60)


if __name__ == "__main__":
    main()

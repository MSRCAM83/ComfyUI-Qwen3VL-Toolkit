#!/usr/bin/env python3
"""
Fleet Manager — Unified management for Klein (ComfyUI) and VLM (Ollama) instances.

Usage:
    # Launch both fleets
    python fleet.py launch --klein 10 --vlm 10 --model 32b

    # Launch only Klein or VLM
    python fleet.py launch --klein-only --klein 10
    python fleet.py launch --vlm-only --vlm 10 --model 7b

    # Check status
    python fleet.py status

    # Get URLs for ComfyUI nodes
    python fleet.py urls --klein      # Klein URLs only
    python fleet.py urls --vlm        # VLM URLs only
    python fleet.py urls --all        # Both (default)

    # Scale fleets
    python fleet.py scale --klein 15  # Scale Klein to 15 instances
    python fleet.py scale --vlm 8     # Scale VLM to 8 instances

    # Destroy fleets
    python fleet.py destroy           # Destroy everything
    python fleet.py destroy --klein-only
    python fleet.py destroy --vlm-only

    # Configure running instances
    python fleet.py configure --vlm-model huihui_ai/qwen2.5-vl-abliterated:7b
    python fleet.py configure --parallel 8 --vlm-only
    python fleet.py configure --update-toolkit --klein-only

    # Run arbitrary commands
    python fleet.py run "nvidia-smi" --klein-only
    python fleet.py run "ollama list"
    python fleet.py run "df -h"

    # Health check
    python fleet.py health

Fleet Types:
    Klein (ComfyUI): Image editing with Klein model
    VLM (Ollama): Vision-language model captioning (Qwen 7B or 32B)

Cost Estimates (Vast.ai RTX 3090 @ ~$0.15-0.30/hr each):
    10 Klein + 10 VLM = ~$3.00-6.00/hr
    500 images (Klein + caption) ≈ 30-50 min = ~$2.50 total
    1000 images ≈ 60-100 min = ~$5.00 total
"""

import json
import os
import sys
import time
import subprocess
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FLEET_STATE = os.path.join(os.path.dirname(__file__), ".fleet_state.json")

MODELS = {
    "7b": "huihui_ai/qwen2.5-vl-abliterated:7b",
    "32b": "huihui_ai/qwen2.5-vl-abliterated:32b",
}

# Vast.ai search criteria
VAST_SEARCH_KLEIN = {
    "gpu_ram": ">=22",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=40",
    "dph_total": "<=0.30",
    "gpu_name": ["RTX_3090", "RTX_4090", "RTX_A6000", "A100_SXM4"],
}

VAST_SEARCH_VLM_32B = {
    "gpu_ram": ">=22",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=40",
    "dph_total": "<=0.30",
    "gpu_name": ["RTX_3090", "RTX_4090", "RTX_A6000", "A100_SXM4"],
}

VAST_SEARCH_VLM_7B = {
    "gpu_ram": ">=10",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=20",
    "dph_total": "<=0.15",
    "gpu_name": ["RTX_3060", "RTX_3070", "RTX_3080", "RTX_3090", "RTX_4060", "RTX_4070"],
}

# Klein Docker image
KLEIN_DOCKER_IMAGE = "msrcam/klein-fleet:latest"
KLEIN_ONSTART_CMD = "/workspace/start.sh"

# VLM setup script
VLM_SETUP_SCRIPT = """#!/bin/bash
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
    """Load fleet state from file."""
    if os.path.exists(FLEET_STATE):
        with open(FLEET_STATE) as f:
            return json.load(f)
    return {
        "klein_instances": [],
        "vlm_instances": [],
        "vlm_model": "",
        "status": "idle",
        "launched_at": ""
    }


def save_state(state):
    """Save fleet state to file."""
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


def search_offers(criteria):
    """Search Vast.ai for instances matching criteria."""
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


def launch_klein_instance(offer_id):
    """Launch a single Klein instance."""
    out, rc = run_vast(["create", "instance", str(offer_id),
                        "--image", KLEIN_DOCKER_IMAGE,
                        "--disk", "40",
                        "--onstart-cmd", KLEIN_ONSTART_CMD])
    if rc != 0:
        return None, f"Create failed: {out}"

    iid = _parse_instance_id(out)
    if iid:
        return iid, "ok"
    return None, f"Could not parse instance ID from: {out}"


def launch_vlm_instance(offer_id, model):
    """Launch a single VLM instance."""
    out, rc = run_vast(["create", "instance", str(offer_id),
                        "--image", "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
                        "--disk", "40",
                        "--onstart-cmd", VLM_SETUP_SCRIPT.format(model=model)])
    if rc != 0:
        return None, f"Create failed: {out}"

    iid = _parse_instance_id(out)
    if iid:
        return iid, "ok"
    return None, f"Could not parse instance ID from: {out}"


def get_instance_info(instance_id):
    """Get instance status and connection info."""
    out, rc = run_vast(["show", "instance", str(instance_id), "--raw"])
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def check_comfyui_ready(url):
    """Check if ComfyUI is ready."""
    try:
        resp = requests.get(f"{url}/system_stats", timeout=5)
        return resp.status_code == 200
    except:
        return False


def wait_for_klein_ready(instance_ids, timeout=900):
    """Wait for Klein instances to be ready."""
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
                ip = info.get("public_ipaddr", "")
                ports = info.get("ports", {})

                # Find ComfyUI port (18188)
                comfy_port = None
                if "18188/tcp" in ports:
                    port_info = ports["18188/tcp"]
                    if isinstance(port_info, list) and port_info:
                        comfy_port = port_info[0].get("HostPort")
                    elif isinstance(port_info, dict):
                        comfy_port = port_info.get("HostPort")

                if ip and comfy_port:
                    url = f"http://{ip}:{comfy_port}"
                    # Verify ComfyUI is responding
                    if check_comfyui_ready(url):
                        urls[iid] = url
                        ready.add(iid)
                        print(f"  Klein {iid}: READY at {url}")

        if len(ready) < len(instance_ids):
            elapsed = int(time.time() - start)
            print(f"  Klein: {len(ready)}/{len(instance_ids)} ready ({elapsed}s elapsed)")
            time.sleep(10)

    return urls


def wait_for_vlm_ready(instance_ids, timeout=600):
    """Wait for VLM instances to be ready."""
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
                    urls[iid] = url
                    ready.add(iid)
                    print(f"  VLM {iid}: READY at {url}")

        if len(ready) < len(instance_ids):
            elapsed = int(time.time() - start)
            print(f"  VLM: {len(ready)}/{len(instance_ids)} ready ({elapsed}s elapsed)")
            time.sleep(10)

    return urls


def launch_klein_fleet(count):
    """Launch Klein fleet in parallel."""
    print(f"\nSearching for {count} Klein instances...")
    offers = search_offers(VAST_SEARCH_KLEIN)

    if len(offers) < count:
        print(f"WARNING: Only {len(offers)} Klein offers found, need {count}")
        count = min(count, len(offers))

    if count == 0:
        print("No suitable Klein instances found.")
        return []

    klein_cost = sum(o.get("dph_total", 0) for o in offers[:count])
    print(f"Renting {count} Klein instances at ~${klein_cost:.2f}/hr")

    # Launch in parallel
    instance_ids = []
    with ThreadPoolExecutor(max_workers=min(count, 10)) as executor:
        futures = {}
        for i, offer in enumerate(offers[:count]):
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
                    print(f"  [{i+1}/{count}] Klein {iid}: created on {gpu} (${price:.3f}/hr)")
                else:
                    print(f"  [{i+1}/{count}] FAILED: {msg}")
            except Exception as e:
                print(f"  [{i+1}/{count}] FAILED: {e}")

    if not instance_ids:
        print("No Klein instances launched successfully.")
        return []

    print(f"\nWaiting for {len(instance_ids)} Klein instances (ComfyUI boot + model load)...")
    print("(This takes 3-10 minutes)")
    urls = wait_for_klein_ready(instance_ids)

    return [{"id": iid, "url": urls.get(iid, "")} for iid in instance_ids]


def launch_vlm_fleet(count, model_size):
    """Launch VLM fleet in parallel."""
    model = MODELS.get(model_size, MODELS["32b"])
    criteria = VAST_SEARCH_VLM_32B if model_size == "32b" else VAST_SEARCH_VLM_7B

    print(f"\nSearching for {count} VLM instances ({model_size} model)...")
    offers = search_offers(criteria)

    if len(offers) < count:
        print(f"WARNING: Only {len(offers)} VLM offers found, need {count}")
        count = min(count, len(offers))

    if count == 0:
        print("No suitable VLM instances found.")
        return []

    vlm_cost = sum(o.get("dph_total", 0) for o in offers[:count])
    print(f"Renting {count} VLM instances at ~${vlm_cost:.2f}/hr")

    # Launch in parallel
    instance_ids = []
    with ThreadPoolExecutor(max_workers=min(count, 10)) as executor:
        futures = {}
        for i, offer in enumerate(offers[:count]):
            future = executor.submit(launch_vlm_instance, offer["id"], model)
            futures[future] = (i, offer)

        for future in as_completed(futures):
            i, offer = futures[future]
            gpu = offer.get("gpu_name", "?")
            price = offer.get("dph_total", 0)
            try:
                iid, msg = future.result()
                if iid:
                    instance_ids.append(iid)
                    print(f"  [{i+1}/{count}] VLM {iid}: created on {gpu} (${price:.3f}/hr)")
                else:
                    print(f"  [{i+1}/{count}] FAILED: {msg}")
            except Exception as e:
                print(f"  [{i+1}/{count}] FAILED: {e}")

    if not instance_ids:
        print("No VLM instances launched successfully.")
        return []

    print(f"\nWaiting for {len(instance_ids)} VLM instances (Ollama install + model pull)...")
    print("(This takes 3-10 minutes)")
    urls = wait_for_vlm_ready(instance_ids)

    return [{"id": iid, "url": urls.get(iid, "")} for iid in instance_ids]


def cmd_launch(args):
    """Launch fleet instances."""
    klein_count = args.klein
    vlm_count = args.vlm
    model_size = args.model
    klein_only = args.klein_only
    vlm_only = args.vlm_only

    state = load_state()

    # Determine what to launch
    if klein_only:
        vlm_count = 0
    elif vlm_only:
        klein_count = 0

    if klein_count == 0 and vlm_count == 0:
        print("Nothing to launch. Specify --klein and/or --vlm counts.")
        return

    # Show cost estimate
    print("=" * 60)
    print("FLEET LAUNCH")
    print("=" * 60)
    if klein_count > 0:
        print(f"Klein instances: {klein_count}")
    if vlm_count > 0:
        print(f"VLM instances: {vlm_count} ({model_size} model)")

    # Launch fleets in parallel
    klein_instances = []
    vlm_instances = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if klein_count > 0:
            futures.append(("klein", executor.submit(launch_klein_fleet, klein_count)))
        if vlm_count > 0:
            futures.append(("vlm", executor.submit(launch_vlm_fleet, vlm_count, model_size)))

        for fleet_type, future in futures:
            try:
                instances = future.result()
                if fleet_type == "klein":
                    klein_instances = instances
                else:
                    vlm_instances = instances
            except Exception as e:
                print(f"ERROR launching {fleet_type} fleet: {e}")

    # Update state
    if klein_instances:
        state["klein_instances"] = klein_instances
    if vlm_instances:
        state["vlm_instances"] = vlm_instances
        state["vlm_model"] = model_size

    state["status"] = "running"
    state["launched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)

    # Print summary
    print("\n" + "=" * 60)
    print("FLEET READY")
    print("=" * 60)

    klein_ready = len([i for i in klein_instances if i.get("url")])
    vlm_ready = len([i for i in vlm_instances if i.get("url")])

    if klein_instances:
        print(f"Klein Fleet: {klein_ready}/{len(klein_instances)} instances online")
    if vlm_instances:
        print(f"VLM Fleet:   {vlm_ready}/{len(vlm_instances)} instances online")

    # Calculate total cost
    total_cost = 0.0
    for inst in klein_instances:
        info = get_instance_info(inst["id"])
        if info:
            total_cost += info.get("dph_total", 0)
    for inst in vlm_instances:
        info = get_instance_info(inst["id"])
        if info:
            total_cost += info.get("dph_total", 0)

    print(f"Total cost:  ~${total_cost:.2f}/hr")
    print()

    # Print URLs
    if klein_instances:
        klein_urls = [i["url"] for i in klein_instances if i.get("url")]
        if klein_urls:
            print("Klein URLs (paste into QVL_KleinFleet 'comfyui_urls'):")
            print(f"  {','.join(klein_urls)}")
            print()

    if vlm_instances:
        vlm_urls = [i["url"] for i in vlm_instances if i.get("url")]
        if vlm_urls:
            print("VLM URLs (paste into QVL_Caption 'ollama_urls'):")
            print(f"  {','.join(vlm_urls)}")
            print()

    print("Run 'python fleet.py destroy' when done!")
    print("=" * 60)


def cmd_status(args):
    """Show fleet status."""
    state = load_state()

    if state["status"] == "idle":
        print("No fleet running.")
        return

    print("=" * 60)
    print("FLEET STATUS")
    print("=" * 60)
    print(f"Status: {state['status']}")
    print(f"Launched: {state.get('launched_at', '?')}")
    print()

    # Klein fleet
    klein_instances = state.get("klein_instances", [])
    if klein_instances:
        print(f"Klein Fleet: {len(klein_instances)} instances")
        for inst in klein_instances:
            iid = inst["id"]
            info = get_instance_info(iid)
            status = "unknown"
            cost = 0.0
            if info:
                status = info.get("actual_status", info.get("status_msg", "?"))
                cost = info.get("dph_total", 0)
            url = inst.get("url", "no url")
            print(f"  {iid}: {status} (${cost:.3f}/hr) — {url}")
        print()

    # VLM fleet
    vlm_instances = state.get("vlm_instances", [])
    if vlm_instances:
        print(f"VLM Fleet: {len(vlm_instances)} instances ({state.get('vlm_model', '?')} model)")
        for inst in vlm_instances:
            iid = inst["id"]
            info = get_instance_info(iid)
            status = "unknown"
            cost = 0.0
            if info:
                status = info.get("actual_status", info.get("status_msg", "?"))
                cost = info.get("dph_total", 0)
            url = inst.get("url", "no url")
            print(f"  {iid}: {status} (${cost:.3f}/hr) — {url}")
        print()

    # Total cost
    total_cost = 0.0
    for inst in klein_instances + vlm_instances:
        info = get_instance_info(inst["id"])
        if info:
            total_cost += info.get("dph_total", 0)
    print(f"Total Fleet Cost: ~${total_cost:.2f}/hr")


def cmd_urls(args):
    """Print fleet URLs."""
    state = load_state()

    klein_instances = state.get("klein_instances", [])
    vlm_instances = state.get("vlm_instances", [])

    klein_urls = [i["url"] for i in klein_instances if i.get("url")]
    vlm_urls = [i["url"] for i in vlm_instances if i.get("url")]

    # Determine output format
    if args.klein:
        if klein_urls:
            print(",".join(klein_urls))
        else:
            print("No Klein URLs available.")
    elif args.vlm:
        if vlm_urls:
            print(",".join(vlm_urls))
        else:
            print("No VLM URLs available.")
    else:
        # Print both labeled
        if klein_urls:
            print("Klein URLs:")
            print(",".join(klein_urls))
            print()
        if vlm_urls:
            print("VLM URLs:")
            print(",".join(vlm_urls))
        if not klein_urls and not vlm_urls:
            print("No URLs available. Run 'launch' first.")


def cmd_destroy(args):
    """Destroy fleet instances."""
    state = load_state()

    klein_instances = state.get("klein_instances", [])
    vlm_instances = state.get("vlm_instances", [])

    klein_only = args.klein_only
    vlm_only = args.vlm_only

    # Determine what to destroy
    instances_to_destroy = []
    if klein_only:
        instances_to_destroy = klein_instances
        fleet_name = "Klein"
    elif vlm_only:
        instances_to_destroy = vlm_instances
        fleet_name = "VLM"
    else:
        instances_to_destroy = klein_instances + vlm_instances
        fleet_name = "all"

    if not instances_to_destroy:
        print(f"No {fleet_name} instances to destroy.")
        return

    print(f"Destroying {len(instances_to_destroy)} {fleet_name} instances...")

    for inst in instances_to_destroy:
        iid = inst["id"]
        out, rc = run_vast(["destroy", "instance", str(iid)])
        if rc == 0:
            print(f"  {iid}: destroyed")
        else:
            print(f"  {iid}: destroy failed — {out}")

    # Update state
    if klein_only:
        state["klein_instances"] = []
    elif vlm_only:
        state["vlm_instances"] = []
    else:
        state["klein_instances"] = []
        state["vlm_instances"] = []
        state["status"] = "idle"

    save_state(state)
    print("Fleet destroyed. Billing stopped.")


def cmd_scale(args):
    """Scale fleets up or down."""
    state = load_state()

    klein_target = args.klein
    vlm_target = args.vlm

    klein_instances = state.get("klein_instances", [])
    vlm_instances = state.get("vlm_instances", [])

    # Scale Klein
    if klein_target is not None:
        klein_current = len(klein_instances)
        if klein_target > klein_current:
            # Scale up
            to_add = klein_target - klein_current
            print(f"Scaling Klein up: {klein_current} -> {klein_target} (+{to_add})")
            new_instances = launch_klein_fleet(to_add)
            state["klein_instances"].extend(new_instances)
        elif klein_target < klein_current:
            # Scale down
            to_remove = klein_current - klein_target
            print(f"Scaling Klein down: {klein_current} -> {klein_target} (-{to_remove})")
            for i in range(to_remove):
                inst = state["klein_instances"].pop()
                iid = inst["id"]
                out, rc = run_vast(["destroy", "instance", str(iid)])
                if rc == 0:
                    print(f"  {iid}: destroyed")
                else:
                    print(f"  {iid}: destroy failed — {out}")
        else:
            print(f"Klein fleet already at {klein_target} instances.")

    # Scale VLM
    if vlm_target is not None:
        vlm_current = len(vlm_instances)
        model_size = state.get("vlm_model", "32b")
        if vlm_target > vlm_current:
            # Scale up
            to_add = vlm_target - vlm_current
            print(f"Scaling VLM up: {vlm_current} -> {vlm_target} (+{to_add})")
            new_instances = launch_vlm_fleet(to_add, model_size)
            state["vlm_instances"].extend(new_instances)
        elif vlm_target < vlm_current:
            # Scale down
            to_remove = vlm_current - vlm_target
            print(f"Scaling VLM down: {vlm_current} -> {vlm_target} (-{to_remove})")
            for i in range(to_remove):
                inst = state["vlm_instances"].pop()
                iid = inst["id"]
                out, rc = run_vast(["destroy", "instance", str(iid)])
                if rc == 0:
                    print(f"  {iid}: destroyed")
                else:
                    print(f"  {iid}: destroy failed — {out}")
        else:
            print(f"VLM fleet already at {vlm_target} instances.")

    save_state(state)
    print("\nFleet scaled. Run 'python fleet.py status' to verify.")


def execute_on_instance(instance_id, command, timeout=120):
    """Execute a command on a single instance and return output."""
    out, rc = run_vast(["execute", str(instance_id), command])
    return {"instance_id": instance_id, "output": out, "rc": rc}


def cmd_configure(args):
    """Push configuration changes to running instances."""
    state = load_state()

    klein_instances = state.get("klein_instances", [])
    vlm_instances = state.get("vlm_instances", [])

    # Determine target instances
    if args.klein_only:
        target_instances = [("Klein", inst["id"]) for inst in klein_instances]
    elif args.vlm_only:
        target_instances = [("VLM", inst["id"]) for inst in vlm_instances]
    else:
        target_instances = (
            [("Klein", inst["id"]) for inst in klein_instances] +
            [("VLM", inst["id"]) for inst in vlm_instances]
        )

    if not target_instances:
        print("No instances to configure.")
        return

    # Build configuration tasks
    tasks = []

    if args.vlm_model:
        # Only apply to VLM instances
        vlm_targets = [(t, iid) for t, iid in target_instances if t == "VLM"]
        if vlm_targets:
            print(f"\nPulling model '{args.vlm_model}' on {len(vlm_targets)} VLM instances...")
            for fleet_type, iid in vlm_targets:
                tasks.append((iid, f"ollama pull {args.vlm_model}", "Model pull"))

    if args.parallel:
        # Only apply to VLM instances
        vlm_targets = [(t, iid) for t, iid in target_instances if t == "VLM"]
        if vlm_targets:
            print(f"\nSetting OLLAMA_NUM_PARALLEL={args.parallel} on {len(vlm_targets)} VLM instances...")
            restart_cmd = f"pkill ollama; OLLAMA_NUM_PARALLEL={args.parallel} OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve > /tmp/ollama.log 2>&1 & sleep 3"
            for fleet_type, iid in vlm_targets:
                tasks.append((iid, restart_cmd, "Parallel config"))

    if args.update_toolkit:
        # Only apply to Klein instances
        klein_targets = [(t, iid) for t, iid in target_instances if t == "Klein"]
        if klein_targets:
            print(f"\nUpdating toolkit on {len(klein_targets)} Klein instances...")
            update_cmd = "cd /workspace/ComfyUI/custom_nodes/ComfyUI-Qwen3VL-Toolkit && git pull"
            for fleet_type, iid in klein_targets:
                tasks.append((iid, update_cmd, "Toolkit update"))

    if not tasks:
        print("No configuration tasks specified. Use --vlm-model, --parallel, or --update-toolkit.")
        return

    # Execute tasks in parallel
    results = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 20)) as executor:
        futures = {}
        for i, (iid, cmd, label) in enumerate(tasks):
            future = executor.submit(execute_on_instance, iid, cmd)
            futures[future] = (i, iid, label)

        for future in as_completed(futures):
            i, iid, label = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "OK" if result["rc"] == 0 else "FAIL"
                print(f"  [{i+1}/{len(tasks)}] Instance {iid} ({label}): {status}")
            except Exception as e:
                print(f"  [{i+1}/{len(tasks)}] Instance {iid} ({label}): ERROR - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("CONFIGURATION RESULTS")
    print("=" * 60)
    success = sum(1 for r in results if r["rc"] == 0)
    failed = len(results) - success
    print(f"Success: {success}/{len(results)}")
    if failed > 0:
        print(f"Failed:  {failed}/{len(results)}")
        print("\nFailed instances:")
        for r in results:
            if r["rc"] != 0:
                print(f"  Instance {r['instance_id']}: {r['output'][:100]}")


def cmd_run(args):
    """Execute arbitrary command across instances."""
    state = load_state()

    klein_instances = state.get("klein_instances", [])
    vlm_instances = state.get("vlm_instances", [])

    # Determine target instances
    if args.klein_only:
        target_instances = [("Klein", inst["id"]) for inst in klein_instances]
    elif args.vlm_only:
        target_instances = [("VLM", inst["id"]) for inst in vlm_instances]
    else:
        target_instances = (
            [("Klein", inst["id"]) for inst in klein_instances] +
            [("VLM", inst["id"]) for inst in vlm_instances]
        )

    if not target_instances:
        print("No instances to run command on.")
        return

    command = args.command
    print(f"\nRunning command on {len(target_instances)} instances:")
    print(f"  Command: {command}")
    print()

    # Execute in parallel
    results = []
    with ThreadPoolExecutor(max_workers=min(len(target_instances), 20)) as executor:
        futures = {}
        for fleet_type, iid in target_instances:
            future = executor.submit(execute_on_instance, iid, command, args.timeout)
            futures[future] = (fleet_type, iid)

        for future in as_completed(futures):
            fleet_type, iid = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Instance {iid} ({fleet_type}):")
                print(f"  Exit code: {result['rc']}")
                if result['output']:
                    for line in result['output'].split('\n')[:20]:  # Limit output
                        print(f"  {line}")
                print()
            except Exception as e:
                print(f"Instance {iid} ({fleet_type}): ERROR - {e}")
                print()


def check_klein_health(instance_id, url):
    """Check Klein instance health."""
    try:
        resp = requests.get(f"{url}/system_stats", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Try to get model count
            models = "?"
            return {"status": "OK", "details": f"ComfyUI running"}
        else:
            return {"status": "FAIL", "details": f"HTTP {resp.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "FAIL", "details": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"status": "FAIL", "details": "Connection refused"}
    except Exception as e:
        return {"status": "FAIL", "details": str(e)[:50]}


def check_vlm_health(instance_id, url):
    """Check VLM instance health."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            if models:
                model_name = models[0].get("name", "unknown")
                return {"status": "OK", "details": f"Ollama running, model: {model_name}"}
            else:
                return {"status": "OK", "details": "Ollama running, no models"}
        else:
            return {"status": "FAIL", "details": f"HTTP {resp.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "FAIL", "details": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"status": "FAIL", "details": "Connection refused"}
    except Exception as e:
        return {"status": "FAIL", "details": str(e)[:50]}


def cmd_health(args):
    """Check health of all instances."""
    state = load_state()

    klein_instances = state.get("klein_instances", [])
    vlm_instances = state.get("vlm_instances", [])

    if not klein_instances and not vlm_instances:
        print("No instances to check. Run 'launch' first.")
        return

    print("=" * 60)
    print("FLEET HEALTH CHECK")
    print("=" * 60)

    # Check Klein fleet
    if klein_instances:
        print("\nKLEIN FLEET HEALTH")
        klein_results = []

        with ThreadPoolExecutor(max_workers=min(len(klein_instances), 20)) as executor:
            futures = {}
            for inst in klein_instances:
                iid = inst["id"]
                url = inst.get("url", "")
                if url:
                    future = executor.submit(check_klein_health, iid, url)
                    futures[future] = iid

            for future in as_completed(futures):
                iid = futures[future]
                try:
                    result = future.result()
                    klein_results.append((iid, result))
                except Exception as e:
                    klein_results.append((iid, {"status": "ERROR", "details": str(e)[:50]}))

        # Print results
        for iid, result in sorted(klein_results):
            status = result["status"]
            details = result["details"]
            print(f"  Instance {iid}: {status} ({details})")

    # Check VLM fleet
    if vlm_instances:
        print("\nVLM FLEET HEALTH")
        vlm_results = []

        with ThreadPoolExecutor(max_workers=min(len(vlm_instances), 20)) as executor:
            futures = {}
            for inst in vlm_instances:
                iid = inst["id"]
                url = inst.get("url", "")
                if url:
                    future = executor.submit(check_vlm_health, iid, url)
                    futures[future] = iid

            for future in as_completed(futures):
                iid = futures[future]
                try:
                    result = future.result()
                    vlm_results.append((iid, result))
                except Exception as e:
                    vlm_results.append((iid, {"status": "ERROR", "details": str(e)[:50]}))

        # Print results
        for iid, result in sorted(vlm_results):
            status = result["status"]
            details = result["details"]
            print(f"  Instance {iid}: {status} ({details})")

    # Summary
    print("\n" + "=" * 60)
    if klein_instances:
        klein_ok = sum(1 for _, r in klein_results if r["status"] == "OK")
        print(f"Klein: {klein_ok}/{len(klein_instances)} healthy")
    if vlm_instances:
        vlm_ok = sum(1 for _, r in vlm_results if r["status"] == "OK")
        print(f"VLM:   {vlm_ok}/{len(vlm_instances)} healthy")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Fleet Manager — Unified Klein (ComfyUI) and VLM (Ollama) fleet management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="command")

    # Launch command
    launch_p = sub.add_parser("launch", help="Launch fleet instances")
    launch_p.add_argument("--klein", type=int, default=10, help="Number of Klein instances (default: 10)")
    launch_p.add_argument("--vlm", type=int, default=10, help="Number of VLM instances (default: 10)")
    launch_p.add_argument("--model", choices=["7b", "32b"], default="32b", help="VLM model size (default: 32b)")
    launch_p.add_argument("--klein-only", action="store_true", help="Launch only Klein instances")
    launch_p.add_argument("--vlm-only", action="store_true", help="Launch only VLM instances")

    # Status command
    sub.add_parser("status", help="Show fleet status")

    # URLs command
    urls_p = sub.add_parser("urls", help="Print fleet URLs")
    urls_p.add_argument("--klein", action="store_true", help="Print Klein URLs only")
    urls_p.add_argument("--vlm", action="store_true", help="Print VLM URLs only")
    urls_p.add_argument("--all", action="store_true", help="Print all URLs (default)")

    # Destroy command
    destroy_p = sub.add_parser("destroy", help="Destroy fleet instances")
    destroy_p.add_argument("--klein-only", action="store_true", help="Destroy only Klein instances")
    destroy_p.add_argument("--vlm-only", action="store_true", help="Destroy only VLM instances")

    # Scale command
    scale_p = sub.add_parser("scale", help="Scale fleet up or down")
    scale_p.add_argument("--klein", type=int, help="Scale Klein fleet to N instances")
    scale_p.add_argument("--vlm", type=int, help="Scale VLM fleet to N instances")

    # Configure command
    configure_p = sub.add_parser("configure", help="Push configuration changes to instances")
    configure_p.add_argument("--vlm-model", type=str, help="Pull new Ollama model on VLM instances")
    configure_p.add_argument("--parallel", type=int, help="Set OLLAMA_NUM_PARALLEL on VLM instances")
    configure_p.add_argument("--update-toolkit", action="store_true", help="Git pull toolkit on Klein instances")
    configure_p.add_argument("--klein-only", action="store_true", help="Only configure Klein instances")
    configure_p.add_argument("--vlm-only", action="store_true", help="Only configure VLM instances")

    # Run command
    run_p = sub.add_parser("run", help="Execute command across instances")
    run_p.add_argument("command", type=str, help="Shell command to run (in quotes)")
    run_p.add_argument("--klein-only", action="store_true", help="Only run on Klein instances")
    run_p.add_argument("--vlm-only", action="store_true", help="Only run on VLM instances")
    run_p.add_argument("--timeout", type=int, default=120, help="Command timeout in seconds (default: 120)")

    # Health command
    sub.add_parser("health", help="Check health of all instances")

    args = parser.parse_args()

    commands = {
        "launch": cmd_launch,
        "status": cmd_status,
        "urls": cmd_urls,
        "destroy": cmd_destroy,
        "scale": cmd_scale,
        "configure": cmd_configure,
        "run": cmd_run,
        "health": cmd_health,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

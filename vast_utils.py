#!/usr/bin/env python3
"""
Vast.ai Fleet Utilities — Shared functionality for fleet management.

This module provides all shared Vast.ai functionality used across the fleet system:
- run_vast() with retry logic and proper timeout handling
- parse_instance_id() with multiple fallback strategies
- get_instance_info() and get_instance_port() with format handling
- search_offers() with proper query escaping
- build_onstart_script() with base64 encoding for multi-line scripts
- State management (load_state, save_state) with atomic writes
- Instance destruction (destroy_instance, destroy_instances_parallel)
- Health checks (check_health_comfyui, check_health_ollama)
- Constants and configuration for all fleet types

Usage:
    from vast_utils import run_vast, parse_instance_id, search_offers

    # Search for instances
    offers = search_offers(VAST_SEARCH_KLEIN)

    # Create instance
    out, rc = run_vast(["create", "instance", str(offer_id), ...])
    instance_id = parse_instance_id(out)

    # Get instance info
    info = get_instance_info(instance_id)
    port = get_instance_port(info, "18188")
"""

import base64
import json
import logging
import os
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fleet")

# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_DOCKER_IMAGE = "msrcam/klein-fleet:latest"

COMFYUI_PORT = "18188"
OLLAMA_PORT = "11434"

# VLM model names
VLM_MODELS = {
    "7b": "huihui_ai/qwen2.5-vl-abliterated:7b",
    "32b": "huihui_ai/qwen2.5-vl-abliterated:32b"
}

# Vast.ai search criteria for main instance (coordinator)
VAST_SEARCH_MAIN = {
    "gpu_name": ["RTX_3090", "RTX_4090"],
    "disk_space": ">=60",
    "total_flops": ">=30",
    "inet_down": ">=100",
    "reliability": ">=0.95",
    "dph_total": "<=0.30"
}

# Vast.ai search criteria for Klein fleet (ComfyUI workers)
VAST_SEARCH_KLEIN = {
    "gpu_name": ["RTX_3090", "RTX_4090"],
    "disk_space": ">=40",
    "total_flops": ">=30",
    "inet_down": ">=100",
    "reliability": ">=0.95",
    "dph_total": "<=0.30"
}

# Vast.ai search criteria for VLM fleet - 32B model
VAST_SEARCH_VLM_32B = {
    "gpu_ram": ">=22",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=40",
    "dph_total": "<=0.30",
    "gpu_name": ["RTX_3090", "RTX_4090", "RTX_A6000", "A100_SXM4"]
}

# Vast.ai search criteria for VLM fleet - 7B model
VAST_SEARCH_VLM_7B = {
    "gpu_ram": ">=10",
    "num_gpus": "1",
    "inet_down": ">=200",
    "disk_space": ">=20",
    "dph_total": "<=0.15",
    "gpu_name": ["RTX_3060", "RTX_3070", "RTX_3080", "RTX_3090", "RTX_4060", "RTX_4070"]
}

# VLM setup script template
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

# Check if vastai CLI exists at import time
_VASTAI_AVAILABLE = None


def _check_vastai_cli():
    """Check if vastai CLI is available."""
    global _VASTAI_AVAILABLE
    if _VASTAI_AVAILABLE is None:
        try:
            subprocess.run(["vastai", "--version"], capture_output=True, timeout=5)
            _VASTAI_AVAILABLE = True
        except FileNotFoundError:
            _VASTAI_AVAILABLE = False
        except Exception:
            _VASTAI_AVAILABLE = False
    return _VASTAI_AVAILABLE


# ==============================================================================
# CORE VAST.AI FUNCTIONS
# ==============================================================================

def run_vast(args_list: List[str], timeout: int = 60, retries: int = 3) -> Tuple[str, int]:
    """
    Run vastai CLI command with retry logic and exponential backoff.

    Args:
        args_list: Command arguments (e.g., ["search", "offers", "--raw"])
        timeout: Command timeout in seconds (default: 60, use 120 for create operations)
        retries: Number of retry attempts (default: 3)

    Returns:
        Tuple of (output_string, return_code)

    Raises:
        FileNotFoundError: If vastai CLI is not installed
        subprocess.TimeoutExpired: If command times out after all retries

    Example:
        out, rc = run_vast(["search", "offers", "--raw"])
        if rc != 0:
            logger.error(f"Search failed: {out}")
    """
    if not _check_vastai_cli():
        raise FileNotFoundError(
            "vastai CLI not found. Install with: pip install vastai"
        )

    cmd = ["vastai"] + args_list

    for attempt in range(retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout.strip(), result.returncode

        except subprocess.TimeoutExpired as e:
            delay = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
            if attempt < retries - 1:
                logger.warning(
                    f"Command timed out (attempt {attempt + 1}/{retries}), "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Command timed out after {retries} attempts")
                raise

        except Exception as e:
            logger.error(f"Unexpected error running vastai command: {e}")
            return str(e), -1

    return "", -1


def parse_instance_id(output: str) -> Optional[int]:
    """
    Parse instance ID from vastai create output.

    Tries multiple strategies in order:
    1. JSON parse (proper format)
    2. Regex for 'new_contract': N (dict format)
    3. Regex for "new_contract": N (JSON format)

    Does NOT fall back to "first bare number" which can incorrectly
    parse error codes like 404 as instance IDs.

    Args:
        output: Raw output from vastai create instance command

    Returns:
        Instance ID as integer, or None if parsing fails

    Example:
        out, rc = run_vast(["create", "instance", ...])
        instance_id = parse_instance_id(out)
        if instance_id is None:
            logger.warning(f"Failed to parse instance ID from: {out[:100]}")
    """
    # Try proper JSON parse first
    try:
        data = json.loads(output)
        if isinstance(data, dict) and "new_contract" in data:
            return int(data["new_contract"])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Try regex for 'new_contract': N (dict format)
    m = re.search(r"'new_contract':\s*(\d+)", output)
    if m:
        return int(m.group(1))

    # Try regex for "new_contract": N (JSON format)
    m = re.search(r'"new_contract":\s*(\d+)', output)
    if m:
        return int(m.group(1))

    # NO fallback to first bare number - that's how 404 becomes an instance ID
    logger.warning(f"Could not parse instance ID from output: {output[:200]}")
    return None


def get_instance_info(instance_id: int) -> Optional[Dict[str, Any]]:
    """
    Get instance details from Vast.ai.

    Args:
        instance_id: Instance ID to query

    Returns:
        Dict with instance info (status, IP, ports, etc.) or None if failed

    Example:
        info = get_instance_info(12345)
        if info:
            status = info.get("actual_status", "unknown")
            ip = info.get("public_ipaddr", "")
    """
    try:
        out, rc = run_vast(["show", "instance", str(instance_id), "--raw"])
        if rc != 0:
            logger.warning(f"Failed to get info for instance {instance_id}: {out}")
            return None

        return json.loads(out)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse instance info JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting instance info: {e}")
        return None


def get_instance_port(info: Dict[str, Any], internal_port: str) -> Optional[str]:
    """
    Extract mapped port from instance info.

    Handles ALL Vast.ai port formats:
    - String: "12345"
    - List of dicts: [{"HostPort": "12345"}]
    - Single dict: {"HostPort": "12345"}
    - List with IP: [{"HostPort": "12345", "HostIp": "0.0.0.0"}]

    Args:
        info: Instance info dict from get_instance_info()
        internal_port: Internal port to look up (e.g., "18188", "11434")

    Returns:
        Port as string, or None if not found

    Example:
        info = get_instance_info(instance_id)
        comfy_port = get_instance_port(info, "18188")
        if comfy_port:
            url = f"http://{info['public_ipaddr']}:{comfy_port}"
    """
    if not info:
        return None

    ports = info.get("ports", {})
    port_key = f"{internal_port}/tcp"

    if port_key not in ports:
        return None

    port_info = ports[port_key]

    # Handle string format: "12345"
    if isinstance(port_info, str):
        return port_info

    # Handle list format: [{"HostPort": "12345"}] or [{"HostPort": "12345", "HostIp": "0.0.0.0"}]
    if isinstance(port_info, list) and port_info:
        if isinstance(port_info[0], dict):
            return port_info[0].get("HostPort")
        return str(port_info[0])

    # Handle dict format: {"HostPort": "12345"}
    if isinstance(port_info, dict):
        return port_info.get("HostPort")

    logger.warning(f"Unknown port format for {port_key}: {port_info}")
    return None


def search_offers(criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search Vast.ai for instances matching criteria.

    Args:
        criteria: Dict of search criteria
            - Lists use bracket syntax: gpu_name: ["RTX_3090", "RTX_4090"]
            - Values with operators: disk_space: ">=40"
            - Simple values: num_gpus: "1"

    Returns:
        List of offer dicts, sorted by price (dph_total)

    Example:
        criteria = {
            "gpu_name": ["RTX_3090", "RTX_4090"],
            "disk_space": ">=40",
            "dph_total": "<=0.30"
        }
        offers = search_offers(criteria)
        if offers:
            best_offer = offers[0]  # Cheapest
    """
    query_parts = []
    for key, val in criteria.items():
        if isinstance(val, list):
            # Use bracket syntax for lists
            query_parts.append(f"{key} in [{','.join(val)}]")
        else:
            # Pass operators through directly
            query_parts.append(f"{key} {val}")

    query = " ".join(query_parts)

    try:
        out, rc = run_vast(["search", "offers", "--raw", query])
        if rc != 0:
            logger.error(f"Search failed: {out}")
            return []

        offers = json.loads(out)
        # Sort by price (cheapest first)
        offers.sort(key=lambda x: x.get("dph_total", 999))
        return offers

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse search results: {e}")
        return []
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


def build_onstart_script(script_content: str) -> str:
    """
    Encode multi-line bash script for --onstart-cmd parameter.

    Multi-line scripts cannot be passed directly as CLI args.
    This function base64 encodes the script and wraps it in a
    bash command that decodes and executes it.

    Args:
        script_content: Multi-line bash script

    Returns:
        Encoded command string safe for --onstart-cmd

    Example:
        script = '''#!/bin/bash
        echo "Starting..."
        apt-get update
        apt-get install -y vim
        '''
        onstart_cmd = build_onstart_script(script)
        run_vast(["create", "instance", ..., "--onstart-cmd", onstart_cmd])
    """
    encoded = base64.b64encode(script_content.encode()).decode()
    return f"bash -c 'echo {encoded} | base64 -d | bash'"


# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================

def load_state(state_file: str) -> Dict[str, Any]:
    """
    Load state from JSON file.

    Returns empty default state if file doesn't exist.
    Handles corrupted JSON gracefully.

    Args:
        state_file: Path to state file

    Returns:
        State dict with default structure

    Example:
        state = load_state(".fleet_state.json")
        instances = state.get("instances", [])
    """
    default_state = {
        "instances": [],
        "klein_instances": [],
        "vlm_instances": [],
        "main_instance": None,
        "status": "idle",
        "launched_at": "",
        "total_cost_per_hour": 0.0,
        "model": ""
    }

    if not os.path.exists(state_file):
        return default_state.copy()

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)

        # Merge with defaults to ensure all keys exist
        merged = default_state.copy()
        merged.update(state)
        return merged

    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted state file {state_file}: {e}. Using defaults.")
        return default_state.copy()
    except Exception as e:
        logger.error(f"Error loading state: {e}. Using defaults.")
        return default_state.copy()


def save_state(state: Dict[str, Any], state_file: str) -> bool:
    """
    Save state to JSON file atomically.

    Writes to temp file first, then atomically replaces the original.
    This prevents corruption if the script is killed during write.

    Args:
        state: State dict to save
        state_file: Path to state file

    Returns:
        True if successful, False otherwise

    Example:
        state["status"] = "running"
        save_state(state, ".fleet_state.json")
    """
    try:
        temp_file = state_file + ".tmp"

        # Write to temp file
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Atomically replace
        os.replace(temp_file, state_file)
        return True

    except Exception as e:
        logger.error(f"Error saving state: {e}")
        return False


# ==============================================================================
# INSTANCE MANAGEMENT
# ==============================================================================

def destroy_instance(instance_id: int) -> bool:
    """
    Destroy a single instance.

    Args:
        instance_id: Instance ID to destroy

    Returns:
        True if successful, False otherwise

    Example:
        if destroy_instance(12345):
            logger.info("Instance destroyed")
    """
    try:
        out, rc = run_vast(["destroy", "instance", str(instance_id)], timeout=15)
        return rc == 0
    except Exception as e:
        logger.error(f"Error destroying instance {instance_id}: {e}")
        return False


def destroy_instances_parallel(instance_ids: List[int], max_workers: int = 10) -> Dict[int, bool]:
    """
    Destroy multiple instances in parallel.

    Args:
        instance_ids: List of instance IDs to destroy
        max_workers: Max parallel workers (default: 10)

    Returns:
        Dict mapping instance_id -> success_bool

    Example:
        results = destroy_instances_parallel([12345, 12346, 12347])
        succeeded = sum(results.values())
        logger.info(f"Destroyed {succeeded}/{len(results)} instances")
    """
    results = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(instance_ids))) as executor:
        futures = {
            executor.submit(destroy_instance, iid): iid
            for iid in instance_ids
        }

        for future in as_completed(futures):
            iid = futures[future]
            try:
                success = future.result()
                results[iid] = success
            except Exception as e:
                logger.error(f"Error destroying instance {iid}: {e}")
                results[iid] = False

    return results


# ==============================================================================
# HEALTH CHECKS
# ==============================================================================

def check_health_comfyui(url: str, timeout: int = 5) -> bool:
    """
    Check if ComfyUI is healthy.

    Args:
        url: Base URL of ComfyUI instance (e.g., "http://1.2.3.4:12345")
        timeout: Request timeout in seconds

    Returns:
        True if healthy, False otherwise

    Example:
        url = f"http://{ip}:{port}"
        if check_health_comfyui(url):
            logger.info("ComfyUI is ready")
    """
    try:
        req = urllib.request.Request(
            f"{url}/system_stats",
            headers={'User-Agent': 'vast-utils/1.0'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception as e:
        logger.debug(f"ComfyUI health check failed for {url}: {e}")
        return False


def check_health_ollama(url: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Check if Ollama is healthy and return model info.

    Args:
        url: Base URL of Ollama instance (e.g., "http://1.2.3.4:12345")
        timeout: Request timeout in seconds

    Returns:
        Dict with model info if healthy, None otherwise
        Dict format: {"healthy": True, "models": ["model1", "model2"]}

    Example:
        url = f"http://{ip}:{port}"
        health = check_health_ollama(url)
        if health:
            logger.info(f"Ollama ready with models: {health['models']}")
    """
    try:
        req = urllib.request.Request(
            f"{url}/api/tags",
            headers={'User-Agent': 'vast-utils/1.0'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                models = [m.get("name", "unknown") for m in data.get("models", [])]
                return {
                    "healthy": True,
                    "models": models
                }
    except Exception as e:
        logger.debug(f"Ollama health check failed for {url}: {e}")

    return None


# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

# Check vastai CLI availability at import time
if not _check_vastai_cli():
    logger.warning(
        "vastai CLI not found. Install with: pip install vastai\n"
        "Some functions will fail until vastai is installed."
    )

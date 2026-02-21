"""Convert UI workflow to API format and queue on ComfyUI for end-to-end test."""
import json
import urllib.request
import sys

# Load workflow
with open("/workspace/ComfyUI/user/default/workflows/full_lora_pipeline_expert.json") as f:
    wf = json.load(f)

# Get node specs
resp = urllib.request.urlopen("http://localhost:18188/object_info")
specs = json.loads(resp.read())

# Build link map: link_id -> (from_node_id, from_slot)
link_map = {}
for link_entry in wf["links"]:
    link_id, from_id, from_slot, to_id, to_slot, dtype = link_entry
    link_map[link_id] = (from_id, from_slot)

# For each node, build map of which input names are linked
node_links = {}
for node in wf["nodes"]:
    nid = node["id"]
    node_links[nid] = {}
    if "inputs" in node:
        for inp in node["inputs"]:
            if inp.get("link") is not None:
                lid = inp["link"]
                if lid in link_map:
                    src_node, src_slot = link_map[lid]
                    node_links[nid][inp["name"]] = [str(src_node), src_slot]

# Types that are link-only (never widgets)
LINK_ONLY_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET", "CLIP_VISION",
    "CLIP_VISION_OUTPUT", "STYLE_MODEL", "GLIGEN", "UPSCALE_MODEL",
    "TAESD", "PHOTOMAKER", "EMBEDS",
}


def is_widget(input_spec):
    t = input_spec[0]
    if isinstance(t, list):  # combo dropdown
        return True
    if isinstance(t, str):
        if t in LINK_ONLY_TYPES:
            return False
        if len(input_spec) > 1 and isinstance(input_spec[1], dict):
            return True
        if t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
            return True
    return False


# Convert to API format
prompt = {}
errors = []

for node in wf["nodes"]:
    ntype = node["type"]
    nid = node["id"]
    title = node.get("title", ntype)

    if ntype in ("Reroute", "Note", "PrimitiveNode"):
        continue
    if ntype not in specs:
        errors.append("Node {} ({}): type {} not in ComfyUI".format(nid, title, ntype))
        continue

    spec = specs[ntype]
    required = spec.get("input", {}).get("required", {})
    optional = spec.get("input", {}).get("optional", {})
    input_order = spec.get("input_order", {})

    req_names = input_order.get("required", list(required.keys()))
    opt_names = input_order.get("optional", list(optional.keys()))

    all_specs = {}
    all_specs.update(required)
    all_specs.update(optional)

    linked = node_links.get(nid, {})
    widgets = node.get("widgets_values", []) or []
    widx = 0

    inputs_dict = {}

    for name in req_names + opt_names:
        if name in linked:
            inputs_dict[name] = linked[name]
            # Linked widget inputs still consume a widget_values slot
            if name in all_specs and is_widget(all_specs[name]):
                widx += 1
        elif name in all_specs:
            ispec = all_specs[name]
            if is_widget(ispec):
                if widx < len(widgets):
                    val = widgets[widx]
                    inputs_dict[name] = val
                    widx += 1
                else:
                    # Use default
                    if len(ispec) > 1 and isinstance(ispec[1], dict):
                        inputs_dict[name] = ispec[1].get("default", "")
                    elif isinstance(ispec[0], list) and len(ispec[0]) > 0:
                        inputs_dict[name] = ispec[0][0]
            # Link-only inputs that aren't connected: skip (optional)

    prompt[str(nid)] = {
        "class_type": ntype,
        "inputs": inputs_dict,
    }

if errors:
    print("CONVERSION ERRORS: {}".format(len(errors)))
    for e in errors:
        print("  " + e)
    sys.exit(1)

print("Converted {} nodes to API format".format(len(prompt)))

# Save API format for inspection
with open("/tmp/workflow_api.json", "w") as f:
    json.dump(prompt, f, indent=2)
print("Saved API format to /tmp/workflow_api.json")

# Queue it
data = json.dumps({"prompt": prompt}).encode()
req = urllib.request.Request(
    "http://localhost:18188/prompt",
    data=data,
    headers={"Content-Type": "application/json"},
)
try:
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    pid = result.get("prompt_id", "?")
    print("QUEUED OK: prompt_id={}".format(pid))
    print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print("QUEUE FAILED (HTTP {}):".format(e.code))
    try:
        err = json.loads(body)
        if "node_errors" in err:
            ne = err["node_errors"]
            if ne:
                for nid_str, nerr in ne.items():
                    ct = prompt.get(nid_str, {}).get("class_type", "?")
                    print("  Node {} ({}):".format(nid_str, ct))
                    for msg in nerr.get("errors", []):
                        print("    - {}".format(msg.get("message", str(msg))))
            else:
                print("  (no node errors, check error field)")
        if "error" in err:
            emsg = err["error"]
            if isinstance(emsg, dict):
                print("  Error: {}".format(emsg.get("message", str(emsg))))
            else:
                print("  Error: {}".format(emsg))
    except Exception as parse_err:
        print("  Parse error: {}".format(parse_err))
        print("  Raw: {}".format(body[:1000]))

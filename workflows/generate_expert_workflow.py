"""
Generate the Ultimate LoRA Dataset Pipeline — Expert Edition workflow with Klein Fleet.
Run on the ComfyUI instance to generate proper graph-format workflow JSON.

COMPLETE END-TO-END PIPELINE:
1. INPUT: Load video frames or browse server images
2. QUALITY GATE: BRISQUE quality scoring + perceptual dedup
3. KLEIN FLEET EDIT: Distribute Flux2 Klein editing across multiple ComfyUI instances
4. CAPTION: VLM captioning with fleet support
5. RESIZE + SAVE: Final dataset output

FLEET MODE — 10-20x FASTER:
- Klein Fleet: Distributes image editing across multiple ComfyUI instances
  - Launch instances with Klein models loaded
  - Paste URLs into comfyui_urls field (comma-separated)
  - Set workers=4-8 per instance
  - Images are round-robin distributed
  - Example: 4 instances × 4 workers = 16 parallel edits

- VLM Fleet: Distributes captioning/detection across multiple Ollama instances
  - Run: python fleet_caption.py launch --count 10 --model 32b
  - Paste URLs into ollama_urls field
  - Set workers = number of instances
  - Single-model fleet: All instances run same model

Widget values are specified as dicts {name: value} and automatically
ordered at build time using ComfyUI's /object_info API. This eliminates
widget alignment bugs entirely.
"""
import json
import urllib.request

# Types that are link-only (never appear as widgets)
LINK_ONLY_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET", "CLIP_VISION",
    "CLIP_VISION_OUTPUT", "STYLE_MODEL", "GLIGEN", "UPSCALE_MODEL",
    "TAESD", "PHOTOMAKER", "EMBEDS",
}


def _load_specs(url="http://localhost:18188/object_info"):
    """Load node specs from ComfyUI API."""
    resp = urllib.request.urlopen(url)
    return json.loads(resp.read())


def _is_widget(input_spec):
    """Check if an input spec is a widget (vs link-only)."""
    t = input_spec[0]
    if isinstance(t, list):  # combo
        return True
    if isinstance(t, str):
        if t in LINK_ONLY_TYPES:
            return False
        return True  # STRING, INT, FLOAT, BOOLEAN, or unknown
    return False


def _get_default(input_spec):
    """Get default value for a widget input."""
    t = input_spec[0]
    if isinstance(t, list):  # combo — first option
        return t[0] if t else ""
    if len(input_spec) > 1 and isinstance(input_spec[1], dict):
        return input_spec[1].get("default", "")
    # Fallback defaults by type
    if t == "INT":
        return 0
    if t == "FLOAT":
        return 0.0
    if t == "BOOLEAN":
        return False
    if t == "STRING":
        return ""
    return ""


class WorkflowBuilder:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self._next_id = 1
        self._next_link = 1

    def _id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def note(self, text, x, y, w=420, h=300, title="Note", color="#332922"):
        nid = self._id()
        self.nodes.append({
            "id": nid, "type": "Note",
            "pos": [x, y], "size": [w, h],
            "flags": {}, "order": nid, "mode": 0,
            "inputs": [], "outputs": [],
            "properties": {"text": text},
            "widgets_values": [text],
            "title": title,
            "color": color, "bgcolor": "#1a1a1a",
        })
        return nid

    def node(self, class_type, x, y, w=330, h=200, title=None,
             widgets=None, inputs=None, outputs=None, mode=0):
        """Create a node. widgets is a dict {input_name: value}."""
        nid = self._id()
        self.nodes.append({
            "id": nid, "type": class_type,
            "pos": [x, y], "size": [w, h],
            "flags": {}, "order": nid, "mode": mode,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "properties": {"Node name for S&R": class_type},
            "_widget_dict": widgets or {},  # stored as dict, resolved at build time
            "widgets_values": [],  # populated by finalize()
            "title": title or class_type,
        })
        return nid

    def link(self, from_id, from_slot, to_id, to_slot, type_str="*"):
        lid = self._next_link
        self._next_link += 1
        self.links.append([lid, from_id, from_slot, to_id, to_slot, type_str])
        for n in self.nodes:
            if n["id"] == from_id:
                while len(n["outputs"]) <= from_slot:
                    n["outputs"].append({"name": "out", "type": type_str, "links": [], "slot_index": len(n["outputs"])})
                n["outputs"][from_slot]["links"].append(lid)
            if n["id"] == to_id:
                while len(n["inputs"]) <= to_slot:
                    n["inputs"].append({"name": "in", "type": type_str, "link": None, "slot_index": len(n["inputs"])})
                n["inputs"][to_slot]["link"] = lid
        return lid

    def group(self, title, x, y, w, h, color="#335533"):
        self.groups.append({
            "title": title,
            "bounding": [x, y, w, h],
            "color": color,
            "font_size": 24,
        })

    def finalize(self, specs):
        """Resolve all widget dicts to ordered arrays using live API specs.
        This is the key function that prevents widget alignment bugs."""
        errors = []
        for node in self.nodes:
            ntype = node["type"]
            if ntype in ("Note",):
                continue  # Notes already have correct widgets_values

            wd = node.pop("_widget_dict", {})
            if ntype not in specs:
                # Unknown node type — keep empty widgets
                if wd:
                    errors.append("Node {} ({}): type not in specs, can't resolve widgets".format(
                        node["id"], node.get("title", ntype)))
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

            # Build ordered widgets_values array
            widgets_values = []
            for name in req_names + opt_names:
                if name not in all_specs:
                    continue
                ispec = all_specs[name]
                if not _is_widget(ispec):
                    continue  # LINK-only types — no widget value

                # Every widget-capable input gets a slot, even if linked
                if name in wd:
                    widgets_values.append(wd[name])
                else:
                    widgets_values.append(_get_default(ispec))

            node["widgets_values"] = widgets_values

            # Validate: warn about widget dict keys that don't match any input
            for key in wd:
                if key not in all_specs:
                    errors.append("Node {} ({}): widget '{}' not in {} inputs".format(
                        node["id"], node.get("title", ntype), key, ntype))

        return errors

    def build(self):
        # Load specs and resolve widgets
        specs = _load_specs()
        errors = self.finalize(specs)
        if errors:
            print("WARNINGS:")
            for e in errors:
                print("  " + e)

        return {
            "last_node_id": self._next_id - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.5, "offset": [-50, 50]}},
            "version": 0.4,
        }


def generate():
    B = WorkflowBuilder()

    # =====================================================
    # LAYOUT CONSTANTS
    # =====================================================
    NOTE_W = 440
    NODE_W = 340
    PREV_W = 260        # Preview node width
    GAP = 30
    COL0 = 50           # Notes
    COL1 = COL0 + NOTE_W + GAP  # First node
    COL2 = COL1 + NODE_W + GAP  # Second node
    COL3 = COL2 + NODE_W + GAP  # Third node
    COL4 = COL3 + NODE_W + GAP  # Fourth node
    COL5 = COL4 + NODE_W + GAP  # Fifth node

    # Group bounding helpers — tight fit with 30px padding
    GRP_PAD = 30
    GRP_X = COL0 - GRP_PAD          # Left edge of all groups
    GRP_W2 = (COL2 + PREV_W + GRP_PAD) - GRP_X   # Groups ending at COL2 preview
    GRP_W3 = (COL3 + PREV_W + GRP_PAD) - GRP_X   # Groups ending at COL3 preview
    GRP_W3F = (COL3 + 360 + GRP_PAD) - GRP_X     # Groups ending at COL3 wide log
    GRP_W4 = (COL4 + PREV_W + GRP_PAD) - GRP_X   # Groups ending at COL4 preview

    # =====================================================
    # FLEET MODE INSTRUCTIONS  (y=0)
    # =====================================================
    FLEET_Y = 0
    B.note("""⚡ FLEET MODE — 10-20x FASTER PROCESSING ⚡

TWO TYPES OF FLEET:

1. KLEIN FLEET (ComfyUI instances for image editing)
   - Launch multiple ComfyUI instances with Klein models loaded
   - Each instance needs: bigLove_klein1_fp8.safetensors, qwen3_8b_abliterated_v2-fp8mixed.safetensors, flux2-vae.safetensors
   - Paste URLs into QVL_KleinFleet comfyui_urls field (comma-separated)
   - Example: "http://1.2.3.4:18188,http://5.6.7.8:18188,http://9.10.11.12:18188"
   - Set workers=4-8 (concurrent edits PER instance)
   - Speed: 4 instances × 4 workers = 16 parallel edits
   - Cost: ~$0.50-1.00/hr per RTX 3090 instance

2. VLM FLEET (Ollama instances for captioning/detection)
   - Run: python fleet_caption.py launch --count 10 --model 32b
   - Copy the URLs output
   - Paste into ollama_urls field of Caption/Detect/Filter/Analyze nodes
   - Set workers = number of instances (e.g., workers=10)
   - Cost: ~$1-3 total for 500 images at 32B quality
   - When done: python fleet_caption.py destroy

SINGLE INSTANCE MODE:
- Klein: Leave comfyui_urls empty, uses local ComfyUI only
- VLM: Set workers=1, leave ollama_urls empty, uses ollama_url
- Local parallel: Set workers=2-4, set OLLAMA_NUM_PARALLEL=4 on server

NO QUALITY LOSS: Fleet mode uses same models, just more instances.
SPEEDUP: 100 images with 32B captions: 50 min single → 5 min with 10 instances""",
        COL0, FLEET_Y, 900, 420, title="⚡ FLEET MODE GUIDE", color="#114488")

    # =====================================================
    # SECTION 1: INPUT SOURCES  (y=480)
    # =====================================================
    SEC1_Y = 480
    B.group("1. INPUT SOURCES", GRP_X, SEC1_Y - 40, GRP_W3, 440, "#2d8a37")

    B.note("""INPUT SOURCES — Choose how to load your images

OPTION A: Load Video Frames
Best for: Extracting training data directly from videos.
• scene_detect: Only grabs frames when the scene changes
  (fewer duplicates, slower but smarter)
• fixed_fps: Grabs at constant rate regardless of content
  (more frames, more duplicates)
• max_frames: 100=quick test, 500=good, 0=everything

OPTION B: Server Image Browse
Best for: Pre-extracted frames, curated collections,
results from previous pipeline runs.
• image_path: Absolute server path to image folder
• Quick visual check before pipeline

START SMALL: Test with 50-100 frames first!""",
        COL0, SEC1_Y, NOTE_W, 380)

    # Load Video Frames node
    load_vid = B.node("QVL_LoadVideoFrames", COL1, SEC1_Y, NODE_W, 160,
        title="Load Video Frames",
        widgets={"video": "** ALL VIDEOS **", "extraction_mode": "scene_detect", "fps": 1.0, "scene_threshold": 0.1, "max_frames": 500, "dedup_on_extract": True, "dedup_threshold": 8, "video_dir": "/workspace/videos", "output_folder": "/workspace/frames", "min_size": 256},
        outputs=[
            {"name": "images", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "filenames", "type": "STRING", "links": [], "slot_index": 1},
            {"name": "count", "type": "INT", "links": [], "slot_index": 2},
        ])

    # Server Image node
    load_img = B.node("QVL_ServerImage", COL2, SEC1_Y, NODE_W, 160,
        title="Browse Server Image",
        widgets={"image_path": "/workspace/scored_keep/"},
        outputs=[
            {"name": "images", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "filenames", "type": "STRING", "links": [], "slot_index": 1},
            {"name": "count", "type": "INT", "links": [], "slot_index": 2},
            {"name": "mask", "type": "MASK", "links": [], "slot_index": 3},
        ])

    # Preview for loaded batch
    prev_loaded = B.node("PreviewImage", COL3, SEC1_Y, PREV_W, 60,
        title="Preview: Loaded Batch",
        inputs=[{"name": "images", "type": "IMAGE", "link": None}])
    B.link(load_vid, 0, prev_loaded, 0, "IMAGE")

    # =====================================================
    # SECTION 2: QUALITY GATE  (y=980)
    # =====================================================
    SEC2_Y = 980
    B.group("2. QUALITY GATE + DEDUP", GRP_X, SEC2_Y - 40, GRP_W4, 540, "#2d5fba")

    B.note("""QUALITY FILTERING — Remove technically bad images

QVL Quality Score — Instant local analysis (no VLM)
Scores each image 0-100 by sharpness, brightness,
contrast, and resolution. No GPU/model needed.
• min_score: Threshold to pass.
  20=keep almost all, 30=reasonable, 50=strict
  TIP: Run once with min_score=0 and check the score_log
  to see your score distribution before setting threshold.
• min_size: Minimum dimension in pixels (256=standard)

QVL Filter — VLM-based keep/reject (OPTIONAL)
Uses VLM to judge each image for quality.
• FLEET-ENABLED: workers + ollama_urls for parallelism
• Slower than QualityScore but smarter
• Leave workers=1 for single instance mode

QVL Perceptual Dedup — Remove near-duplicate images
Uses perceptual hashing to find similar images.
• threshold: Similarity distance (0-30)
  6=RECOMMENDED for video frames (somewhat similar)
  3=very similar only, 10+=aggressive

ORDER: QualityScore → Filter (optional) → Dedup""",
        COL0, SEC2_Y, NOTE_W, 480)

    quality = B.node("QVL_QualityScore", COL1, SEC2_Y, NODE_W, 160,
        title="Quality Score (skip bad frames)",
        widgets={"min_score": 30.0, "min_size": 256},
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
        ],
        outputs=[
            {"name": "passed", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "failed", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "passed_filenames", "type": "STRING", "links": [], "slot_index": 2},
            {"name": "score_log", "type": "STRING", "links": [], "slot_index": 3},
            {"name": "passed_count", "type": "INT", "links": [], "slot_index": 4},
            {"name": "failed_count", "type": "INT", "links": [], "slot_index": 5},
        ])
    B.link(load_vid, 0, quality, 0, "IMAGE")

    vlm_filter = B.node("QVL_Filter", COL2, SEC2_Y, NODE_W, 180,
        title="VLM Filter (optional)",
        widgets={
            "model": "huihui_ai/qwen2.5-vl-abliterated:7b",
            "ollama_url": "http://127.0.0.1:11434",
            "min_quality": 5,
            "api_type": "ollama",
            "workers": 1,
            "ollama_urls": "",
        },
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
        ],
        outputs=[
            {"name": "kept", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "rejected", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "kept_filenames", "type": "STRING", "links": [], "slot_index": 2},
            {"name": "filter_log", "type": "STRING", "links": [], "slot_index": 3},
            {"name": "kept_count", "type": "INT", "links": [], "slot_index": 4},
            {"name": "rejected_count", "type": "INT", "links": [], "slot_index": 5},
        ])
    B.link(quality, 0, vlm_filter, 0, "IMAGE")

    dedup = B.node("QVL_Dedup", COL3, SEC2_Y, NODE_W, 140,
        title="Perceptual Dedup",
        widgets={"threshold": 6, "hash_size": 16},
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
        ],
        outputs=[
            {"name": "unique", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "dupes", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "unique_filenames", "type": "STRING", "links": [], "slot_index": 2},
            {"name": "unique_count", "type": "INT", "links": [], "slot_index": 3},
            {"name": "dupe_count", "type": "INT", "links": [], "slot_index": 4},
        ])
    B.link(vlm_filter, 0, dedup, 0, "IMAGE")

    prev_quality = B.node("PreviewImage", COL4, SEC2_Y, PREV_W, 60,
        title="Preview: Passed Quality",
        inputs=[{"name": "images", "type": "IMAGE", "link": None}])
    B.link(dedup, 0, prev_quality, 0, "IMAGE")

    log_quality = B.node("PreviewAny", COL4, SEC2_Y + 120, PREV_W, 80,
        title="Quality Score Log",
        inputs=[{"name": "source", "type": "*", "link": None}])
    B.link(quality, 3, log_quality, 0, "STRING")

    # =====================================================
    # SECTION 3: KLEIN FLEET EDIT  (y=1580)
    # =====================================================
    SEC3_Y = 1580
    B.group("3. KLEIN FLEET IMAGE EDIT [OPTIONAL]", GRP_X, SEC3_Y - 40, GRP_W3, 600, "#8030ba")

    B.note("""⚡ KLEIN FLEET — Distributed Flux2 Klein editing

QVL_KleinFleet — Parallel image editing across ComfyUI instances
Replaces the old 11-node Klein chain with a single fleet node.
Distributes images across multiple ComfyUI instances for speed.

SETUP:
1. Launch multiple ComfyUI instances with Klein models:
   • bigLove_klein1_fp8.safetensors (UNET)
   • qwen3_8b_abliterated_v2-fp8mixed.safetensors (CLIP)
   • flux2-vae.safetensors (VAE)
2. Paste URLs into comfyui_urls field (comma-separated)
   Example: "http://1.2.3.4:18188,http://5.6.7.8:18188"
3. Set workers=4-8 (concurrent edits PER instance)

PROMPT TIPS:
• "solo woman, sharp focus, professional studio photograph"
• "remove man from image, keep woman only"
• "denoise, upscale, masterpiece, 8k resolution"
• "soft studio lighting, clean neutral background"

SETTINGS:
• steps: 3=fast (Klein is distilled), 6-10=higher quality
• cfg: 1.0=minimal guidance (RECOMMENDED for Klein)
• megapixels: 1.0=1MP resolution (1024×1024 equivalent)

SPEED: Single RTX 3090 @ 3 steps = ~2-4 images/min
        4 instances × 4 workers = ~32-64 images/min

BYPASS: Delete comfyui_urls to skip Klein entirely.""",
        COL0, SEC3_Y, NOTE_W, 540)

    klein_fleet = B.node("QVL_KleinFleet", COL1, SEC3_Y, NODE_W + 40, 320,
        title="Klein Fleet (distributed edit)",
        widgets={
            "comfyui_urls": "",
            "prompt": "solo woman, sharp focus, professional studio photograph, soft studio lighting, clean neutral background, high resolution DSLR photograph, pristine image quality, no artifacts, no noise",
            "workers": 4,
            "unet_name": "bigLove_klein1_fp8.safetensors",
            "clip_name": "qwen3_8b_abliterated_v2-fp8mixed.safetensors",
            "vae_name": "flux2-vae.safetensors",
            "steps": 3,
            "cfg": 1.0,
            "width": 1024,
            "height": 1024,
            "megapixels": 1.0,
        },
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
            {"name": "filenames", "type": "STRING", "link": None, "slot_index": 1},
        ],
        outputs=[
            {"name": "images", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "filenames", "type": "STRING", "links": [], "slot_index": 1},
        ])
    B.link(dedup, 0, klein_fleet, 0, "IMAGE")
    B.link(dedup, 2, klein_fleet, 1, "STRING")

    prev_klein = B.node("PreviewImage", COL2, SEC3_Y, PREV_W, 60,
        title="Preview: Klein Output",
        inputs=[{"name": "images", "type": "IMAGE", "link": None}])
    B.link(klein_fleet, 0, prev_klein, 0, "IMAGE")

    # =====================================================
    # SECTION 4: CAPTIONING  (y=2240)
    # =====================================================
    SEC4_Y = 2240
    B.group("4. CAPTIONING [VLM FLEET]", GRP_X, SEC4_Y - 40, GRP_W3F, 650, "#2d4080")

    B.note("""⚡ CAPTIONING — VLM-powered with fleet support

QVL Caption — Generate training captions
Each image is sent to VLM with a caption prompt
tuned for your target model architecture.

MODEL:
• 7b: Fast (~2 sec/img), less detailed
• 32b: Detailed (~10-30 sec/img), rich descriptions
  RECOMMENDATION: Use 32b for final datasets

PRESET: Caption style for your target training model
• 'flux2': Natural language, detailed, for Flux 2 Dev
• 'flux': Similar but tuned for Flux 1.0
• 'sdxl': Comma-separated tags for SDXL models
• 'booru': Danbooru tag format (anime/illustration)
• 'pony': Pony Diffusion specific tag format
• 'natural': Simple natural language
• 'structured': JSON metadata format

⚡ FLEET MODE (10-20x FASTER):
1. Run: python fleet_caption.py launch --count 10 --model 32b
2. Copy the URLs output
3. Paste into ollama_urls field below
4. Set workers = number of instances (e.g., 10)
5. Run workflow — captions distribute across all instances
6. When done: python fleet_caption.py destroy

SINGLE INSTANCE:
• Set workers=1, leave ollama_urls empty
• Uses ollama_url (local Ollama server)

SPEED:
• 100 images @ 32B: 17-50 min single → 2-5 min with 10 instances
• No quality loss — same model, just more instances""",
        COL0, SEC4_Y, NOTE_W, 600)

    caption = B.node("QVL_Caption", COL1, SEC4_Y, NODE_W, 220,
        title="Caption (VLM Fleet)",
        widgets={
            "model": "huihui_ai/qwen2.5-vl-abliterated:32b",
            "ollama_url": "http://localhost:11434",
            "preset": "flux2",
            "api_type": "ollama",
            "workers": 1,
            "ollama_urls": "",
        },
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
            {"name": "filenames", "type": "STRING", "link": None, "slot_index": 1},
        ],
        outputs=[
            {"name": "images", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "captions_json", "type": "STRING", "links": [], "slot_index": 1},
            {"name": "captions_text", "type": "STRING", "links": [], "slot_index": 2},
            {"name": "prompt_used", "type": "STRING", "links": [], "slot_index": 3},
            {"name": "count", "type": "INT", "links": [], "slot_index": 4},
        ])
    B.link(klein_fleet, 0, caption, 0, "IMAGE")
    B.link(klein_fleet, 1, caption, 1, "STRING")

    prev_caption = B.node("PreviewImage", COL2, SEC4_Y, PREV_W, 60,
        title="Preview: Captioned Images",
        inputs=[{"name": "images", "type": "IMAGE", "link": None}])
    B.link(caption, 0, prev_caption, 0, "IMAGE")

    log_captions = B.node("PreviewAny", COL2, SEC4_Y + 120, 350, 120,
        title="Caption Text Output",
        inputs=[{"name": "source", "type": "*", "link": None}])
    B.link(caption, 2, log_captions, 0, "STRING")

    log_prompt = B.node("PreviewAny", COL2, SEC4_Y + 280, 350, 80,
        title="Prompt Used",
        inputs=[{"name": "source", "type": "*", "link": None}])
    B.link(caption, 3, log_prompt, 0, "STRING")

    # =====================================================
    # SECTION 5: RESIZE & SAVE  (y=2940)
    # =====================================================
    SEC5_Y = 2940
    B.group("5. RESIZE & SAVE", GRP_X, SEC5_Y - 40, GRP_W2, 470, "#5030a8")

    B.note("""RESIZE & SAVE — Final dataset output

QVL Resize — Uniform dimensions for training
• target_size: Target pixel dimension
  1024=standard for Flux/SDXL LoRA training
  512=for SD 1.5 or quick low-res training
• mode: How to handle non-square images
  — 'longest_side': Scale so longest edge = target_size
    Preserves aspect ratio. RECOMMENDED.
  — 'resize': Stretch to exact square (DISTORTS)
  — 'pad': Fit inside square, add black padding
  — 'crop_center': Crop from center to exact square

QVL Save Dataset — Save images + caption files
• output_folder: Destination path
  Each image gets image.ext + image.txt (caption)
• format: Image format
  — 'png': Lossless, largest files (~2-5MB each)
  — 'jpg': Lossy, small files (~100-300KB each)
  — 'webp': Best compression, smallest
  For training: png or jpg@95. File size doesn't
  affect training quality — images are loaded as tensors.
• quality: JPEG/WebP compression (50-100)
  95=near-lossless (RECOMMENDED)

OUTPUT STRUCTURE:
  /workspace/dataset/
    image_0000.png
    image_0000.txt  (caption)
    image_0001.png
    image_0001.txt
    ...""",
        COL0, SEC5_Y, NOTE_W, 420)

    resize = B.node("QVL_Resize", COL1, SEC5_Y, NODE_W, 120,
        title="Resize (training dimensions)",
        widgets={"target_size": 1024, "mode": "longest_side"},
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
        ],
        outputs=[
            {"name": "resized", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "count", "type": "INT", "links": [], "slot_index": 1},
        ])
    B.link(caption, 0, resize, 0, "IMAGE")

    prev_resize = B.node("PreviewImage", COL2, SEC5_Y, PREV_W, 60,
        title="Preview: Final Resized",
        inputs=[{"name": "images", "type": "IMAGE", "link": None}])
    B.link(resize, 0, prev_resize, 0, "IMAGE")

    save = B.node("QVL_SaveDataset", COL1, SEC5_Y + 160, NODE_W, 200,
        title="Save Dataset (images + captions)",
        widgets={"output_folder": "/workspace/dataset", "format": "png", "quality": 95},
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
            {"name": "captions_json", "type": "STRING", "link": None, "slot_index": 1},
            {"name": "filenames", "type": "STRING", "link": None, "slot_index": 2},
        ],
        outputs=[
            {"name": "output_path", "type": "STRING", "links": [], "slot_index": 0},
            {"name": "count", "type": "INT", "links": [], "slot_index": 1},
        ])
    B.link(resize, 0, save, 0, "IMAGE")
    B.link(caption, 1, save, 1, "STRING")
    B.link(klein_fleet, 1, save, 2, "STRING")

    save_log = B.node("PreviewAny", COL2, SEC5_Y + 200, PREV_W, 80,
        title="Save Results",
        inputs=[{"name": "source", "type": "*", "link": None}])
    B.link(save, 0, save_log, 0, "STRING")

    # =====================================================
    # SECTION 6: UTILITY NODES  (y=3460)
    # =====================================================
    SEC6_Y = 3460
    B.group("6. UTILITY NODES (wire in as needed)", GRP_X, SEC6_Y - 40, GRP_W3, 530, "#707070")

    B.note("""UTILITY NODES — Extra tools, not connected by default

QVL Detect — VLM bounding box detection
Find and locate subjects in images.
• FLEET-ENABLED: workers + ollama_urls
• Returns bbox as [x1, y1, x2, y2] percentages
• Use with QVL_SmartCrop for intelligent cropping

QVL Analyze — Full metadata extraction (VLM)
• FLEET-ENABLED: workers + ollama_urls
• Extracts pose, angle, lighting, quality, subject count
• Structured JSON output
• Use before MetadataRouter for advanced filtering

QVL Custom Query — Send ANY prompt to VLM
• FLEET-ENABLED: workers + ollama_urls
• Custom analysis, alternative captions, quality checks
• Wire images in, write your prompt, get responses

QVL Metadata Router — Split images by metadata
Use AFTER QVL Analyze. Routes images based on
detected attributes (pose, angle, quality tier).
• route_field: Which metadata field to match
• route_value: Comma-separated values to match

These are disconnected — drag connections to wire them
into your pipeline wherever needed.""",
        COL0, SEC6_Y, NOTE_W, 480)

    detect = B.node("QVL_Detect", COL1, SEC6_Y, NODE_W, 200,
        title="Detect Subject (VLM bbox)",
        widgets={
            "model": "huihui_ai/qwen2.5-vl-abliterated:7b",
            "ollama_url": "http://127.0.0.1:11434",
            "prompt": "Return JSON with bbox [x1,y1,x2,y2] as percentages (0-100) of the main female subject. Include all visible body parts from head to toe.\n{\"bbox\": [x1, y1, x2, y2], \"confidence\": 0.0-1.0}",
            "api_type": "ollama",
            "img_size": 512,
            "workers": 1,
            "ollama_urls": "",
        },
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
        ],
        outputs=[
            {"name": "images", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "metadata_json", "type": "STRING", "links": [], "slot_index": 1},
            {"name": "count", "type": "INT", "links": [], "slot_index": 2},
        ])

    analyze = B.node("QVL_Analyze", COL2, SEC6_Y, NODE_W, 160,
        title="Deep Analyze (VLM metadata)",
        widgets={
            "model": "huihui_ai/qwen2.5-vl-abliterated:7b",
            "ollama_url": "http://127.0.0.1:11434",
            "api_type": "ollama",
            "workers": 1,
            "ollama_urls": "",
        },
        inputs=[{"name": "images", "type": "IMAGE", "link": None, "slot_index": 0}],
        outputs=[
            {"name": "images", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "metadata_json", "type": "STRING", "links": [], "slot_index": 1},
            {"name": "summary", "type": "STRING", "links": [], "slot_index": 2},
            {"name": "count", "type": "INT", "links": [], "slot_index": 3},
        ])

    custom_q = B.node("QVL_CustomQuery", COL3, SEC6_Y, NODE_W, 180,
        title="Custom VLM Query",
        widgets={
            "prompt": "Describe this image in detail.",
            "model": "huihui_ai/qwen2.5-vl-abliterated:7b",
            "ollama_url": "http://127.0.0.1:11434",
            "temperature": 0.3, "max_tokens": 500, "api_type": "ollama",
            "workers": 1,
            "ollama_urls": "",
        },
        inputs=[{"name": "images", "type": "IMAGE", "link": None, "slot_index": 0}],
        outputs=[
            {"name": "images", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "responses", "type": "STRING", "links": [], "slot_index": 1},
            {"name": "count", "type": "INT", "links": [], "slot_index": 2},
        ])

    router = B.node("QVL_MetadataRouter", COL1, SEC6_Y + 240, NODE_W, 140,
        title="Metadata Router (split by attribute)",
        widgets={"route_field": "pose", "route_value": "standing"},
        inputs=[
            {"name": "images", "type": "IMAGE", "link": None, "slot_index": 0},
            {"name": "metadata_json", "type": "STRING", "link": None, "slot_index": 1},
        ],
        outputs=[
            {"name": "matched", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "unmatched", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "matched_count", "type": "INT", "links": [], "slot_index": 2},
            {"name": "unmatched_count", "type": "INT", "links": [], "slot_index": 3},
        ])

    # Build and save
    wf = B.build()
    out_path = "/workspace/ComfyUI/user/default/workflows/full_lora_pipeline_expert.json"
    with open(out_path, "w") as f:
        json.dump(wf, f, indent=2)

    print(f"Generated: {len(B.nodes)} nodes, {len(B.links)} links, {len(B.groups)} groups")
    print(f"Saved to: {out_path}")
    return wf


if __name__ == "__main__":
    generate()

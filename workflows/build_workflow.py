"""Build a clean, working LoRA pipeline workflow.
Verified against actual node specs on instance."""
import json

nodes = []
links = []
nid = 0
lid = 0

def node(type_, title, widgets_values, inputs=None, outputs=None, pos=None, size=None):
    global nid
    nid += 1
    n = {
        "id": nid,
        "type": type_,
        "pos": pos or [0, 0],
        "size": size or [300, 200],
        "flags": {},
        "order": nid,
        "mode": 0,
        "properties": {"Node name for S&R": type_},
        "widgets_values": widgets_values,
    }
    if inputs:
        n["inputs"] = [{"name": i[0], "type": i[1], "link": None} for i in inputs]
    if outputs:
        n["outputs"] = [{"name": o[0], "type": o[1], "links": [], "slot_index": idx} for idx, o in enumerate(outputs)]
    if title:
        n["title"] = title
    nodes.append(n)
    return nid

def link(from_id, from_slot, to_id, to_slot, dtype="*"):
    global lid
    lid += 1
    links.append([lid, from_id, from_slot, to_id, to_slot, dtype])
    for n in nodes:
        if n["id"] == from_id and "outputs" in n:
            if from_slot < len(n["outputs"]):
                n["outputs"][from_slot]["links"].append(lid)
    for n in nodes:
        if n["id"] == to_id and "inputs" in n:
            if to_slot < len(n["inputs"]):
                n["inputs"][to_slot]["link"] = lid
    return lid

# ============================================
# GROUP 1: INPUT
# ============================================

# 1. Video input (DEFAULT - connected to pipeline)
# widgets: video, extraction_mode, fps, scene_threshold, max_frames, dedup_on_extract, dedup_threshold
vid = node("QVL_LoadVideoFrames", "1. VIDEO INPUT (Default)",
    ["** ALL VIDEOS **", "scene_detect", 1.0, 0.1, 500, True, 2],
    outputs=[
        ("frames", "IMAGE"), ("filenames", "STRING"),
        ("kept_count", "INT"), ("total_extracted", "INT"),
    ],
    pos=[50, 100], size=[350, 260])

# 2. Server Image browse (ALTERNATIVE - disconnected, user swaps wire)
# widgets: image_path
img = node("QVL_ServerImage", "1b. BROWSE IMAGE (Swap wire to use)",
    ["/workspace/scored_keep/"],
    outputs=[
        ("images", "IMAGE"), ("filenames", "STRING"),
        ("count", "INT"), ("mask", "MASK"),
    ],
    pos=[50, 450], size=[360, 300])

# ============================================
# GROUP 2: KLEIN EDIT
# ============================================

# 3. UNET Loader - widgets: unet_name, weight_dtype
unet = node("UNETLoader", "Klein Model",
    ["bigLove_klein1_fp8.safetensors", "fp8_e4m3fn"],
    outputs=[("MODEL", "MODEL")],
    pos=[500, -200], size=[320, 100])

# 4. CLIP Loader - widgets: clip_name, type
# Use CLIPLoader (single) with type=flux2 for Klein
clip = node("CLIPLoader", "Klein CLIP",
    ["qwen3_8b_abliterated_v2-fp8mixed.safetensors", "flux2"],
    outputs=[("CLIP", "CLIP")],
    pos=[500, -60], size=[320, 100])

# 5. VAE Loader - widgets: vae_name
vae = node("VAELoader", "Klein VAE",
    ["flux2-vae.safetensors"],
    outputs=[("VAE", "VAE")],
    pos=[500, 80], size=[320, 80])

# 6. Positive prompt - widgets: text
# CLIPTextEncode: inputs=clip, outputs=CONDITIONING
pos_prompt = node("CLIPTextEncode", "Positive Prompt",
    ["solo woman, sharp focus, professional studio photograph, soft studio lighting, clean neutral background, high resolution DSLR photograph, pristine image quality, no artifacts, no noise, exact same face and expression"],
    inputs=[("clip", "CLIP")],
    outputs=[("CONDITIONING", "CONDITIONING")],
    pos=[880, -100], size=[420, 160])

# 7. Negative (zero out) - no widgets
# ConditioningZeroOut: inputs=conditioning, outputs=CONDITIONING
neg = node("ConditioningZeroOut", "Negative (Zero)",
    [],
    inputs=[("conditioning", "CONDITIONING")],
    outputs=[("CONDITIONING", "CONDITIONING")],
    pos=[880, 100], size=[280, 80])

# 8. Scale image - widgets: upscale_method, megapixels, resolution_steps
# ImageScaleToTotalPixels: inputs=image
scale = node("ImageScaleToTotalPixels", "Scale to 1MP",
    ["lanczos", 1.0, 1],
    inputs=[("image", "IMAGE")],
    outputs=[("IMAGE", "IMAGE")],
    pos=[880, 250], size=[300, 100])

# 9. VAE Encode - no widgets
# inputs: pixels(IMAGE), vae(VAE)
encode = node("VAEEncode", "VAE Encode",
    [],
    inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
    outputs=[("LATENT", "LATENT")],
    pos=[1250, 250], size=[250, 80])

# 10. ReferenceLatent - no widgets
# ReferenceLatent: inputs=conditioning(CONDITIONING, required), latent(LATENT, optional)
# Output: CONDITIONING
ref1 = node("ReferenceLatent", "Reference (Source Image)",
    [],
    inputs=[("conditioning", "CONDITIONING"), ("latent", "LATENT")],
    outputs=[("CONDITIONING", "CONDITIONING")],
    pos=[1250, 380], size=[300, 100])

# 11. Empty Flux2 Latent - widgets: width, height, batch_size
empty_lat = node("EmptyFlux2LatentImage", "Empty Latent",
    [1024, 1024, 1],
    outputs=[("LATENT", "LATENT")],
    pos=[880, 420], size=[250, 120])

# 12. Random Noise - widgets: noise_seed
noise = node("RandomNoise", "Noise",
    [42],
    outputs=[("NOISE", "NOISE")],
    pos=[1600, -200], size=[220, 80])

# 13. Sampler Select - widgets: sampler_name
sampler = node("KSamplerSelect", "Sampler (euler)",
    ["euler"],
    outputs=[("SAMPLER", "SAMPLER")],
    pos=[1600, -80], size=[220, 80])

# 14. Flux2 Scheduler - widgets: steps, width, height
# No model input! Just steps + dimensions
sched = node("Flux2Scheduler", "Scheduler (3 steps)",
    [3, 1024, 1024],
    outputs=[("SIGMAS", "SIGMAS")],
    pos=[1600, 40], size=[250, 120])

# 15. CFG Guider - widgets: cfg
# inputs: model(MODEL), positive(CONDITIONING), negative(CONDITIONING)
guider = node("CFGGuider", "CFG Guider (1.0)",
    [1.0],
    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("cfg", "FLOAT")],
    outputs=[("GUIDER", "GUIDER")],
    pos=[1600, 200], size=[260, 130])

# 16. Sampler Custom Advanced - no widgets
# inputs: noise, guider, sampler, sigmas, latent_image
sampler_adv = node("SamplerCustomAdvanced", "Klein Sampler",
    [],
    inputs=[("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
    outputs=[("output", "LATENT"), ("denoised_output", "LATENT")],
    pos=[1950, 50], size=[300, 160])

# 17. VAE Decode - no widgets
decode = node("VAEDecode", "VAE Decode",
    [],
    inputs=[("samples", "LATENT"), ("vae", "VAE")],
    outputs=[("IMAGE", "IMAGE")],
    pos=[1950, 280], size=[250, 80])

# 18. Preview Klein output
prev1 = node("PreviewImage", "Preview: Klein Output",
    [],
    inputs=[("images", "IMAGE")],
    pos=[2300, 50], size=[400, 400])

# ============================================
# GROUP 3: CAPTION + SAVE
# ============================================

# 19. Caption
# widgets: api_url, model, preset, max_resolution, max_tokens, temperature, batch_size
caption = node("QVL_Caption", "3. Caption",
    ["http://localhost:11434", "huihui_ai/qwen2.5-vl-abliterated:32b", "flux2", 512, 150, 0.3, 1],
    inputs=[("images", "IMAGE"), ("filenames", "STRING")],
    outputs=[("images", "IMAGE"), ("captions", "STRING"), ("filenames", "STRING")],
    pos=[2300, 520], size=[400, 220])

# 20. Resize
# widgets: target_size, mode, keep_aspect
resize = node("QVL_Resize", "4. Resize 1024",
    [1024, "cover", True],
    inputs=[("images", "IMAGE")],
    outputs=[("images", "IMAGE"), ("filenames", "STRING")],
    pos=[2750, 520], size=[300, 160])

# 21. Save Dataset
# widgets: output_dir, format, save_captions, overwrite
save = node("QVL_SaveDataset", "5. Save Dataset",
    ["/workspace/dataset", "kohya", True, True],
    inputs=[("images", "IMAGE"), ("captions", "STRING"), ("filenames", "STRING")],
    pos=[3100, 520], size=[350, 200])

# 22. Preview final
prev2 = node("PreviewImage", "Preview: Final",
    [],
    inputs=[("images", "IMAGE")],
    pos=[2750, 50], size=[400, 400])

# ============================================
# CONNECTIONS
# ============================================

# --- Input to Klein ---
# Video frames → Scale
link(vid, 0, scale, 0, "IMAGE")

# --- Klein Model/CLIP/VAE ---
link(clip, 0, pos_prompt, 0, "CLIP")       # CLIP → positive prompt
link(pos_prompt, 0, neg, 0, "CONDITIONING") # positive → zero out for negative
link(pos_prompt, 0, ref1, 0, "CONDITIONING") # positive conditioning → ReferenceLatent

# --- Image → Encode → Reference ---
link(scale, 0, encode, 0, "IMAGE")          # scaled image → VAE encode
link(vae, 0, encode, 1, "VAE")              # VAE → encode
link(encode, 0, ref1, 1, "LATENT")          # encoded image → ReferenceLatent (latent input)

# --- Guider setup ---
link(unet, 0, guider, 0, "MODEL")           # model → guider
link(ref1, 0, guider, 1, "CONDITIONING")     # reference conditioning → guider positive
link(neg, 0, guider, 2, "CONDITIONING")      # zero conditioning → guider negative

# --- Sampler ---
link(noise, 0, sampler_adv, 0, "NOISE")
link(guider, 0, sampler_adv, 1, "GUIDER")
link(sampler, 0, sampler_adv, 2, "SAMPLER")
link(sched, 0, sampler_adv, 3, "SIGMAS")
link(empty_lat, 0, sampler_adv, 4, "LATENT") # empty latent → sampler (generation starts from noise)

# --- Decode ---
link(sampler_adv, 0, decode, 0, "LATENT")
link(vae, 0, decode, 1, "VAE")

# --- Preview Klein ---
link(decode, 0, prev1, 0, "IMAGE")

# --- Caption + Save pipeline ---
link(decode, 0, caption, 0, "IMAGE")         # Klein output → caption
link(vid, 1, caption, 1, "STRING")            # filenames from video input

link(caption, 0, resize, 0, "IMAGE")          # captioned → resize
link(resize, 0, save, 0, "IMAGE")             # resized → save
link(caption, 1, save, 1, "STRING")           # captions → save
link(caption, 2, save, 2, "STRING")           # filenames → save

# --- Preview final ---
link(resize, 0, prev2, 0, "IMAGE")

# ============================================
# BUILD
# ============================================
workflow = {
    "last_node_id": nid,
    "last_link_id": lid,
    "nodes": nodes,
    "links": links,
    "groups": [
        {"title": "1. INPUT", "bounding": [20, 40, 420, 750], "color": "#3f789e", "font_size": 24, "flags": {}},
        {"title": "2. KLEIN EDIT (remove man + enhance)", "bounding": [470, -260, 1870, 800], "color": "#8A2BE2", "font_size": 24, "flags": {}},
        {"title": "3. CAPTION + SAVE", "bounding": [2270, 470, 1200, 320], "color": "#2E8B57", "font_size": 24, "flags": {}},
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.6, "offset": [-100, 300]}},
    "version": 0.4,
}

with open("/workspace/ComfyUI/user/default/workflows/lora_pipeline_v2.json", "w") as f:
    json.dump(workflow, f, indent=2)
print(f"Workflow saved: {nid} nodes, {lid} links")
print("File: /workspace/ComfyUI/user/default/workflows/lora_pipeline_v2.json")

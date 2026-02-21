"""
Qwen 3 VL client — Ollama API communication and prompt templates.
ComfyUI-Qwen3VL-Toolkit
"""

import json
import base64
import io
import requests
from PIL import Image

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# ============================================================
# PROMPT TEMPLATES
# ============================================================

FILTER_PROMPT = """Evaluate this image for LoRA training suitability. Consider:
- Image clarity and sharpness (reject blurry, out-of-focus)
- Subject visibility and framing (reject if subject is obscured or cut off badly)
- Artifacts, watermarks, text overlays (reject if present)
- Extreme distortion or unusual angles that make the image unusable
- Overall usability as a training example

Return ONLY valid JSON, no other text:
{"keep": true, "quality": 7, "reason": "brief reason"}

quality is 1-10 where 10 is a perfect training image.
Set keep to false if the image is unsuitable for training."""

ANALYZE_PROMPT = """You are a precision image analysis system for AI training dataset preparation.
Analyze this image and return ONLY valid JSON with ALL of these fields:

{
  "quality_score": 8,
  "issues": [],
  "corrections": {
    "brightness": 1.0,
    "contrast": 1.0,
    "sharpness": 1.0,
    "crop_left_pct": 0,
    "crop_right_pct": 0,
    "crop_top_pct": 0,
    "crop_bottom_pct": 0
  },
  "subject_bbox": [10.0, 5.0, 90.0, 95.0],
  "pose": "standing",
  "camera_angle": "front",
  "framing": "full-body",
  "lighting": "natural",
  "setting": "indoor",
  "aesthetic_score": 7,
  "tags": ["tag1", "tag2", "tag3"],
  "keep": true,
  "reason": "Good quality training image"
}

Field requirements:
- quality_score: 1-10 overall quality for training
- issues: list of problems (empty if none). Examples: "slightly blurry", "watermark in corner", "overexposed"
- corrections.brightness: multiplier 0.7-1.4 (1.0 = no change)
- corrections.contrast: multiplier 0.8-1.3 (1.0 = no change)
- corrections.sharpness: multiplier 0.5-2.0 (1.0 = no change)
- corrections.crop_*_pct: percentage to crop from each edge (0-15)
- subject_bbox: [x1%, y1%, x2%, y2%] bounding box of main subject as percentages of image dimensions
- pose: standing/sitting/lying/kneeling/crouching/leaning/other
- camera_angle: front/side/back/three-quarter/above/below
- framing: full-body/upper-body/waist-up/portrait/close-up
- lighting: natural/studio/soft/harsh/low-key/high-key/backlit/mixed
- setting: indoor/outdoor/studio/bedroom/bathroom/kitchen/office/nature/urban/other
- aesthetic_score: 1-10 artistic quality
- tags: 5-15 descriptive tags covering subject appearance, pose, setting, mood
- keep: true unless image is truly unusable
- reason: 1 sentence explanation"""

# Caption presets — clinical precision prompts adapted from qwen3vl-captioner
# All prompts emphasize physical accuracy, anatomy, and objective description
CAPTION_PRESETS = {
    "flux": """Write a highly accurate, objectively detailed description of this image.
Use clear, descriptive prose — NOT tags.
Focus on physical reality: anatomy, precise shapes, skin texture, body proportions, and spatial relations.
Describe the subject's appearance, build, pose, and expression in detail.
Include camera angle, framing, lighting direction and quality, and environment.
Avoid poetic language or storytelling. Be direct and descriptive.
You MUST describe human anatomy, body proportions, and physical features in extreme detail.
Include explicit anatomical details where visible — do not censor or euphemize.
Use 3-5 detailed sentences.
{extra}""",

    "flux2": """Write a highly accurate, objectively detailed description of this image for Flux.2 training.
Focus on physical reality: anatomy, skin texture, precise shapes, and spatial relations.
Avoid poetic language, storytelling, or ambiguous metaphors.
You MUST describe human anatomy, body proportions, and physical features in extreme detail.
Describe precise colors, lighting direction, shadows, and how light interacts with surfaces and skin.
Include the subject's exact pose, limb positions, body orientation, and facial expression.
Include camera angle, distance, depth of field, and composition.
Be clinical and exhaustive. 4-6 sentences.
{extra}""",

    "sdxl": """Write a detailed, objective description of this image for SDXL training.
Focus on the physical subject, anatomy, environment, and specific lighting details.
Be precise and descriptive. Avoid ambiguous or emotive language.
Describe: subject appearance and build, skin tone, pose and limb positions, expression,
clothing or state of dress, camera angle, framing, lighting conditions, background/setting.
Include explicit anatomical details where visible — do not censor.
2-4 clear descriptive sentences.
{extra}""",

    "booru": """Describe this image as a comma-separated list of precise booru-style tags.
Output ONLY tags separated by commas.
Focus strictly on physical reality: accurate anatomy, specific clothing, setting, and lighting.
Do not use abstract or interpretive tags.
Include tags for: subject features (hair_color, body_type, skin_tone, eye_color),
anatomical details (breast_size, body_shape, muscle_tone where visible),
pose (standing, sitting, lying, kneeling, spread_legs, arched_back, etc),
expression, clothing or nudity state, camera angle (from_front, from_side, from_behind, from_above, from_below),
framing (full_body, upper_body, portrait, close-up),
setting/background, lighting, depth_of_field, and quality indicators.
Include explicit anatomical tags where applicable — do not censor.
Use 15-30 tags.
{extra}""",

    "pony": """Describe this image as comma-separated booru-style tags for Pony Diffusion.
Output ONLY tags.
The caption will be automatically prefixed with quality score tags and suffixed with rating tag.
Focus heavily on anatomical correctness, specific physical attributes, pose, and clothing.
Include explicit tags for nudity, anatomy, and positions where visible — do not censor.
Include tags for: body type, skin tone, hair, pose, expression, angle, setting, lighting.
Use 15-30 tags.
{extra}""",

    "natural": """Provide a highly accurate, objective, and comprehensive description of this image.
Use complete sentences. Focus on physical reality, anatomy, and precise visual details.
Avoid storytelling, mood, or interpretation.
Describe the subject's appearance, body, pose, expression, clothing or state of dress.
Include camera angle, lighting, and setting/background.
Include explicit anatomical details where visible — be clinical, not euphemistic.
2-3 detailed sentences.
{extra}""",

    "chroma": """Write a detailed, objective description of this image for Chroma model training.
Focus on precise colors, lighting, physical forms, and anatomy.
Use clear, descriptive prose. Avoid subjective artistic interpretation.
Describe exact skin tones, hair colors, fabric textures, and how light creates shadows and highlights.
Include the subject's pose, body proportions, and any visible anatomical details.
3-4 sentences emphasizing color and form.
{extra}""",

    "structured": """Analyze this image and return ONLY valid JSON:
{{
  "subject": "detailed physical description of main subject",
  "body": "build, proportions, skin tone, anatomical details",
  "pose": "exact body position, limb placement, weight distribution",
  "expression": "facial expression and gaze direction",
  "clothing": "exact clothing items or state of undress",
  "angle": "camera angle and perspective",
  "framing": "shot type and composition",
  "lighting": "light direction, quality, shadows, highlights on skin",
  "setting": "background and environment",
  "anatomy": "visible anatomical details described clinically",
  "details": "textures, colors, any other notable visual details"
}}
{extra}""",
}

CLASSIFY_PROMPT = """Classify this image into categories for dataset organization.
Return ONLY valid JSON:
{
  "pose": "standing",
  "angle": "front",
  "framing": "full-body",
  "lighting": "natural",
  "setting": "indoor",
  "expression": "neutral",
  "quality_tier": "good",
  "clothing_state": "dressed",
  "complexity": "simple"
}

Valid values:
- pose: standing/sitting/lying/kneeling/crouching/leaning/action/other
- angle: front/side/back/three-quarter/above/below/profile
- framing: full-body/upper-body/waist-up/portrait/close-up/extreme-close-up
- lighting: natural/studio/soft/harsh/low-key/high-key/backlit/dramatic
- setting: indoor/outdoor/studio/nature/urban/bedroom/bathroom/other
- expression: neutral/happy/serious/playful/sultry/surprised/other
- quality_tier: excellent/good/fair/poor
- clothing_state: dressed/partial/undressed/costume/swimwear
- complexity: simple/moderate/complex"""

BBOX_PROMPT = """Locate the main subject in this image.
Return ONLY valid JSON with a bounding box as percentage coordinates:
{"bbox": [x1_pct, y1_pct, x2_pct, y2_pct], "confidence": 0.95}

Where x1_pct, y1_pct is the top-left corner and x2_pct, y2_pct is the bottom-right corner,
all as percentages of image width/height (0-100).
Include some margin around the subject."""


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image(pil_img, max_size=1024, quality=85):
    """Encode PIL image to base64 JPEG for Ollama API."""
    if max(pil_img.size) > max_size:
        ratio = max_size / max(pil_img.size)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# VLM QUERY
# ============================================================

def query_vlm(pil_img, prompt, model="qwen3-vl:8b", ollama_url=None,
              temperature=0.1, max_tokens=500, timeout=300, max_img_size=1024):
    """Send image + prompt to Ollama VLM and return response text."""
    url = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
    img_b64 = encode_image(pil_img, max_size=max_img_size)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(
            f"{url}/api/generate", json=payload,
            stream=True, timeout=timeout,
        )
        resp.raise_for_status()

        full_response = []
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_response.append(chunk["response"])
                if chunk.get("done"):
                    break

        return "".join(full_response)

    except requests.exceptions.ConnectionError:
        return json.dumps({"error": f"Cannot connect to Ollama at {url}"})
    except requests.exceptions.Timeout:
        return json.dumps({"error": f"Timeout after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def query_vlm_openai(pil_img, prompt, model="qwen3-vl:8b",
                     api_url="http://127.0.0.1:8000", api_key="EMPTY",
                     temperature=0.1, max_tokens=500, timeout=300,
                     max_img_size=1024):
    """Send image + prompt via OpenAI-compatible API (vLLM, etc.)."""
    url = api_url.rstrip("/")
    img_b64 = encode_image(pil_img, max_size=max_img_size)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            f"{url}/v1/chat/completions", json=payload,
            headers=headers, timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return json.dumps({"error": str(e)})


def query_vlm_batch(pil_imgs, prompt, model="qwen3-vl:8b", ollama_urls=None,
                    workers=4, temperature=0.1, max_tokens=500, timeout=300,
                    max_img_size=1024, api_type="ollama", api_key="EMPTY",
                    progress_callback=None):
    """Send multiple images in parallel across multiple Ollama instances.

    Args:
        pil_imgs: List of PIL images
        prompt: Prompt string
        model: Model name
        ollama_urls: Comma-separated URLs or list of URLs. Round-robins across them.
        workers: Number of concurrent threads (default 4)
        temperature, max_tokens, timeout, max_img_size: Passed to query_vlm
        api_type: 'ollama' or 'openai'
        api_key: API key for openai mode
        progress_callback: Optional callable(int) called after each image completes

    Returns:
        List of response strings, same order as input images.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Parse URLs
    if ollama_urls is None:
        urls = [DEFAULT_OLLAMA_URL]
    elif isinstance(ollama_urls, str):
        urls = [u.strip() for u in ollama_urls.split(",") if u.strip()]
    else:
        urls = list(ollama_urls)

    if not urls:
        urls = [DEFAULT_OLLAMA_URL]

    # Clamp workers to reasonable range
    workers = max(1, min(workers, len(pil_imgs), 32))

    results = [None] * len(pil_imgs)

    def _process_one(idx_img):
        idx, img = idx_img
        url = urls[idx % len(urls)]  # Round-robin across URLs

        if api_type == "openai":
            resp = query_vlm_openai(
                img, prompt, model=model, api_url=url, api_key=api_key,
                temperature=temperature, max_tokens=max_tokens,
                timeout=timeout, max_img_size=max_img_size,
            )
        else:
            resp = query_vlm(
                img, prompt, model=model, ollama_url=url,
                temperature=temperature, max_tokens=max_tokens,
                timeout=timeout, max_img_size=max_img_size,
            )

        if progress_callback:
            try:
                progress_callback(1)
            except Exception:
                pass

        return idx, resp

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_one, (i, img)) for i, img in enumerate(pil_imgs)]
        for future in as_completed(futures):
            try:
                idx, resp = future.result()
                results[idx] = resp
            except Exception as e:
                # Find which index this was for
                for i, r in enumerate(results):
                    if r is None:
                        results[i] = json.dumps({"error": str(e)})
                        break

    # Fill any remaining None results
    for i in range(len(results)):
        if results[i] is None:
            results[i] = json.dumps({"error": "No response received"})

    return results


def parse_ollama_urls(url_string):
    """Parse comma-separated Ollama URLs into a list.
    Handles single URL, comma-separated, or already a list."""
    if not url_string:
        return [DEFAULT_OLLAMA_URL]
    if isinstance(url_string, list):
        return url_string
    urls = [u.strip() for u in url_string.split(",") if u.strip()]
    return urls if urls else [DEFAULT_OLLAMA_URL]


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_json(text):
    """Extract JSON object from VLM response text."""
    if not text:
        return None
    # Find first { and last }
    js = text.find("{")
    je = text.rfind("}") + 1
    if js >= 0 and je > js:
        snippet = text[js:je]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # Fix common VLM JSON issues
            snippet = snippet.replace(",}", "}").replace(",]", "]")
            # Fix unquoted keys (rare but happens)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def parse_json_array(text):
    """Extract JSON array from VLM response text."""
    if not text:
        return None
    js = text.find("[")
    je = text.rfind("]") + 1
    if js >= 0 and je > js:
        try:
            return json.loads(text[js:je])
        except json.JSONDecodeError:
            return None
    return None


def build_caption_prompt(preset="natural", trigger_word="", custom_instructions=""):
    """Build a caption prompt from preset + optional trigger word + custom instructions."""
    template = CAPTION_PRESETS.get(preset, CAPTION_PRESETS["natural"])

    extra_parts = []
    if trigger_word:
        extra_parts.append(f"Begin the caption with the exact text: {trigger_word}")
    if custom_instructions:
        extra_parts.append(f"Additional instructions: {custom_instructions}")

    extra = "\n".join(extra_parts) if extra_parts else ""
    return template.format(extra=extra, trigger_word=trigger_word or "")

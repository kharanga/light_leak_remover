import argparse, base64, json, os, sys
import numpy as np, cv2
from . import core

try:
    import anthropic
except ImportError:
    anthropic = None

DEFAULT_MODEL = "claude-opus-5-5"
CLASSES = ("leak", "cool_tint", "dull", "hsv")
MAX_TOKENS = 16000
BETAS = ["server-side-fallback-2026-07-01"]
FALLBACKS = "default"
PREVIEW_JPEG_QUALITY = 90
MIN_BOX_PX = 16

STEP_DEFINITIONS = """You are helping a film-preservation tool remove light leaks from colour-negative film scans. The user will draw boxes and apply corrections; you only PROPOSE boxes. Nothing you return is applied automatically.

The pipeline has four box-driven steps. Propose boxes per class:

leak: a light leak is an additive, smooth, one-coloured warm fog (orange/red) because the orange film base absorbs blue. On the scan it is a soft band or wedge, usually running from a frame edge, brightest at one edge and fading. Draw one box per leak band, covering the whole visible fog with a small margin (about 5 percent of its width) so that the tiles just outside the box are clean. The tool subtracts the cast measured against those neighbours.

cool_tint: a cyan or blue patch left after an earlier over-correction, visible as a cool smear on a surface that should be neutral. Rarely present on a raw scan. Only propose it when you see a distinctly cyan patch with soft edges.

dull: a patch inside or next to a leak band that is darker than the same surface beside it (the subtraction removed light as well as colour). The box covers the dull patch only; the reference surface is derived at runtime from the same rows beside the patch, so do not propose a reference box.

hsv: a thin warm residue that remains inside or along a corrected band, bright and saturated (hue 0-25 degrees, saturation at least 40 of 255, value at least 180 of 255). Propose only where such a residue is visible.

Warnings (from the project's known limits):
- Warm scene objects are NOT leaks: a red car, a yellow wall, an orange scarf, a sunset sky, lit skin, wood, brick. A leak has no texture of its own, follows no object edge, and fades smoothly. If a warm region has sharp edges or texture, it is scene content; do not box it.
- Keep boxes off skin and faces unless the fog clearly runs across them; the correction can flatten sunlit skin.
- A clipped core (pure white or pure red with no detail) inside a leak still belongs in the leak box, but note it in the box note.
- Boxes are axis-aligned rectangles in pixels of the image you are shown: x0 < x1 from the left, y0 < y1 from the top. A box may touch the frame edge.
- If you see no leak, return an empty list rather than guessing.

Give each box a confidence from 0 to 1 and a short note saying what you saw (colour, where it fades, any scene object you deliberately excluded)."""

BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cls": {"type": "string", "enum": list(CLASSES)},
                    "x0": {"type": "integer"}, "x1": {"type": "integer"},
                    "y0": {"type": "integer"}, "y1": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["cls", "x0", "x1", "y0", "y1", "confidence", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["boxes"],
    "additionalProperties": False,
}


def sdk_available():
    return anthropic is not None


def credentials_present():
    env_keys = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
    profile_dir = os.path.join(os.path.expanduser("~"), ".config", "anthropic")
    return any(os.environ.get(k) for k in env_keys) or os.path.isdir(profile_dir)


def available():
    return sdk_available() and credentials_present()


def preview_jpeg_base64(rgb):
    from .app import _preview
    preview_u8, scale = _preview(rgb)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(preview_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), scale, preview_u8.shape[:2]


def user_prompt(classes, preview_shape):
    H, W = preview_shape
    return (f"This image is {W} by {H} pixels. Propose boxes for these classes only: {', '.join(classes)}. "
            "Return every box in pixel coordinates of this image. Return an empty list if nothing qualifies.")


def request_body(image_base64, classes, preview_shape, model=DEFAULT_MODEL):
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "betas": BETAS,
        "fallbacks": FALLBACKS,
        "system": STEP_DEFINITIONS,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
            {"type": "text", "text": user_prompt(classes, preview_shape)},
        ]}],
        "output_config": {"format": {"type": "json_schema", "schema": BOX_SCHEMA}},
    }


def scale_box_to_full(box, scale, shape):
    H, W = shape[:2]
    x0, x1 = sorted((int(round(box["x0"] / scale)), int(round(box["x1"] / scale))))
    y0, y1 = sorted((int(round(box["y0"] / scale)), int(round(box["y1"] / scale))))
    return (max(0, min(W, x0)), max(0, min(W, x1)), max(0, min(H, y0)), max(0, min(H, y1)))


def parse_proposals(text, scale, shape, classes):
    proposals = []
    for raw in json.loads(text)["boxes"]:
        if raw["cls"] not in classes: continue
        x0, x1, y0, y1 = scale_box_to_full(raw, scale, shape)
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX: continue
        proposals.append({"cls": raw["cls"], "box": (x0, x1, y0, y1),
                          "confidence": float(min(1.0, max(0.0, raw["confidence"]))), "note": str(raw["note"])})
    return proposals


def response_text(response):
    return next((block.text for block in response.content if block.type == "text"), "")


def api_error_types():
    return (anthropic.APIError,) if anthropic is not None else ()


def propose_boxes(rgb, model=DEFAULT_MODEL, classes=CLASSES, client=None):
    if client is None and not sdk_available(): return [], "the anthropic package is not installed (pip install -e '.[propose]')"
    image_base64, scale, preview_shape = preview_jpeg_base64(rgb)
    body = request_body(image_base64, classes, preview_shape, model)
    try:
        response = (client or anthropic.Anthropic()).beta.messages.create(**body)
    except api_error_types() as e:
        return [], f"API error: {e}"
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        return [], f"the model declined to propose boxes ({category or 'refusal'})"
    if response.stop_reason == "max_tokens":
        return [], "the response was cut off before the box list was complete"
    try:
        return parse_proposals(response_text(response), scale, rgb.shape, classes), ""
    except (ValueError, KeyError, TypeError) as e:
        return [], f"could not read the box list: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--classes", default=",".join(CLASSES))
    a = p.parse_args()
    try:
        rgb = core.load(a.image)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
    proposals, reason = propose_boxes(rgb, model=a.model, classes=tuple(c for c in a.classes.split(",") if c))
    print(json.dumps({"image": os.path.basename(a.image), "model": a.model, "boxes": [dict(p, box=list(p["box"])) for p in proposals], "reason": reason}, indent=2))
    if reason: sys.exit(1)


if __name__ == "__main__":
    main()

import base64, json, os, tempfile, time, numpy as np, cv2, gradio as gr
from pathlib import Path
from . import core, propose, inpaint, detect, recipe

PREVIEW_MAX = 1600
STEP_INFO = {
    "1 · Remove leak (warm)":   "Draw one box per light leak. The tile method subtracts the orange cast measured against the tiles just outside the box.",
    "2 · Remove blue tint (cool)": "Draw a box around any cyan/blue patch left after step 1. Same method with channels swapped, gentler.",
    "3 · Match exposure":       "Draw one box per dull patch, press 'Next drag = reference', then drag a REFERENCE box of the same surface that looks right (same rows). All patches are lifted to the reference's brightness in linear light.",
    "4 · Neutralize warm (HSV)": "Final pass, automatic: masks bright warm pixels (hue 0–25, sat ≥ 40, value ≥ 180) inside the leak boxes from steps 1–3 (padded by one tile), softens the mask and pulls red toward green by the strength. Tick 'whole frame' to run everywhere; draw boxes to override.",
    "5 · Inpaint residue (opt-in)": "Only for clipped cores (red ≥ 0.97 while blue is not, in the ORIGINAL scan). Draw a box around the pale bar; LaMa when torch and big-lama.pt are present, else OpenCV Telea; colour matched to a 61 px ring, the scan's own grain blended back, seam feathered. Refuses when the mask is over 15 % of the frame. This step invents pixels: use it last and sparingly.",
    "R · Recipe v2 (main path)": "The main path: draw one box per light leak and press Apply. Runs the whole recipe at fixed gains (no strength control): the boxes are padded by two tiles, two deleak passes (1.3, 1.3), blue tint only where it is cyan against the same row, exposure matched to a side strip whose rows follow the box edge, HSV 0.45, a black-point floor, then a per-tile fade so only leak tiles change. Logged as one recipe_v2 stage that `cli --replay` reproduces. Steps 1–5 stay available as the manual path.",
}
RECIPE_MODE = "R · Recipe v2 (main path)"
UNDO_LOG_EDITED_NOTE = "log not changed (boxes.json was edited elsewhere)"
UNDO_LOG_UNCHANGED_PREFIX = "log not changed ("
INPAINT_METHODS = ("auto", "telea", "lama")
INPAINT_MASK_MODES = ("clip", "score", "box")
TILE_CHOICES = ("auto", "64", "32", "16")
PROPOSAL_COLOURS = {"leak": (255, 80, 0), "cool_tint": (0, 200, 255), "dull": (230, 230, 0), "hsv": (255, 0, 200)}
PROPOSAL_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
PROPOSAL_STEP = {"leak": "1", "cool_tint": "2", "dull": "3", "hsv": "4"}

def _tile_from_choice(choice, shape):
    return core.resolve_tile(None if choice == "auto" else int(choice), shape)

STAGE_HTML = (
    '<div class="leak-stage">'
    '<img class="leak-img" src="${value}" draggable="false" alt="">'
    '<canvas class="leak-band"></canvas>'
    '</div>'
)
STAGE_CSS = """
.leak-stage { position: relative; display: inline-block; max-width: 100%; line-height: 0; user-select: none; touch-action: none; }
.leak-img { display: block; max-width: 100%; max-height: 640px; cursor: crosshair; }
.leak-img[src=""] { display: none; }
.leak-band { position: absolute; left: 0; top: 0; pointer-events: none; }
"""
STAGE_JS = """
let dragStart = null;
const img = () => element.querySelector('.leak-img');
const band = () => element.querySelector('.leak-band');

function fitBand() {
  const i = img(), c = band();
  if (!i || !c) return;
  c.width = i.clientWidth; c.height = i.clientHeight;
}
function imagePoint(e) {
  const r = img().getBoundingClientRect();
  return [Math.min(Math.max(e.clientX - r.left, 0), r.width), Math.min(Math.max(e.clientY - r.top, 0), r.height)];
}
function drawBand(a, b) {
  fitBand();
  const c = band(); if (!c) return;
  const g = c.getContext('2d'); g.clearRect(0, 0, c.width, c.height);
  if (!a || !b) return;
  const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]), w = Math.abs(a[0] - b[0]), h = Math.abs(a[1] - b[1]);
  g.fillStyle = 'rgba(255,255,255,0.18)'; g.fillRect(x, y, w, h);
  g.lineWidth = 2; g.strokeStyle = '#ffffff'; g.setLineDash([6, 4]); g.strokeRect(x + 1, y + 1, Math.max(w - 2, 0), Math.max(h - 2, 0));
}
function previewBox(a, b) {
  const i = img();
  const sx = i.naturalWidth / i.clientWidth, sy = i.naturalHeight / i.clientHeight;
  return [Math.round(Math.min(a[0], b[0]) * sx), Math.round(Math.max(a[0], b[0]) * sx),
          Math.round(Math.min(a[1], b[1]) * sy), Math.round(Math.max(a[1], b[1]) * sy)];
}
element.addEventListener('pointerdown', e => {
  if (e.button !== 0 || !e.target.classList.contains('leak-img')) return;
  e.preventDefault();
  dragStart = imagePoint(e);
  drawBand(dragStart, dragStart);
});
window.addEventListener('pointermove', e => { if (dragStart) drawBand(dragStart, imagePoint(e)); });
window.addEventListener('pointerup', e => {
  if (!dragStart) return;
  const box = previewBox(dragStart, imagePoint(e));
  dragStart = null; drawBand(null, null);
  trigger('select', { index: box, value: null });
});
window.addEventListener('pointercancel', () => { dragStart = null; drawBand(null, null); });
"""


def _preview(rgb):
    H, W = rgb.shape[:2]; s = min(1.0, PREVIEW_MAX / max(H, W))
    im = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    return (np.clip(im, 0, 1) * 255).astype(np.uint8), s

def _draw_boxes(prev_u8, boxes, scale, colours):
    im = prev_u8.copy()
    for (x0, x1, y0, y1), col in zip(boxes, colours):
        cv2.rectangle(im, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale) - 1, int(y1 * scale) - 1), col, 3)
    return im

def _dashed_rectangle(im, x0, y0, x1, y1, colour, dash=12, thickness=2):
    def along(a, b, fixed, horizontal):
        for start in range(a, b, dash * 2):
            end = min(start + dash, b)
            p0, p1 = ((start, fixed), (end, fixed)) if horizontal else ((fixed, start), (fixed, end))
            cv2.line(im, p0, p1, colour, thickness)
    along(x0, x1, y0, True); along(x0, x1, y1, True); along(y0, y1, x0, False); along(y0, y1, x1, False)

def _draw_proposals(prev_u8, proposals, scale):
    im = prev_u8.copy()
    for i, p in enumerate(proposals):
        x0, x1, y0, y1 = p["box"]; colour = PROPOSAL_COLOURS.get(p["cls"], (255, 255, 255))
        _dashed_rectangle(im, int(x0 * scale), int(y0 * scale), int(x1 * scale) - 1, int(y1 * scale) - 1, colour)
        cv2.putText(im, f"P{i} {p['cls']} {p['confidence']:.2f}", (int(x0 * scale) + 6, int(y0 * scale) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return im

def _data_url(rgb_u8):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

def _new_state(path):
    rgb = core.load(path); prev, s = _preview(rgb)
    return {"path": path, "orig": rgb, "cur": rgb.copy(), "history": [], "scale": s, "prev": prev,
            "boxes": [], "box_sources": [], "ref": None, "next_is_ref": False, "applied": [], "proposals": [],
            "log": str(Path(path).with_suffix(".boxes.json"))}

def _pad_box(box, pad, W, H):
    x0, x1, y0, y1 = box
    return (max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad), min(H, y1 + pad))


def load_image(file, state):
    if file is None: return state, "", "Load a scan to start.", ""
    state = _new_state(file)
    return state, _render(state, "1"), f"{state['orig'].shape[1]}×{state['orig'].shape[0]} loaded. Press and drag on the image to draw a box.", ""

def on_select(state, mode, evt: gr.SelectData):
    if state is None: return state, "", "Load a scan first."
    s = state["scale"]
    box = tuple(int(v / s) for v in evt.index)
    if box[1] - box[0] < 16 or box[3] - box[2] < 16: return state, _render(state, mode), "Box too small, drag a larger one." + _next_drag_hint(state, mode)
    if mode.startswith("3") and state["next_is_ref"]:
        state["ref"] = box; state["next_is_ref"] = False; msg = f"Reference box {box}. Press Apply."
    else:
        _add_box(state, box, "human"); msg = f"Box {len(state['boxes'])-1}: {box}. Add more or press Apply."
    return state, _render(state, mode), msg + _next_drag_hint(state, mode)

def _add_box(state, box, source):
    state["boxes"].append(tuple(box)); state["box_sources"].append(source)

def _proposal_choices(state):
    return [f"P{i} {p['cls']} {p['confidence']:.2f} {p['box']}" for i, p in enumerate(state["proposals"])] if state else []

def _proposal_index(choice):
    try: return int(str(choice).split()[0][1:])
    except (ValueError, IndexError, AttributeError): return None

def propose_click(state, mode, model):
    if state is None: return state, "", "Load a scan first.", gr.update(choices=[], value=None)
    proposals, reason = propose.propose_boxes(state["cur"], model=model or propose.DEFAULT_MODEL)
    for p in proposals: p["model"] = model or propose.DEFAULT_MODEL
    state["proposals"] = proposals
    if not proposals:
        msg = f"No proposals ({reason})." if reason else "No proposals: the model found nothing to box."
    else:
        msg = f"{len(proposals)} proposal(s) from {model or propose.DEFAULT_MODEL}, dashed on the image. Pick one and Accept / Adjust / Dismiss; nothing is applied until you press Apply. " + "; ".join(f"P{i} {p['cls']} {p['confidence']:.2f}: {p['note']}" for i, p in enumerate(proposals))
    choices = _proposal_choices(state)
    return state, _render(state, mode), msg, gr.update(choices=choices, value=choices[0] if choices else None)

def yolo_button_label():
    return "Propose (YOLO)" if detect.available() else f"Propose (YOLO): {detect.INSTALL_HINT}"

def yolo_proposals(detected, weights):
    return [{"cls": "leak", "box": tuple(box), "confidence": confidence, "note": "yolo detector", "model": os.path.basename(weights), "source": "yolo"} for box, confidence in detected]

def yolo_propose_click(state, mode, weights, conf):
    if state is None: return state, "", "Load a scan first.", gr.update(choices=[], value=None)
    if not detect.available(): return state, _render(state, mode), f"YOLO proposals need ultralytics: {detect.INSTALL_HINT}.", gr.update(choices=_proposal_choices(state))
    weights = weights or detect.DEFAULT_WEIGHTS
    try:
        detected = detect.propose_rgb(detect.load_detector(weights), state["cur"], conf=float(conf))
    except (FileNotFoundError, RuntimeError) as e:
        return state, _render(state, mode), f"No YOLO proposals: {e}", gr.update(choices=_proposal_choices(state))
    state["proposals"] = yolo_proposals(detected, weights)
    msg = (f"{len(detected)} YOLO proposal(s) from {os.path.basename(weights)}, dashed on the image. Pick one and Accept / Adjust / Dismiss; nothing is applied until you press Apply."
           if detected else f"No YOLO proposals above confidence {float(conf):.2f}.")
    choices = _proposal_choices(state)
    return state, _render(state, mode), msg, gr.update(choices=choices, value=choices[0] if choices else None)

def _take_proposal(state, choice):
    i = _proposal_index(choice)
    if i is None or not state["proposals"] or i >= len(state["proposals"]): return None
    return state["proposals"].pop(i)

def _record_proposal(state, proposal, accepted, adjusted=False):
    entry = {"cls": proposal["cls"], "box": list(map(int, proposal["box"])), "confidence": proposal["confidence"], "note": proposal["note"],
             "model": proposal.get("model"), "source": proposal.get("source", "claude"), "accepted": accepted, "adjusted": adjusted, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = _read_log(state["log"], os.path.basename(state["path"]))
    log.setdefault("proposals", []).append(entry)
    _write_atomic(state["log"], json.dumps(log, indent=2))

def _resolve_proposal(state, mode, choice, accepted, adjusted=False):
    if state is None: return state, "", "Load a scan first.", gr.update(choices=[], value=None)
    proposal = _take_proposal(state, choice)
    if proposal is None: return state, _render(state, mode), "No proposal selected.", gr.update(choices=_proposal_choices(state))
    try:
        _record_proposal(state, proposal, accepted, adjusted)
    except (OSError, ValueError) as e:
        state["proposals"].append(proposal)
        return state, _render(state, mode), f"Not recorded: could not write {os.path.basename(state['log'])}: {e}", gr.update(choices=_proposal_choices(state))
    if accepted:
        _add_box(state, proposal["box"], proposal.get("source", "claude"))
        step = PROPOSAL_STEP.get(proposal["cls"], "?")
        msg = f"Accepted P {proposal['cls']} {proposal['box']} as box {len(state['boxes'])-1} (a step {step} box). " + ("Clear boxes and redraw to adjust, then " if adjusted else "") + "Press Apply to use it."
    else:
        msg = f"Dismissed {proposal['cls']} {proposal['box']} (logged as a hard negative)."
    choices = _proposal_choices(state)
    return state, _render(state, mode), msg, gr.update(choices=choices, value=choices[0] if choices else None)

def accept_proposal(state, mode, choice): return _resolve_proposal(state, mode, choice, True)

def adjust_proposal(state, mode, choice): return _resolve_proposal(state, mode, choice, True, adjusted=True)

def dismiss_proposal(state, mode, choice): return _resolve_proposal(state, mode, choice, False)

def _next_drag_hint(state, mode):
    if not mode.startswith("3"): return ""
    if state["next_is_ref"]: return " Next drag: REFERENCE box."
    return " Next drag: patch box." + ("" if state["ref"] else " Press 'Next drag = reference' before drawing the reference.")

def mark_next_as_reference(state, mode):
    if state is None: return state, "", "Load a scan first."
    if not mode.startswith("3"): return state, _render(state, mode), "Only step 3 uses a reference box."
    state["next_is_ref"] = True
    return state, _render(state, mode), "Next drag: REFERENCE box. Drag over a surface that looks right, on the same rows as the patch."

def _clear_pending_reference(state):
    state["ref"], state["next_is_ref"] = None, False

def change_step(mode, state):
    if state is None: return STEP_INFO[mode], "", _default_strength(mode), ""
    _clear_pending_reference(state)
    return STEP_INFO[mode], _render(state, mode), _default_strength(mode), ("Recipe v2." if mode == RECIPE_MODE else f"Step {mode[0]}.") + _next_drag_hint(state, mode)

def _default_strength(mode): return {"1": 1.3, "2": 0.9, "3": 1.0, "4": 0.45, "5": inpaint.MAX_MASK_FRACTION, "R": recipe.SECOND_GAIN}[mode[0]]

def _render(state, mode):
    cols = [(255, 160, 0)] * len(state["boxes"]) if not mode.startswith("2") else [(0, 200, 255)] * len(state["boxes"])
    boxes, colours = list(state["boxes"]), cols
    if state["ref"]: boxes.append(state["ref"]); colours.append((80, 255, 120))
    return _data_url(_draw_boxes(_draw_proposals(state["prev"], state.get("proposals", []), state["scale"]), boxes, state["scale"], colours))

def clear_boxes(state, mode):
    if state is None: return state, "", ""
    state["boxes"] = []; state["box_sources"] = []; _clear_pending_reference(state)
    return state, _render(state, mode), "Boxes cleared." + _next_drag_hint(state, mode)

def undo(state, mode):
    if state is None or not state["history"]: return state, "" if state is None else _render(state, mode), "Nothing to undo."
    before, added, entry = state["history"].pop()
    if added: del state["applied"][-len(added):]
    state["cur"] = before; state["prev"], _ = _preview(state["cur"])
    return state, _render(state, mode), f"Undone. {len(state['history'])} step(s) remain." + _undo_log_note(state, entry) + _next_drag_hint(state, mode)

def _undo_log_note(state, entry):
    try:
        removed = _remove_last_stage(state, entry)
    except (OSError, ValueError) as e:
        return f" {UNDO_LOG_UNCHANGED_PREFIX}could not update {os.path.basename(state['log'])}: {e})"
    return "" if removed else f" {UNDO_LOG_EDITED_NOTE}"

def _remove_last_stage(state, entry):
    log = _read_log(state["log"], os.path.basename(state["path"]))
    if not log["stages"] or log["stages"][-1] != entry: return False
    log["stages"].pop(); _write_atomic(state["log"], json.dumps(log, indent=2))
    return True

def apply(state, mode, gain, second, max_stops, whole_frame, tile_choice="auto", inpaint_method="auto", inpaint_mask="clip"):
    if state is None: return state, "", "Load a scan first."
    try:
        return _apply_step(state, mode, gain, second, max_stops, whole_frame, tile_choice, inpaint_method, inpaint_mask)
    except ValueError as e:
        return state, _render(state, mode), f"Not applied: {e}"

def _apply_step(state, mode, gain, second, max_stops, whole_frame, tile_choice, inpaint_method="auto", inpaint_mask="clip"):
    if not state["boxes"] and not mode.startswith("4"): return state, _render(state, mode), "Draw at least one box."
    before = state["cur"]; out = before.copy(); notes = []; t = time.time(); H, W = before.shape[:2]
    used, ref, added, extra = list(state["boxes"]), None, [], {}
    tile = _tile_from_choice(tile_choice, before.shape); audit_boxes = None
    if mode == RECIPE_MODE:
        out, info = recipe.run_recipe(before, used, tile=tile)
        notes.append(recipe.describe(info) + f" (fixed gains {recipe.SECOND_GAIN:.1f} / {recipe.TINT_GAIN:.1f})")
        stage, param, added, audit_boxes = "recipe_v2", None, used, info["boxes"]
        extra = {"tile": info["tile"], "second_gain": recipe.SECOND_GAIN, "tint_gain": recipe.TINT_GAIN}
    elif mode.startswith("4"):
        if whole_frame: used = []
        elif not used:
            used = [_pad_box(b, tile, W, H) for b in state["applied"]]
            if not used: notes.append("no leak boxes yet — ran on whole frame")
        for i, b in enumerate(used or [None]):
            out, frac = core.hsv_neutralize(out, b, strength=gain); notes.append(f"{'box %d' % i if b else 'whole frame'}: mask {frac*100:.1f}% of frame")
        stage, param = "hsv", gain
    elif mode.startswith("5"):
        for i, b in enumerate(used):
            out, info = inpaint.inpaint_residue(out, state["orig"], b, method=inpaint_method, mask_mode=inpaint_mask)
            notes.append(f"box {i}: mask {info['mask_fraction']*100:.2f}% of frame, " + (f"filled with {info['method']}" if info["applied"] else "not applied"))
        stage, param, extra = "inpaint", inpaint.MAX_MASK_FRACTION, {"method": info["method"], "mask_mode": inpaint_mask}
    elif mode.startswith("1"):
        for i, b in enumerate(used):
            out, f, af = core.deleak(out, b, gain=gain, second_pass=second, tile=tile); notes.append(f"box {i}: peak {f[...,0].max():.3f}, blue frac {af:.2f}")
        notes.append(f"tile {tile} px"); stage, param, added, extra = "leak", gain, used, {"tile": tile}
    elif mode.startswith("2"):
        for i, b in enumerate(used):
            a0, b0 = core.lab_ab(out, b); out, f, _ = core.detint(out, b, gain=gain, second_pass=second, tile=tile); a1, b1 = core.lab_ab(out, b)
            notes.append(f"box {i}: Lab b {b0:+.1f} → {b1:+.1f}")
        notes.append(f"tile {tile} px"); stage, param, added, extra = "cool_tint", gain, used, {"tile": tile}
    else:
        if state["ref"] is None: return state, _render(state, mode), "Step 3 needs a reference box: press 'Next drag = reference', then drag over a surface that looks right."
        ref = state["ref"]
        for i, b in enumerate(used):
            out, lift = core.exposure_match(out, b, ref, max_stops=max_stops); notes.append(f"box {i}: lift {lift:+.2f} stops")
        stage, param, added = "exposure", max_stops, used
    inside, outside = core.audit(before, out, audit_boxes or used or [(0, W, 0, H)])
    if used == list(state["boxes"]) and state["box_sources"]:
        extra = dict(extra, sources=list(state["box_sources"]))
        if any(source != "human" for source in state["box_sources"]): extra["model"] = _last_proposal_model(state)
    try:
        entry = _log(state, stage, used, ref, param, extra)
    except (OSError, ValueError) as e:
        return state, _render(state, mode), f"Not applied: could not write {os.path.basename(state['log'])}: {e}"
    state["history"].append((before, added, entry)); state["applied"].extend(added); state["cur"] = out; state["prev"], _ = _preview(out)
    state["boxes"] = []; state["box_sources"] = []; _clear_pending_reference(state)
    msg = f"{stage}: " + "; ".join(notes) + f"\nchange inside {inside.round(1)}  outside {outside.round(2)}  ({time.time()-t:.1f}s)" + _next_drag_hint(state, mode)
    return state, _render(state, mode), msg

def _last_proposal_model(state):
    try:
        log = _read_log(state["log"], os.path.basename(state["path"]))
        accepted = [p for p in log.get("proposals", []) if p.get("accepted")]
        return accepted[-1]["model"] if accepted else None
    except (OSError, ValueError):
        return None

def _read_log(path, image_name):
    if not os.path.exists(path): return {"image": image_name, "stages": []}
    try:
        log = json.load(open(path))
    except json.JSONDecodeError as e:
        raise ValueError(f"{os.path.basename(path)} is not valid JSON ({e}); fix or move it before applying")
    if not isinstance(log, dict) or not isinstance(log.get("stages"), list):
        raise ValueError(f"{os.path.basename(path)} has no 'stages' list; fix or move it before applying")
    return log

def _replace_with_temp(path, prefix, suffix, fill_temp):
    folder = os.path.dirname(os.path.abspath(path))
    handle, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=folder)
    os.close(handle)
    try:
        fill_temp(temp_path)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise

def _write_atomic(path, text):
    def fill_temp(temp_path):
        with open(temp_path, "w") as f: f.write(text)
    _replace_with_temp(path, ".boxes-", ".tmp", fill_temp)

def _save_atomic(path, rgb):
    def fill_temp(temp_path):
        core.save(temp_path, rgb)
        if os.path.getsize(temp_path) == 0: raise OSError(f"could not write {os.path.basename(path)} (folder not writable?)")
    _replace_with_temp(path, ".fixed-", ".jpg", fill_temp)

def _stage_entry(stage, boxes, ref, param, extra=None):
    entry = {"stage": stage, "boxes": [list(map(int, b)) for b in boxes]}
    if stage != "recipe_v2": entry.update(ref=list(map(int, ref)) if ref else None, param=float(param))
    entry.update(extra or {}); entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return json.loads(json.dumps(entry))

def _log(state, stage, boxes, ref, param, extra=None):
    log = _read_log(state["log"], os.path.basename(state["path"]))
    entry = _stage_entry(stage, boxes, ref, param, extra)
    log["stages"].append(entry)
    _write_atomic(state["log"], json.dumps(log, indent=2))
    return entry

def save(state):
    if state is None: return None, "Nothing to save."
    p = Path(state["path"]); out = p.with_name(p.stem + "_fixed.jpg")
    try:
        _save_atomic(str(out), state["cur"])
    except (OSError, cv2.error) as e:
        return None, f"Save failed: {e}"
    return str(out), f"Saved {out.name} ({state['cur'].shape[1]}×{state['cur'].shape[0]}) and {Path(state['log']).name}."

def compare(state):
    if state is None: return None
    a, _ = _preview(state["orig"]); b = state["prev"]; return np.hstack([a, b])


def build():
    with gr.Blocks(title="Leak Fix") as demo:
        gr.Markdown("## Leak Fix — light-leak removal for film scans\nDraw boxes (press on the image and drag in any direction), choose a step, Apply. Steps chain: each one works on the previous result. Nothing outside your boxes is changed.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.HTML(value="", label="scan (press and drag to draw a box)", show_label=True, container=True,
                              html_template=STAGE_HTML, css_template=STAGE_CSS, js_on_load=STAGE_JS)
                status = gr.Textbox(label="status", lines=2, interactive=False)
            with gr.Column(scale=1):
                file = gr.File(label="scan", file_types=["image"], type="filepath")
                mode = gr.Radio(list(STEP_INFO), value=RECIPE_MODE, label="step")
                info = gr.Markdown(STEP_INFO[RECIPE_MODE])
                gain = gr.Slider(0.1, 2.0, 1.3, step=0.05, label="strength (steps 1–2: gain · step 4: blend 0–1)")
                second = gr.Checkbox(True, label="leftover pass at half the tile (steps 1–2)")
                tile_choice = gr.Dropdown(list(TILE_CHOICES), value="auto", label="tile, px (steps 1–2: auto = 64 / 32 / 16 by image size)")
                max_stops = gr.Slider(0.25, 2.0, 1.0, step=0.25, label="max lift, stops (step 3)")
                whole_frame = gr.Checkbox(False, label="step 4: whole frame instead of leak boxes")
                with gr.Row():
                    inpaint_method = gr.Dropdown(list(INPAINT_METHODS), value="auto", label="step 5 method")
                    inpaint_mask = gr.Dropdown(list(INPAINT_MASK_MODES), value="clip", label="step 5 mask")
                with gr.Row():
                    apply_btn = gr.Button("Apply", variant="primary"); clear_btn = gr.Button("Clear boxes"); undo_btn = gr.Button("Undo")
                ref_btn = gr.Button("Next drag = reference (step 3)")
                with gr.Group():
                    with gr.Group(visible=propose.available()):
                        model_choice = gr.Dropdown(list(PROPOSAL_MODELS), value=propose.DEFAULT_MODEL, allow_custom_value=True, label="proposer model")
                        propose_btn = gr.Button("Propose boxes (Claude)")
                    with gr.Row():
                        yolo_weights = gr.Textbox(detect.DEFAULT_WEIGHTS, label="YOLO weights")
                        yolo_conf = gr.Slider(0.05, 0.95, detect.DEFAULT_CONF, step=0.05, label="YOLO confidence")
                    yolo_btn = gr.Button(yolo_button_label(), interactive=detect.available())
                    proposal_choice = gr.Dropdown([], label="proposal", interactive=True)
                    with gr.Row():
                        accept_btn = gr.Button("Accept"); adjust_btn = gr.Button("Adjust"); dismiss_btn = gr.Button("Dismiss")
                save_btn = gr.Button("Save full-resolution JPEG"); out_file = gr.File(label="result", interactive=False)
                cmp_btn = gr.Button("Show original | current")
        cmp = gr.Image(label="original | current", interactive=False, visible=True, height=360)

        file.change(load_image, [file, state], [state, img, status, info])
        mode.change(change_step, [mode, state], [info, img, gain, status])
        img.select(on_select, [state, mode], [state, img, status])
        apply_btn.click(apply, [state, mode, gain, second, max_stops, whole_frame, tile_choice, inpaint_method, inpaint_mask], [state, img, status])
        clear_btn.click(clear_boxes, [state, mode], [state, img, status])
        ref_btn.click(mark_next_as_reference, [state, mode], [state, img, status])
        undo_btn.click(undo, [state, mode], [state, img, status])
        propose_btn.click(propose_click, [state, mode, model_choice], [state, img, status, proposal_choice])
        yolo_btn.click(yolo_propose_click, [state, mode, yolo_weights, yolo_conf], [state, img, status, proposal_choice])
        accept_btn.click(accept_proposal, [state, mode, proposal_choice], [state, img, status, proposal_choice])
        adjust_btn.click(adjust_proposal, [state, mode, proposal_choice], [state, img, status, proposal_choice])
        dismiss_btn.click(dismiss_proposal, [state, mode, proposal_choice], [state, img, status, proposal_choice])
        save_btn.click(save, [state], [out_file, status])
        cmp_btn.click(compare, [state], [cmp])
    return demo


def main():
    build().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)

if __name__ == "__main__":
    main()

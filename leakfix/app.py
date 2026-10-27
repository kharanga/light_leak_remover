import base64, json, os, tempfile, time, numpy as np, cv2, gradio as gr
from pathlib import Path
from . import core, recipe, compare as collateral

PREVIEW_MAX = 1600
STEP_INFO = {
    "1 · Remove leak (warm)":   "Draw one box per light leak. The tile method subtracts the orange cast measured against the tiles just outside the box.",
    "2 · Remove blue tint (cool)": "Draw a box around any cyan/blue patch left after step 1. Same method with channels swapped, gentler.",
    "3 · Match exposure":       "Draw one box per dull patch, press 'Next drag = reference', then drag a REFERENCE box of the same surface that looks right (same rows). All patches are lifted to the reference's brightness in linear light.",
    "4 · Neutralize warm (HSV)": "Final pass, automatic: masks bright warm pixels (hue 0–25, sat ≥ 40, value ≥ 180) inside the leak boxes from steps 1–3 (padded by one tile), softens the mask and pulls red toward green by the strength. Tick 'whole frame' to run everywhere; draw boxes to override.",
    "R · Recipe v2 (main path)": "The main path: draw one box per light leak and press Apply. Runs the whole recipe at fixed gains (no strength control): the boxes are padded by two tiles, two deleak passes (1.3, 1.3), blue tint only where it is cyan against the same row, exposure matched to a side strip whose rows follow the box edge, HSV 0.45, a black-point floor, then a per-tile fade so only leak tiles change. Logged as one recipe_v2 stage that `cli --replay` reproduces. Steps 1–4 stay available as the manual path.",
}
RECIPE_MODE = "R · Recipe v2 (main path)"
UNDO_LOG_EDITED_NOTE = "log not changed (boxes.json was edited elsewhere)"
UNDO_LOG_UNCHANGED_PREFIX = "log not changed ("
TILE_CHOICES = ("auto", "64", "32", "16")
COMPARE_SLOTS = 3
COMPARE_OUT_DIR = "work/compare"
COMPARE_CSV = "work/compare/compare.csv"
COMPARE_FILE_BOX_COLOUR = (255, 255, 255)
COMPARE_DRAWN_BOX_COLOUR = (255, 160, 0)

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

def _data_url(rgb_u8):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

def _new_state(path):
    rgb = core.load(path); prev, s = _preview(rgb)
    return {"path": path, "orig": rgb, "cur": rgb.copy(), "history": [], "scale": s, "prev": prev,
            "boxes": [], "box_sources": [], "ref": None, "next_is_ref": False, "applied": [],
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

def _default_strength(mode): return {"1": 1.3, "2": 0.9, "3": 1.0, "4": 0.45, "R": recipe.SECOND_GAIN}[mode[0]]

def _render(state, mode):
    cols = [(255, 160, 0)] * len(state["boxes"]) if not mode.startswith("2") else [(0, 200, 255)] * len(state["boxes"])
    boxes, colours = list(state["boxes"]), cols
    if state["ref"]: boxes.append(state["ref"]); colours.append((80, 255, 120))
    return _data_url(_draw_boxes(state["prev"], boxes, state["scale"], colours))

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

def apply(state, mode, gain, second, max_stops, whole_frame, tile_choice="auto"):
    if state is None: return state, "", "Load a scan first."
    try:
        return _apply_step(state, mode, gain, second, max_stops, whole_frame, tile_choice)
    except ValueError as e:
        return state, _render(state, mode), f"Not applied: {e}"

def _apply_step(state, mode, gain, second, max_stops, whole_frame, tile_choice):
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
    try:
        entry = _log(state, stage, used, ref, param, extra)
    except (OSError, ValueError) as e:
        return state, _render(state, mode), f"Not applied: could not write {os.path.basename(state['log'])}: {e}"
    state["history"].append((before, added, entry)); state["applied"].extend(added); state["cur"] = out; state["prev"], _ = _preview(out)
    state["boxes"] = []; state["box_sources"] = []; _clear_pending_reference(state)
    msg = f"{stage}: " + "; ".join(notes) + f"\nchange inside {inside.round(1)}  outside {outside.round(2)}  ({time.time()-t:.1f}s)" + _next_drag_hint(state, mode)
    return state, _render(state, mode), msg

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


def _fix_tab_log_for(fix_state, original_path):
    if not fix_state or os.path.basename(fix_state["path"]) != os.path.basename(original_path): return None
    return fix_state["log"] if os.path.exists(fix_state["log"]) else None

def _compare_boxes(cstate):
    return collateral.unique_union(cstate["file_boxes"] + cstate["drawn_boxes"])

def _compare_render(cstate):
    boxes = cstate["file_boxes"] + cstate["drawn_boxes"]
    colours = [COMPARE_FILE_BOX_COLOUR] * len(cstate["file_boxes"]) + [COMPARE_DRAWN_BOX_COLOUR] * len(cstate["drawn_boxes"])
    return _data_url(_draw_boxes(cstate["prev"], boxes, cstate["scale"], colours))

def _compare_box_note(cstate):
    return f"{len(cstate['file_boxes'])} box(es) from {cstate['boxes_source'] or 'no boxes.json'}, {len(cstate['drawn_boxes'])} drawn."

def _with_file_boxes(cstate, path, source):
    try:
        cstate["file_boxes"] = collateral.read_label_boxes(path); cstate["boxes_source"] = source
        return ""
    except (ValueError, OSError) as e:
        cstate["file_boxes"] = []; cstate["boxes_source"] = None
        return f" Boxes file ignored: {e}."

def compare_load_original(file, fix_state):
    if file is None: return None, "", "Load the original scan to compare against."
    try:
        rgb = core.load(file)
    except (FileNotFoundError, ValueError) as e:
        return None, "", f"Could not load the original: {e}"
    prev, s = _preview(rgb)
    cstate = {"path": file, "shape": rgb.shape, "prev": prev, "scale": s, "file_boxes": [], "drawn_boxes": [], "boxes_source": None}
    log = _fix_tab_log_for(fix_state, file)
    note = _with_file_boxes(cstate, log, f"{Path(log).name} (Fix tab)") if log else ""
    return cstate, _compare_render(cstate), f"{rgb.shape[1]}×{rgb.shape[0]} original loaded. " + _compare_box_note(cstate) + note + " Drag on the image to add boxes."

def compare_load_boxes(file, cstate):
    if cstate is None: return cstate, "", "Load the original first."
    if file is None:
        cstate["file_boxes"] = []; cstate["boxes_source"] = None
        return cstate, _compare_render(cstate), _compare_box_note(cstate)
    note = _with_file_boxes(cstate, file, Path(file).name)
    return cstate, _compare_render(cstate), _compare_box_note(cstate) + note

def compare_on_select(cstate, evt: gr.SelectData):
    if cstate is None: return cstate, "", "Load the original first."
    box = tuple(int(v / cstate["scale"]) for v in evt.index)
    if box[1] - box[0] < 16 or box[3] - box[2] < 16: return cstate, _compare_render(cstate), "Box too small, drag a larger one."
    cstate["drawn_boxes"].append(box)
    return cstate, _compare_render(cstate), f"Drawn box {box}. " + _compare_box_note(cstate)

def compare_clear_drawn(cstate):
    if cstate is None: return cstate, "", "Load the original first."
    cstate["drawn_boxes"] = []
    return cstate, _compare_render(cstate), _compare_box_note(cstate)

def _compare_versions_from(slots):
    versions = []
    for index, (path, label) in enumerate(slots, start=1):
        if path: versions.append(((label or "").strip() or Path(path).stem or f"version{index}", path))
    return versions

def compare_run(cstate, margin, csv_path, out_dir, *slot_values):
    if cstate is None: return None, "", "Load the original first."
    slots = list(zip(slot_values[0::2], slot_values[1::2]))
    lines = []
    try:
        rows, figure_path = collateral.compare_versions(cstate["path"], _compare_versions_from(slots), _compare_boxes(cstate),
                                                        int(margin), csv_path or COMPARE_CSV, out_dir or COMPARE_OUT_DIR, report=lines.append)
    except (ValueError, OSError) as e:
        return None, "", f"Compare failed: {e}"
    return figure_path, "```\n" + collateral.table_text(rows) + "\n```", "\n".join(line for line in lines if "\n" not in line)


def build():
    with gr.Blocks(title="Leak Fix") as demo:
        gr.Markdown("## Leak Fix — light-leak removal for film scans\nDraw boxes (press on the image and drag in any direction), choose a step, Apply. Steps chain: each one works on the previous result. Nothing outside your boxes is changed.")
        state = gr.State(None)
        with gr.Tab("Fix"):
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
                        apply_btn = gr.Button("Apply", variant="primary"); clear_btn = gr.Button("Clear boxes"); undo_btn = gr.Button("Undo")
                    ref_btn = gr.Button("Next drag = reference (step 3)")
                    save_btn = gr.Button("Save full-resolution JPEG"); out_file = gr.File(label="result", interactive=False)
                    cmp_btn = gr.Button("Show original | current")
            cmp = gr.Image(label="original | current", interactive=False, visible=True, height=360)

        with gr.Tab("Compare"):
            gr.Markdown("Measure how much each version changed the scan **outside** the leak boxes (ΔE76 / ΔE2000 after SIFT alignment). Load the original; its boxes come from the Fix tab's boxes.json when it is the same scan, from an uploaded boxes.json, or drag on the image to draw them.")
            cstate = gr.State(None)
            with gr.Row():
                with gr.Column(scale=3):
                    cimg = gr.HTML(value="", label="original (press and drag to draw a box)", show_label=True, container=True,
                                   html_template=STAGE_HTML, css_template=STAGE_CSS, js_on_load=STAGE_JS)
                    cstatus = gr.Textbox(label="status", lines=3, interactive=False)
                with gr.Column(scale=1):
                    coriginal = gr.File(label="original scan", file_types=["image"], type="filepath")
                    cboxes = gr.File(label="boxes.json (optional)", file_types=[".json"], type="filepath")
                    cclear = gr.Button("Clear drawn boxes")
                    slot_inputs = []
                    for index in range(1, COMPARE_SLOTS + 1):
                        with gr.Row():
                            slot_inputs.append(gr.File(label=f"version {index}", file_types=["image"], type="filepath"))
                            slot_inputs.append(gr.Textbox(label=f"label {index}", placeholder="e.g. gemini-3.1-pro"))
                    cmargin = gr.Number(collateral.DEFAULT_MARGIN, precision=0, label="margin around boxes, px")
                    ccsv = gr.Textbox(COMPARE_CSV, label="CSV (rows appended)")
                    cout = gr.Textbox(COMPARE_OUT_DIR, label="output folder")
                    crun = gr.Button("Run", variant="primary")
            cfigure = gr.Image(label="comparison", type="filepath", interactive=False)
            ctable = gr.Markdown()

        file.change(load_image, [file, state], [state, img, status, info])
        mode.change(change_step, [mode, state], [info, img, gain, status])
        img.select(on_select, [state, mode], [state, img, status])
        apply_btn.click(apply, [state, mode, gain, second, max_stops, whole_frame, tile_choice], [state, img, status])
        clear_btn.click(clear_boxes, [state, mode], [state, img, status])
        ref_btn.click(mark_next_as_reference, [state, mode], [state, img, status])
        undo_btn.click(undo, [state, mode], [state, img, status])
        save_btn.click(save, [state], [out_file, status])
        cmp_btn.click(compare, [state], [cmp])
        coriginal.change(compare_load_original, [coriginal, state], [cstate, cimg, cstatus])
        cboxes.change(compare_load_boxes, [cboxes, cstate], [cstate, cimg, cstatus])
        cimg.select(compare_on_select, [cstate], [cstate, cimg, cstatus])
        cclear.click(compare_clear_drawn, [cstate], [cstate, cimg, cstatus])
        crun.click(compare_run, [cstate, cmargin, ccsv, cout] + slot_inputs, [cfigure, ctable, cstatus])
    return demo


def main():
    build().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)

if __name__ == "__main__":
    main()

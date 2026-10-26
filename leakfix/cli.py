import argparse, json, os, time
from . import core, inpaint, recipe

STEP_FILES = {"leak": "step1_deleak.jpg", "cool_tint": "step2_detint.jpg", "exposure": "step3_final.jpg", "hsv": "step4_hsv.jpg", "inpaint": "step5_inpaint.jpg"}

def box(s): return tuple(int(v) for v in s.split(","))

def pad_box(box, pad, shape):
    H, W = shape[:2]; x0, x1, y0, y1 = box
    return (max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad), min(H, y1 + pad))

def hsv_boxes_for(a, shape):
    if a.hsv is None: return []
    if a.hsv_box: return list(a.hsv_box)
    if a.hsv_whole_frame: return [None]
    padded = [pad_box(b, core.resolve_tile(a.tile, shape), shape) for b in a.leak + a.tint + a.expose]
    return list(dict.fromkeys(padded)) or [None]

def stage_audit_boxes(st, shape):
    if st["stage"] == "recipe_v2": return recipe.padded_leak_boxes(st["boxes"], core.resolve_tile(st.get("tile"), shape), shape)
    return [tuple(b) for b in st["boxes"]]

def audited_boxes(a, stages, shape):
    H, W = shape[:2]; whole_frame = [(0, W, 0, H)]
    if stages is not None:
        if any(st["stage"] == "hsv" and not st["boxes"] for st in stages): return whole_frame
        return [b for st in stages for b in stage_audit_boxes(st, shape)]
    hsv_boxes = hsv_boxes_for(a, shape)
    if None in hsv_boxes: return whole_frame
    return a.leak + a.tint + a.expose + hsv_boxes + list(a.inpaint)

def step_file_path(out_path, stage):
    return os.path.join(os.path.dirname(out_path) or ".", STEP_FILES[stage])

def save_step(a, out_path, stage, rgb):
    if not a.save_steps: return
    path = step_file_path(out_path, stage); core.save(path, rgb); print("saved", path)

REPLAY_STAGES = ("leak", "cool_tint", "exposure", "hsv", "inpaint", "recipe_v2")
REPLAY_DEFAULT_TILE = 64
CLASSIC_GAIN = 1.3
CLASSIC_TINT_GAIN = 0.9
RECIPE_CONFLICTS = (("gain", "--gain"), ("tint_gain", "--tint-gain"), ("tint", "--tint"), ("expose", "--expose"), ("hsv", "--hsv"), ("hsv_box", "--hsv-box"), ("hsv_whole_frame", "--hsv-whole-frame"), ("inpaint", "--inpaint"), ("replay", "--replay"), ("ref", "--ref"), ("save_steps", "--save-steps"))

def tile_choice(s):
    if s.lower() == "auto": return None
    if s not in ("64", "32", "16"): raise argparse.ArgumentTypeError(f"--tile must be auto, 64, 32 or 16, not {s!r}")
    return int(s)

def validate_stages(stages):
    for i, st in enumerate(stages):
        name = st.get("stage")
        if name == "inpaint" and st.get("method") not in ("telea", "lama"): raise SystemExit(f"stage {i}: inpaint stage needs \"method\": \"telea\" or \"lama\" to replay")
        if name not in REPLAY_STAGES: raise SystemExit(f"stage {i}: unknown stage name {name!r}; expected one of {', '.join(REPLAY_STAGES)}")
        if name == "exposure" and not st.get("ref"): raise SystemExit(f"stage {i}: exposure stage has no ref box")
        if name == "recipe_v2":
            missing = [k for k in ("second_gain", "tint_gain") if st.get(k) is None]
            if missing: raise SystemExit(f"stage {i}: recipe_v2 stage needs {' and '.join(missing)} to replay")
            if not st.get("boxes"): raise SystemExit(f"stage {i}: recipe_v2 stage has no leak boxes")

def replay_recipe_stage(rgb, st):
    rgb, _ = recipe.run_recipe(rgb, [tuple(b) for b in st["boxes"]], second_gain=st["second_gain"], tint_gain=st["tint_gain"], tile=st.get("tile"))
    return rgb

def replay_stages(rgb, stages, original=None):
    validate_stages(stages)
    original = rgb if original is None else original
    for st in stages:
        if st["stage"] == "recipe_v2": rgb = replay_recipe_stage(rgb, st); print(f"replayed recipe_v2: {len(st['boxes'])} box(es)"); continue
        for b in st["boxes"]:
            if st["stage"] == "inpaint": rgb, _ = inpaint.inpaint_residue(rgb, original, b, method=st["method"], mask_mode=st.get("mask_mode", "clip")); continue
            if st["stage"] == "leak": rgb, _, _ = core.deleak(rgb, b, gain=st["param"], tile=st.get("tile", REPLAY_DEFAULT_TILE))
            elif st["stage"] == "cool_tint": rgb, _, _ = core.detint(rgb, b, gain=st["param"], tile=st.get("tile", REPLAY_DEFAULT_TILE))
            elif st["stage"] == "exposure": rgb, _ = core.exposure_match(rgb, b, st["ref"], max_stops=st["param"])
            elif st["stage"] == "hsv": rgb, _ = core.hsv_neutralize(rgb, b, strength=st["param"])
        if st["stage"] == "hsv" and not st["boxes"]: rgb, _ = core.hsv_neutralize(rgb, None, strength=st["param"])
        print(f"replayed {st['stage']}: {len(st['boxes'])} box(es)")
    return rgb

def load_or_exit(path):
    try: return core.load(path)
    except (FileNotFoundError, ValueError) as e: raise SystemExit(str(e))

def steps_or_exit(run_steps):
    try: return run_steps()
    except ValueError as e: raise SystemExit(str(e))

def main():
    p = argparse.ArgumentParser(); p.add_argument("image"); p.add_argument("-o", "--out", default=None)
    p.add_argument("--leak", type=box, action="append", default=[]); p.add_argument("--tint", type=box, action="append", default=[])
    p.add_argument("--expose", type=box, action="append", default=[]); p.add_argument("--ref", type=box, default=None)
    p.add_argument("--gain", type=float, default=None, help="step 1 gain (default 1.3)"); p.add_argument("--tint-gain", type=float, default=None, help="step 2 gain (default 0.9)"); p.add_argument("--max-stops", type=float, default=1.0)
    p.add_argument("--tile", type=tile_choice, default="auto", help="tile size for steps 1-2: auto (64 / 32 / 16 by image size), 64, 32 or 16")
    p.add_argument("--replay", default=None, help="boxes.json written by the app")
    p.add_argument("--hsv", type=float, default=None, help="final HSV warm-neutralize pass, blend strength 0-1 (e.g. 0.45); whole frame unless --hsv-box")
    p.add_argument("--hsv-box", type=box, action="append", default=[])
    p.add_argument("--inpaint", type=box, action="append", default=[], help="step 5 box (opt-in): inpaint the clipped cores of the ORIGINAL inside this box")
    p.add_argument("--inpaint-method", choices=("auto", "telea", "lama"), default="auto")
    p.add_argument("--inpaint-mask", choices=("clip", "score", "box"), default="clip")
    p.add_argument("--hsv-whole-frame", action="store_true", help="run --hsv on the whole frame instead of inside the steps 1-3 boxes padded by one tile")
    p.add_argument("--save-steps", action="store_true", help="also write step1_deleak.jpg, step2_detint.jpg, step3_final.jpg, step4_hsv.jpg next to the output, one per step run")
    p.add_argument("--recipe", choices=("v2",), default=None, help="run recipe v2 on the --leak boxes (two deleak passes, gated blue tint, matched exposure, hsv, black-point floor, tile fade) and write <out>.boxes.json")
    a = p.parse_args()
    check_recipe_flags(a)
    if not a.recipe: a.gain, a.tint_gain = with_default(a.gain, CLASSIC_GAIN), with_default(a.tint_gain, CLASSIC_TINT_GAIN)
    rgb = load_or_exit(a.image); orig = rgb.copy()
    out = a.out or a.image.rsplit(".", 1)[0] + "_fixed.jpg"
    stages = json.load(open(a.replay))["stages"] if a.replay else None
    if a.recipe:
        rgb, stages = steps_or_exit(lambda: run_recipe_v2(a, rgb))
        write_recipe_label(label_path_for(out), a.image, stages)
    else:
        rgb = steps_or_exit(lambda: apply_stages(a, rgb, stages, out))
    inside, outside = core.audit(orig, rgb, audited_boxes(a, stages, rgb.shape)); print(f"change inside {inside.round(1)}  outside {outside.round(2)}")
    core.save(out, rgb); print("saved", out)

def flag_was_given(value):
    if value is None or value is False: return False
    return not isinstance(value, list) or len(value) > 0

def check_recipe_flags(a):
    if not a.recipe: return
    used = [flag for attr, flag in RECIPE_CONFLICTS if flag_was_given(getattr(a, attr))]
    if used: raise SystemExit(f"--recipe v2 cannot be combined with {', '.join(used)}: the recipe runs its own blue tint, exposure and hsv steps at fixed gains and writes no per-step files")
    if not a.leak: raise SystemExit("--recipe v2 needs at least one --leak box")

def with_default(value, default):
    return default if value is None else value

def recipe_stage(boxes, info, second_gain, tint_gain):
    return {"stage": "recipe_v2", "boxes": [list(b) for b in boxes], "tile": info["tile"], "second_gain": second_gain, "tint_gain": tint_gain,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}

def run_recipe_v2(a, rgb):
    fixed, info = recipe.run_recipe(rgb, a.leak, tile=a.tile)
    print(f"recipe v2 {len(a.leak)} box(es) | " + recipe.describe(info))
    return fixed, [recipe_stage(a.leak, info, recipe.SECOND_GAIN, recipe.TINT_GAIN)]

def label_path_for(out_path):
    return out_path.rsplit(".", 1)[0] + ".boxes.json"

def write_recipe_label(path, image_path, stages):
    temp = path + ".tmp"
    with open(temp, "w") as f: json.dump({"image": os.path.basename(image_path), "stages": stages}, f, indent=2)
    os.replace(temp, path); print("saved", path)

def apply_stages(a, rgb, stages, out):
    original = rgb
    if a.replay:
        rgb = replay_stages(rgb, stages, original)
    else:
        tile = core.resolve_tile(a.tile, rgb.shape)
        if a.leak or a.tint: print(f"tile {tile} px" + (" (auto)" if a.tile is None else ""))
        for b in a.leak: rgb, f, af = core.deleak(rgb, b, gain=a.gain, tile=tile); print(f"leak {b}: peak {f[...,0].max():.3f} blue frac {af:.2f}")
        if a.leak: save_step(a, out, "leak", rgb)
        for b in a.tint: rgb, _, _ = core.detint(rgb, b, gain=a.tint_gain, tile=tile); print(f"tint {b}")
        if a.tint: save_step(a, out, "cool_tint", rgb)
        for b in a.expose:
            if a.ref is None: raise SystemExit("--expose needs --ref")
            rgb, lift = core.exposure_match(rgb, b, a.ref, max_stops=a.max_stops); print(f"expose {b}: {lift:+.2f} stops")
        if a.expose: save_step(a, out, "exposure", rgb)
        if a.hsv is not None:
            hsv_boxes = hsv_boxes_for(a, rgb.shape)
            if hsv_boxes == [None] and not a.hsv_whole_frame and not a.hsv_box: print("no leak boxes: hsv runs on the whole frame")
            for b in hsv_boxes: rgb, frac = core.hsv_neutralize(rgb, b, strength=a.hsv); print(f"hsv {b or 'whole frame'}: mask {frac*100:.1f}%")
            save_step(a, out, "hsv", rgb)
        for b in a.inpaint:
            rgb, info = inpaint.inpaint_residue(rgb, original, b, method=a.inpaint_method, mask_mode=a.inpaint_mask)
            print(f"inpaint {b}: mask {info['mask_fraction']*100:.2f}% " + (f"filled with {info['method']}" if info["applied"] else "not applied"))
        if a.inpaint: save_step(a, out, "inpaint", rgb)
    return rgb

if __name__ == "__main__":
    main()

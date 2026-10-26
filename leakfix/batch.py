import argparse, contextlib, csv, glob, io, json, os, shutil, time
import numpy as np, cv2
from . import core, cli, recipe, inpaint

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
GRID_LONG_SIDE = 900
GRID_COLOUR = (255, 255, 0)
PANEL_LONG_SIDE = 900
VERDICTS = ("", "good", "partial-clip", "bad", "no-leak")
RECIPES = (None, "v2")
MANIFEST_COLUMNS = ("scan", "size", "verdict", "boxes", "gain", "tile", "inside", "outside", "accepted")
INPAINT_SUMMARY_COLUMNS = ("scan", "mask_pct", "method", "status")
INPAINT_METHOD_FREE_STATUSES = ("skipped-empty", "skipped-large", "skipped-no-recipe")
DEFAULT_ORIGINAL, DEFAULT_FIXED, DEFAULT_LABELS, DEFAULT_COMPARE = "photos/original", "photos/fixed", "photos/labels", "photos/compare"


def scan_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def list_scans(folder):
    paths = [p for p in sorted(glob.glob(os.path.join(folder, "*"))) if p.lower().endswith(IMAGE_EXTENSIONS)]
    return {scan_stem(p): p for p in paths}


def find_scan(original_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        for candidate in (os.path.join(original_dir, stem + ext), os.path.join(original_dir, stem + ext.upper())):
            if os.path.isfile(candidate): return candidate
    raise FileNotFoundError(f"no image for {stem!r} under {original_dir}")


def grid_step(shape):
    return 50 if max(shape[0], shape[1]) < 600 else 100


def grid_image(rgb, long_side=GRID_LONG_SIDE):
    H, W = rgb.shape[:2]; s = long_side / max(H, W); step = grid_step(rgb.shape)
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    big = cv2.resize(u8, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    for x in range(0, W, step):
        X = int(x * s); cv2.line(big, (X, 0), (X, big.shape[0]), GRID_COLOUR, 1)
        cv2.putText(big, str(x), (X + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRID_COLOUR, 1)
    for y in range(0, H, step):
        Y = int(y * s); cv2.line(big, (0, Y), (big.shape[1], Y), GRID_COLOUR, 1)
        cv2.putText(big, str(y), (2, Y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRID_COLOUR, 1)
    return big


def write_grid(scan_path, out_dir):
    rgb = core.load(scan_path); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, scan_stem(scan_path) + "_grid.png")
    cv2.imwrite(out, cv2.cvtColor(grid_image(rgb), cv2.COLOR_RGB2BGR))
    return out, rgb.shape[1], rgb.shape[0], grid_step(rgb.shape)


def read_box_file(path):
    try:
        spec = json.load(open(path))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e})")
    if not isinstance(spec, dict): raise ValueError(f"{path} must hold a JSON object")
    for key in ("leak", "tint"):
        boxes = spec.get(key) or []
        if not all(isinstance(b, (list, tuple)) and len(b) == 4 for b in boxes): raise ValueError(f"{path}: every {key} box must be [x0, x1, y0, y1]")
        spec[key] = [tuple(int(v) for v in b) for b in boxes]
    spec["recipe"] = spec.get("recipe")
    if spec["recipe"] not in RECIPES: raise ValueError(f"{path}: recipe must be \"v2\" or absent, not {spec['recipe']!r}")
    if spec["recipe"] and spec["tint"]: raise ValueError(f"{path}: a \"recipe\": \"v2\" box file takes leak boxes only; the recipe runs its own blue tint")
    spec["gain"] = recipe.SECOND_GAIN if spec["recipe"] else float(spec.get("gain", cli.CLASSIC_GAIN))
    spec["tint_gain"] = recipe.TINT_GAIN if spec["recipe"] else float(spec.get("tint_gain", cli.CLASSIC_TINT_GAIN))
    spec["hsv"] = None if spec.get("hsv") is None else float(spec["hsv"])
    spec["hsv_whole_frame"] = bool(spec.get("hsv_whole_frame", False))
    spec["verdict"] = str(spec.get("verdict", "") or ""); spec["note"] = str(spec.get("note", "") or "")
    if spec["verdict"] not in VERDICTS: raise ValueError(f"{path}: verdict must be one of {', '.join(repr(v) for v in VERDICTS)}")
    return spec


def validate_boxes(spec, shape):
    for b in spec["leak"] + spec["tint"]: core.clamp_box(b, shape)


def pad_box(box, pad, shape):
    H, W = shape[:2]; x0, x1, y0, y1 = box
    return (max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad), min(H, y1 + pad))


def hsv_boxes_for(spec, tile, shape):
    if spec["hsv"] is None: return []
    if spec["hsv_whole_frame"]: return [None]
    padded = [pad_box(b, tile, shape) for b in spec["leak"] + spec["tint"]]
    return list(dict.fromkeys(padded))


def apply_recipe_file(original, spec, gain):
    if not spec["leak"]: return original, [], ["recipe v2: no leak boxes"], *core.audit(original, original, [])
    rgb, info = recipe.run_recipe(original, spec["leak"], second_gain=gain, tint_gain=spec["tint_gain"], tile=spec.get("tile"))
    stage = {"stage": "recipe_v2", "boxes": [list(b) for b in spec["leak"]], "tile": info["tile"], "second_gain": gain, "tint_gain": spec["tint_gain"]}
    inside, outside = core.audit(original, rgb, info["boxes"])
    return rgb, [stage], [recipe.describe(info)], inside, outside


def apply_box_file(original, spec, gain=None):
    gain = spec["gain"] if gain is None else float(gain)
    if spec["recipe"] == "v2": return apply_recipe_file(original, spec, gain)
    tile = core.resolve_tile(spec.get("tile"), original.shape)
    validate_boxes(spec, original.shape)
    rgb, stages, notes = original, [], [f"tile {tile}"]
    for b in spec["leak"]:
        rgb, f, af = core.deleak(rgb, b, gain=gain, tile=tile); notes.append(f"leak {b} peak {f[..., 0].max():.3f} bf {af:.2f}")
    if spec["leak"]: stages.append({"stage": "leak", "boxes": [list(b) for b in spec["leak"]], "ref": None, "param": gain, "tile": tile})
    for b in spec["tint"]:
        a0, b0 = core.lab_ab(rgb, b); rgb, _, _ = core.detint(rgb, b, gain=spec["tint_gain"], tile=tile); a1, b1 = core.lab_ab(rgb, b)
        notes.append(f"tint {b} Lab b {b0:+.1f}->{b1:+.1f}")
    if spec["tint"]: stages.append({"stage": "cool_tint", "boxes": [list(b) for b in spec["tint"]], "ref": None, "param": spec["tint_gain"], "tile": tile})
    hsv_boxes = hsv_boxes_for(spec, tile, original.shape)
    for b in hsv_boxes:
        rgb, frac = core.hsv_neutralize(rgb, b, strength=spec["hsv"]); notes.append(f"hsv {b or 'whole frame'} mask {frac * 100:.1f}%")
    if hsv_boxes: stages.append({"stage": "hsv", "boxes": [list(b) for b in hsv_boxes if b is not None], "ref": None, "param": spec["hsv"]})
    H, W = original.shape[:2]
    audit_boxes = [(0, W, 0, H)] if None in hsv_boxes else [tuple(b) for st in stages for b in st["boxes"]]
    inside, outside = core.audit(original, rgb, audit_boxes)
    return rgb, stages, notes, inside, outside


def label_record(image_name, spec, stages, shape, inside, outside):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for st in stages: st["time"] = stamp
    return {"image": image_name, "source": "claude-session", "verdict": spec["verdict"], "note": spec["note"],
            "size": f"{shape[1]}x{shape[0]}", "audit": audit_record(inside, outside),
            "stages": stages}


def panel(rgb, long_side=PANEL_LONG_SIDE, caption=None):
    H, W = rgb.shape[:2]; s = min(1.0, long_side / max(H, W))
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    im = cv2.resize(u8, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else u8.copy()
    if caption: cv2.putText(im, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return im


def side_by_side(panels):
    height = min(p.shape[0] for p in panels)
    return np.hstack([p[:height] for p in panels])


def result_caption(spec, gain):
    return "recipe v2" if spec["recipe"] else f"gain {gain:g}"


def write_outputs(stem, image_name, original, spec, fixed_dir, labels_dir, compare_dir, variants):
    results = [(spec["gain"], *apply_box_file(original, spec))]
    for g in variants:
        if g != spec["gain"]: results.append((g, *apply_box_file(original, spec, gain=g)))
    gain, fixed, stages, notes, inside, outside = results[0]
    for folder in (fixed_dir, labels_dir, compare_dir): os.makedirs(folder, exist_ok=True)
    core.save(os.path.join(fixed_dir, f"{stem}_fixed.jpg"), fixed)
    json.dump(label_record(image_name, spec, stages, original.shape, inside, outside), open(os.path.join(labels_dir, f"{stem}.boxes.json"), "w"), indent=2)
    panels = [panel(original, caption="original")] + [panel(r[1], caption=result_caption(spec, r[0])) for r in results]
    cv2.imwrite(os.path.join(compare_dir, f"{stem}_cmp.jpg"), cv2.cvtColor(side_by_side(panels), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    lines = [f"{stem}: {result_caption(spec, r[0])} | " + "; ".join(r[3]) + f" | inside {r[4].round(1)} outside {r[5].round(2)}" for r in results]
    return lines


def run_box_file(box_path, original_dir, fixed_dir, labels_dir, compare_dir, variants=()):
    stem = scan_stem(box_path); spec = read_box_file(box_path)
    if spec["recipe"] and variants: raise ValueError(f"{box_path}: --variants does not apply to a \"recipe\": \"v2\" box file; the recipe gains are fixed at {recipe.SECOND_GAIN:.1f} / {recipe.TINT_GAIN:.1f}")
    scan_path = find_scan(original_dir, stem); original = core.load(scan_path)
    validate_boxes(spec, original.shape)
    return write_outputs(stem, os.path.basename(scan_path), original, spec, fixed_dir, labels_dir, compare_dir, variants)


def scan_state(stem, boxes_dir, labels_dir):
    label_path = os.path.join(labels_dir, f"{stem}.boxes.json")
    if os.path.isfile(label_path):
        try: verdict = json.load(open(label_path)).get("verdict", "") or ""
        except (json.JSONDecodeError, AttributeError): verdict = ""
        return (verdict or "run"), verdict
    if os.path.isfile(os.path.join(boxes_dir, f"{stem}.json")): return "boxes written", ""
    return "unlabelled", ""


def status_rows(original_dir, boxes_dir, labels_dir):
    rows = []
    for stem, path in list_scans(original_dir).items():
        state, verdict = scan_state(stem, boxes_dir, labels_dir)
        rows.append({"scan": stem, "state": state, "verdict": verdict})
    return rows


def format_status(rows):
    width = max([len(r["scan"]) for r in rows] + [4])
    counts = {}
    for r in rows: counts[r["state"]] = counts.get(r["state"], 0) + 1
    lines = [f"{'scan':<{width}}  state"] + [f"{r['scan']:<{width}}  {r['state']}" for r in rows]
    lines.append(f"{len(rows)} scan(s): " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def parse_accepted_line(line):
    parts = line.split()
    if len(parts) == 1: return parts[0], None
    if len(parts) == 2:
        try: return parts[0], float(parts[1])
        except ValueError: pass
    raise ValueError(f"accepted line {line.strip()!r} must be '<stem>' or '<stem> <gain>'")


def read_accepted(path):
    if not os.path.isfile(path): return {}
    entries = [parse_accepted_line(line) for line in open(path) if line.strip() and not line.lstrip().startswith("#")]
    return dict(entries)


def label_with_gain(label, gain):
    chosen = json.loads(json.dumps(label))
    for st in chosen.get("stages", []):
        if st["stage"] == "leak": st["param"] = float(gain)
        if st["stage"] == "recipe_v2": st["second_gain"] = float(gain)
    chosen["chosen_gain"] = float(gain)
    return chosen


def replay_stages_quietly(original, stages):
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.replay_stages(original, stages)


def replay_label(label, original_dir, stem):
    return replay_stages_quietly(core.load(find_scan(original_dir, stem)), label["stages"])


def label_audit_boxes(stages, shape):
    H, W = shape[:2]
    if any(st["stage"] == "hsv" and not st["boxes"] for st in stages): return [(0, W, 0, H)]
    return [b for st in stages for b in cli.stage_audit_boxes(st, shape)]


def audit_record(inside, outside):
    return {"inside": [round(float(v), 2) + 0.0 for v in inside], "outside": [round(float(v), 2) + 0.0 for v in outside]}


def manifest_row(label, stem, accepted):
    leak = next((st for st in label.get("stages", []) if st["stage"] in ("leak", "recipe_v2")), None)
    boxes = sum(len(st.get("boxes", [])) for st in label.get("stages", []) if st["stage"] in ("leak", "cool_tint", "recipe_v2"))
    audit = label.get("audit", {})
    return {"scan": stem, "size": label.get("size", ""), "verdict": label.get("verdict", ""), "boxes": boxes,
            "gain": (leak["second_gain"] if leak["stage"] == "recipe_v2" else leak["param"]) if leak else "", "tile": leak.get("tile", "") if leak else "",
            "inside": " ".join(str(v) for v in audit.get("inside", [])), "outside": " ".join(str(v) for v in audit.get("outside", [])),
            "accepted": stem in accepted}


def export_accepted(path, stem, label, gain, accepted_dir, fixed_out, original_dir, fixed_dir):
    name = os.path.basename(path)
    if gain is None:
        shutil.copy2(path, os.path.join(accepted_dir, name))
        existing = os.path.join(fixed_dir, f"{stem}_fixed.jpg")
        if os.path.isfile(existing): shutil.copy2(existing, os.path.join(fixed_out, f"{stem}_fixed.jpg"))
        return label
    chosen = label_with_gain(label, gain)
    original = core.load(find_scan(original_dir, stem))
    replayed = replay_stages_quietly(original, chosen["stages"])
    chosen["audit"] = audit_record(*core.audit(original, replayed, label_audit_boxes(chosen["stages"], original.shape)))
    core.save(os.path.join(fixed_out, f"{stem}_fixed.jpg"), replayed)
    json.dump(chosen, open(os.path.join(accepted_dir, name), "w"), indent=2)
    return chosen


def export_labels(labels_dir, accepted, out_dir, original_dir=DEFAULT_ORIGINAL, fixed_dir=DEFAULT_FIXED):
    accepted_dir, rejected_dir, fixed_out = (os.path.join(out_dir, n) for n in ("labels", "rejected", "fixed"))
    for folder in (accepted_dir, rejected_dir, fixed_out): os.makedirs(folder, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(labels_dir, "*.boxes.json"))):
        stem = os.path.basename(path)[:-len(".boxes.json")]
        label = json.load(open(path))
        if stem in accepted:
            label = export_accepted(path, stem, label, accepted[stem], accepted_dir, fixed_out, original_dir, fixed_dir)
        else:
            shutil.copy2(path, os.path.join(rejected_dir, os.path.basename(path)))
        rows.append(manifest_row(label, stem, accepted))
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS); writer.writeheader(); writer.writerows(rows)
    return rows


def label_stems(labels_dir):
    return {os.path.basename(p)[:-len(".boxes.json")] for p in glob.glob(os.path.join(labels_dir, "*.boxes.json"))}


def unmatched_accepted(accepted, labels_dir):
    return sorted(set(accepted) - label_stems(labels_dir))


def recipe_stage_of(label):
    return next((st for st in label.get("stages", []) if st["stage"] == "recipe_v2"), None)


def clip_mask_fraction(original, boxes):
    union = np.zeros(original.shape[:2], bool)
    for b in boxes: union |= inpaint.residue_mask(original, b)
    return inpaint.mask_fraction(union)


def inpaint_status(fraction):
    if fraction == 0: return "skipped-empty"
    if fraction > inpaint.MAX_MASK_FRACTION: return "skipped-large"
    return "inpainted"


def inpaint_stage(boxes, method):
    return {"stage": "inpaint", "boxes": [list(b) for b in boxes], "ref": None, "param": inpaint.MAX_MASK_FRACTION,
            "method": method, "mask_mode": "clip", "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def inpaint_recipe_output(original, fixed, padded, method):
    applied, rgb = [], fixed
    for b in padded:
        rgb, info = inpaint.inpaint_residue(rgb, original, b, method=method, mask_mode="clip")
        if info["applied"]: applied.append(b)
    return rgb, applied


def inpaint_label(label, original_dir, stem, out_dir, method):
    stage = recipe_stage_of(label)
    if stage is None: return {"scan": stem, "mask_pct": "", "method": "", "status": "skipped-no-recipe"}
    original = core.load(find_scan(original_dir, stem))
    tile = core.resolve_tile(stage.get("tile"), original.shape)
    padded = recipe.padded_leak_boxes(stage["boxes"], tile, original.shape)
    fraction = clip_mask_fraction(original, padded); status = inpaint_status(fraction)
    row = {"scan": stem, "mask_pct": round(fraction * 100, 3), "method": "", "status": status}
    if status != "inpainted": return row
    fixed = replay_stages_quietly(original, [stage])
    with contextlib.redirect_stdout(io.StringIO()):
        rgb, applied = inpaint_recipe_output(original, fixed, padded, method)
    written = json.loads(json.dumps(label)); written["stages"].append(inpaint_stage(applied, method))
    core.save(os.path.join(out_dir, f"{stem}_inpainted.jpg"), rgb)
    write_json_atomic(os.path.join(out_dir, f"{stem}.boxes.json"), written)
    return dict(row, method=method)


def write_json_atomic(path, data):
    with open(path + ".tmp", "w") as f: json.dump(data, f, indent=2)
    os.replace(path + ".tmp", path)


def read_inpaint_summary(path):
    if not os.path.isfile(path): return []
    return list(csv.DictReader(open(path, newline="")))


def write_inpaint_summary(path, rows):
    with open(path + ".tmp", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INPAINT_SUMMARY_COLUMNS); writer.writeheader(); writer.writerows(rows)
    os.replace(path + ".tmp", path)


def finished_stems(rows, method):
    return {r["scan"] for r in rows if r["status"] in INPAINT_METHOD_FREE_STATUSES or (r["status"] == "inpainted" and r["method"] == method)}


def kept_lama_stems(rows, method, force):
    if force or method != "telea": return set()
    return {r["scan"] for r in rows if r["status"] == "inpainted" and r["method"] == "lama"}


def inpaint_dataset(dataset_dir, method="lama", original_dir=DEFAULT_ORIGINAL, report=print, force=False):
    inpaint.choose_method(method)
    out_dir = os.path.join(dataset_dir, "inpainted"); os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv"); rows = read_inpaint_summary(summary_path); done = finished_stems(rows, method); kept = kept_lama_stems(rows, method, force)
    for path in sorted(glob.glob(os.path.join(dataset_dir, "labels", "*.boxes.json"))):
        stem = os.path.basename(path)[:-len(".boxes.json")]
        if stem in kept: report(f"{stem}: kept-lama"); continue
        if stem in done: continue
        row = inpaint_label(json.load(open(path)), original_dir, stem, out_dir, method)
        rows = [r for r in rows if r["scan"] != stem] + [row]; write_inpaint_summary(summary_path, rows)
        report(f"{stem}: {row['status']}" + (f", mask {row['mask_pct']}%" if row["mask_pct"] != "" else ""))
    return rows


def gain_list(text):
    return [float(v) for v in text.split(",") if v.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m leakfix.batch"); sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grid"); g.add_argument("target"); g.add_argument("--out", default="work/grids")
    r = sub.add_parser("run"); r.add_argument("box_file", nargs="+"); r.add_argument("--original", default=DEFAULT_ORIGINAL)
    r.add_argument("--fixed", default=DEFAULT_FIXED); r.add_argument("--labels", default=DEFAULT_LABELS); r.add_argument("--compare", default=DEFAULT_COMPARE)
    r.add_argument("--variants", type=gain_list, default=[])
    s = sub.add_parser("status"); s.add_argument("boxes_dir", nargs="?", default="work/boxes"); s.add_argument("labels_dir", nargs="?", default=DEFAULT_LABELS); s.add_argument("--original", default=DEFAULT_ORIGINAL)
    e = sub.add_parser("export"); e.add_argument("--accepted", default="work/accepted.txt"); e.add_argument("--labels", default=DEFAULT_LABELS); e.add_argument("--out", default="dataset")
    e.add_argument("--original", default=DEFAULT_ORIGINAL); e.add_argument("--fixed", default=DEFAULT_FIXED)
    i = sub.add_parser("inpaint"); i.add_argument("--dataset", default="dataset"); i.add_argument("--method", choices=("lama", "telea"), default="lama")
    i.add_argument("--original", default=DEFAULT_ORIGINAL); i.add_argument("--force", action="store_true", help="let a telea run redo scans LaMa already filled")
    a = p.parse_args(argv)
    try:
        if a.command == "grid":
            targets = list(list_scans(a.target).values()) if os.path.isdir(a.target) else [a.target]
            for t in targets:
                out, W, H, step = write_grid(t, a.out); print(f"{os.path.basename(t)} {W}x{H} step {step} -> {out}")
        elif a.command == "run":
            for box_path in a.box_file:
                for line in run_box_file(box_path, a.original, a.fixed, a.labels, a.compare, a.variants): print(line)
        elif a.command == "status":
            print(format_status(status_rows(a.original, a.boxes_dir, a.labels_dir)))
        elif a.command == "export":
            accepted = read_accepted(a.accepted)
            missing = unmatched_accepted(accepted, a.labels)
            if missing: raise SystemExit(f"{len(missing)} name(s) in {a.accepted} match no label in {a.labels}: {', '.join(missing)}; fix them, nothing was exported")
            rows = export_labels(a.labels, accepted, a.out, a.original, a.fixed)
            kept = sum(r["accepted"] for r in rows); print(f"{kept} accepted -> {a.out}/labels, {len(rows) - kept} rejected -> {a.out}/rejected, manifest {a.out}/manifest.csv ({len(rows)} rows)")
        elif a.command == "inpaint":
            rows = inpaint_dataset(a.dataset, a.method, a.original, force=a.force)
            counts = {}
            for r in rows: counts[r["status"]] = counts.get(r["status"], 0) + 1
            print(f"{len(rows)} scan(s) in {a.dataset}/inpainted/summary.csv: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()

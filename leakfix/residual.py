import argparse, csv, glob, os, sys
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from . import core, compare, gemini

DEFAULT_TILE = 64
MIN_LEVEL = 1e-4
LEVELS = 255.0
P95 = 95
STRIP_TILES = 2
COOL_VALID_MAX_DE2000 = 10.0
LOW_BOX_COVERAGE_PCT = 95
DEFAULT_CSV = "residual.csv"
GEMINI_CSV_NAME = "residual.csv"
ORIGINAL_LABEL = "original"
RECIPE_LABEL = "recipe-v2"
ALL_BOXES = "all"
COOL_INVALID_TEXT = "n/a (neighbour differs)"
CSV_COLUMNS = ("scan", "version", "box", "warm_mean", "warm_p95", "warm_left_pct", "cool_mean", "cool_p95", "cool_valid", "one_sided", "box_covered_pct")


def box_region(image, box):
    x0, x1, y0, y1 = box
    return image[y0:y1, x0:x1]


def leak_levels(rgb, box, mode, tile):
    _, field, _ = core.tile_fix(rgb, box, mode=mode, tile=tile, gain=1.0)
    channel = 0 if mode == "warm" else 2
    return box_region(field[..., channel], box) * LEVELS


def residual(rgb, box, tile=DEFAULT_TILE):
    rgb = np.clip(rgb, MIN_LEVEL, 1); box = core.clamp_box(box, rgb.shape)
    warm, cool = leak_levels(rgb, box, "warm", tile), leak_levels(rgb, box, "cool", tile)
    return {"warm_mean": float(warm.mean()), "warm_p95": float(np.percentile(warm, P95)),
            "cool_mean": float(cool.mean()), "cool_p95": float(np.percentile(cool, P95))}


def is_vertical(box):
    x0, x1, y0, y1 = box
    return (y1 - y0) >= (x1 - x0)


def touches_edge(box, shape):
    H, W = shape[:2]; x0, x1, y0, y1 = box
    return x0 <= 0 or y0 <= 0 or x1 >= W or y1 >= H


def side_strip_boxes(box, shape, tile=DEFAULT_TILE):
    H, W = shape[:2]; x0, x1, y0, y1 = box; width = STRIP_TILES * core.fit_tile(int(tile), box)
    if is_vertical(box): candidates = [(max(0, x0 - width), x0, y0, y1), (x1, min(W, x1 + width), y0, y1)]
    else: candidates = [(x0, x1, max(0, y0 - width), y0), (x0, x1, y1, min(H, y1 + width))]
    return [c for c in candidates if c[1] > c[0] and c[3] > c[2]]


def median_lab(lab, regions, covered):
    pixels = [box_region(lab, r)[box_region(covered, r)] for r in regions]
    pixels = np.concatenate([p for p in pixels if len(p)]) if any(len(p) for p in pixels) else np.zeros((0, 3))
    return np.median(pixels, axis=0) if len(pixels) else None


def neighbour_difference(lab, box, shape, covered, tile=DEFAULT_TILE):
    inside, strips = median_lab(lab, [box], covered), median_lab(lab, side_strip_boxes(box, shape, tile), covered)
    if inside is None or strips is None: return None
    return float(deltaE_ciede2000(inside[None], strips[None])[0])


def cool_is_valid(difference):
    return difference is not None and difference < COOL_VALID_MAX_DE2000


def box_covered_percent(box, covered):
    return float(box_region(covered, box).mean() * 100)


def box_text(box):
    return ",".join(str(int(v)) for v in box)


def box_area(box):
    x0, x1, y0, y1 = box
    return (x1 - x0) * (y1 - y0)


def left_percent(version_warm, original_warm):
    if original_warm <= 0: return None
    return 100.0 * (version_warm / original_warm)


def box_rows(scan, label, rgb, boxes, covered, original_rows=None, tile=DEFAULT_TILE):
    rgb = np.clip(rgb, MIN_LEVEL, 1); lab = rgb2lab(rgb); rows = []
    for index, box in enumerate(boxes):
        stats = residual(rgb, box, tile)
        original_warm = original_rows[index]["warm_mean"] if original_rows else stats["warm_mean"]
        rows.append({"scan": scan, "version": label, "box": box_text(box), "area": box_area(box), **stats,
                     "warm_left_pct": 100.0 if original_rows is None else left_percent(stats["warm_mean"], original_warm),
                     "cool_valid": cool_is_valid(neighbour_difference(lab, box, rgb.shape, covered, tile)),
                     "one_sided": touches_edge(box, rgb.shape), "box_covered_pct": box_covered_percent(box, covered)})
    return rows


def area_weighted(rows, key):
    return float(sum(r[key] * r["area"] for r in rows) / sum(r["area"] for r in rows))


def all_boxes_row(rows, original_all=None):
    row = {"scan": rows[0]["scan"], "version": rows[0]["version"], "box": ALL_BOXES, "area": sum(r["area"] for r in rows)}
    row.update({key: area_weighted(rows, key) for key in ("warm_mean", "warm_p95", "cool_mean", "cool_p95", "box_covered_pct")})
    row["warm_left_pct"] = 100.0 if original_all is None else left_percent(row["warm_mean"], original_all["warm_mean"])
    row.update(cool_valid=all(r["cool_valid"] for r in rows), one_sided=any(r["one_sided"] for r in rows))
    return row


def with_all_row(rows, original_all=None):
    return rows + [all_boxes_row(rows, original_all)] if len(rows) > 1 else rows


def scan_level_row(rows):
    return rows[-1]


def residual_versions(original_path, versions, boxes, tile=DEFAULT_TILE, report=print):
    versions = compare.checked_versions(list(versions)) if versions else []
    original = core.load(original_path); boxes = compare.checked_boxes(boxes, original.shape)
    scan = compare.scan_name(original_path); everywhere = np.ones(original.shape[:2], bool)
    original_boxes = box_rows(scan, ORIGINAL_LABEL, original, boxes, everywhere, None, tile)
    original_rows = with_all_row(original_boxes)
    rows = list(original_rows)
    for label, path in versions:
        aligned, inliers, covered = compare.align_with_coverage(original, compare.load_version(label, path), label)
        if inliers < compare.RELIABLE_INLIERS: report(f"{label}: {inliers} RANSAC inliers  WARNING: alignment unreliable (under {compare.RELIABLE_INLIERS} inliers)")
        version_rows = with_all_row(box_rows(scan, label, aligned, boxes, covered, original_boxes, tile), scan_level_row(original_rows) if len(boxes) > 1 else None)
        for r in version_rows:
            if r["box_covered_pct"] < LOW_BOX_COVERAGE_PCT: report(f"{label} box {r['box']}: covers {r['box_covered_pct']:.1f} % of the box  WARNING: under {LOW_BOX_COVERAGE_PCT} %")
        rows += version_rows
    return rows


def format_number(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def cool_text(row):
    return format_number(row["cool_mean"]) if row["cool_valid"] else COOL_INVALID_TEXT


def table_text(rows):
    lines = [f"{'version':<28} {'box':<22} {'warm':>7} {'p95':>7} {'left%':>7} {'cool':>24} {'cool p95':>8} {'1-side':>6} {'cov%':>6}"]
    for r in rows:
        lines.append(f"{r['version']:<28} {r['box']:<22} {r['warm_mean']:>7.2f} {r['warm_p95']:>7.2f} {format_number(r['warm_left_pct'], 1):>7} "
                     f"{cool_text(r):>24} {r['cool_p95']:>8.2f} {str(r['one_sided']):>6} {r['box_covered_pct']:>6.1f}")
    return "\n".join(lines)


def check_csv_header(path):
    if compare.csv_is_new(path): return
    with open(path, newline="") as f: header = tuple(next(csv.reader(f), []))
    if header != CSV_COLUMNS: raise ValueError(f"{path} has a different header ({', '.join(header)}); use a new --csv or move it aside")


def csv_value(value):
    if value is None: return ""
    return round(value, 4) if isinstance(value, float) else value


def write_csv_rows(path, rows):
    check_csv_header(path)
    is_new = compare.csv_is_new(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new: writer.writeheader()
        for row in rows: writer.writerow({k: csv_value(v) for k, v in row.items()})


def run_scan(original_path, versions, boxes, csv_path, tile=DEFAULT_TILE, report=print):
    check_csv_header(csv_path)
    rows = residual_versions(original_path, versions, boxes, tile, report)
    write_csv_rows(csv_path, rows)
    report(table_text(rows))
    report(f"appended {len(rows)} row(s) to {csv_path}")
    return rows


def gemini_stems(gemini_dir, stems_file=None):
    if stems_file: return gemini.read_stems([], stems_file)
    return sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(gemini_dir, "*", "*.png"))})


def stem_versions(gemini_dir, stem, fixed_dir=gemini.FIXED_DIR):
    versions = gemini.stem_versions(gemini_dir, stem)
    recipe_path = os.path.join(fixed_dir, f"{stem}_fixed.jpg")
    return versions + [(RECIPE_LABEL, recipe_path)] if os.path.isfile(recipe_path) else versions


def median_and_mean(values):
    return (float(np.median(values)), float(np.mean(values))) if values else (None, None)


def version_summary(rows):
    per_version = {}
    for scan in dict.fromkeys(r["scan"] for r in rows):
        scan_rows = [r for r in rows if r["scan"] == scan]
        for label in dict.fromkeys(r["version"] for r in scan_rows):
            level = scan_level_row([r for r in scan_rows if r["version"] == label])
            if level["warm_left_pct"] is not None: per_version.setdefault(label, []).append(level["warm_left_pct"])
    return {label: (len(values),) + median_and_mean(values) for label, values in per_version.items()}


def summary_text(summary):
    lines = [f"{'version':<28} {'scans':>5} {'median left%':>12} {'mean left%':>10}"]
    for label, (n, median, mean) in summary.items():
        lines.append(f"{label:<28} {n:>5} {format_number(median, 1):>12} {format_number(mean, 1):>10}")
    return "\n".join(lines)


def run_gemini_dir(gemini_dir, stems_file=None, csv_path=None, original_dir=gemini.ORIGINAL_DIR, fixed_dir=gemini.FIXED_DIR,
                   dataset_dir=gemini.DATASET_DIR, tile=DEFAULT_TILE, report=print):
    csv_path = csv_path or os.path.join(gemini_dir, GEMINI_CSV_NAME); check_csv_header(csv_path)
    all_rows = []
    for stem in gemini_stems(gemini_dir, stems_file):
        label_path = gemini.label_path_for(stem, dataset_dir)
        if label_path is None: report(f"{stem}: no boxes.json in {dataset_dir}/labels or rejected, skipped"); continue
        try:
            rows = residual_versions(os.path.join(original_dir, f"{stem}.jpg"), stem_versions(gemini_dir, stem, fixed_dir),
                                     compare.read_label_boxes(label_path), tile, report)
        except (ValueError, OSError) as e:
            report(f"{stem}: skipped: {e}"); continue
        write_csv_rows(csv_path, rows); report(f"{stem}: {len(rows)} row(s)"); all_rows += rows
    report(summary_text(version_summary(all_rows)))
    report(f"appended {len(all_rows)} row(s) to {csv_path}")
    return all_rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="leakfix.residual", description="How much leak is left inside the leak boxes: tile-method warm / cool estimate at gain 1.0, per box, against the original.")
    p.add_argument("original", nargs="?", help="the raw scan the boxes were drawn on (single-scan mode)")
    p.add_argument("--version", action="append", type=compare.parse_version, default=[], metavar="LABEL=PATH", help="a version to measure (repeatable)")
    p.add_argument("--boxes", action="append", default=[], metavar="BOXES_JSON", help="<scan>.boxes.json: leak, cool_tint, exposure and recipe_v2 boxes (repeatable)")
    p.add_argument("--box", action="append", type=compare.parse_box, default=[], metavar="x0,x1,y0,y1", help="a leak box in full-res pixels (repeatable)")
    p.add_argument("--tile", type=int, default=DEFAULT_TILE, help="tile size in px for the estimate")
    p.add_argument("--csv", default=None, help=f"CSV to append to (default {DEFAULT_CSV}; with --gemini-dir, <gemini-dir>/{GEMINI_CSV_NAME})")
    p.add_argument("--gemini-dir", default=None, help="batch mode: every stem folder of a leakfix.gemini run, plus recipe-v2 = <fixed-dir>/<stem>_fixed.jpg")
    p.add_argument("--stems-file", default=None, help="with --gemini-dir: only these stems")
    p.add_argument("--original-dir", default=gemini.ORIGINAL_DIR)
    p.add_argument("--fixed-dir", default=gemini.FIXED_DIR)
    p.add_argument("--dataset", default=gemini.DATASET_DIR)
    a = p.parse_args(argv)
    try:
        if a.tile < core.MIN_TILE_PX: raise ValueError(f"--tile must be at least {core.MIN_TILE_PX}, got {a.tile}")
        if a.gemini_dir:
            if a.original or a.version or a.box or a.boxes: raise ValueError("--gemini-dir takes its scans, versions and boxes from the run folder; drop original / --version / --box / --boxes")
            run_gemini_dir(a.gemini_dir, a.stems_file, a.csv, a.original_dir, a.fixed_dir, a.dataset, a.tile)
        else:
            if not a.original: raise ValueError("give the original scan, or --gemini-dir")
            run_scan(a.original, a.version, compare.gathered_boxes(a), a.csv or DEFAULT_CSV, a.tile)
    except (ValueError, OSError) as e:
        sys.exit(f"leakfix.residual: {e}")


if __name__ == "__main__":
    main()

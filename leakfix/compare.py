import argparse, csv, json, os, re, sys
import numpy as np, cv2
from skimage.color import rgb2lab, deltaE_cie76, deltaE_ciede2000
from . import core

SIFT_FEATURES = 6000
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 4.0
MIN_MATCHES = 4
RELIABLE_INLIERS = 100
IDENTITY_CORNER_TOLERANCE_PX = 0.05
COVERAGE_ERODE_PX = 3
LOW_COVERAGE_PCT = 90
DEFAULT_MARGIN = 64
BORDER_PX = 12
DEFAULT_CSV = "compare.csv"
LEAK_STAGES = ("leak", "cool_tint", "exposure", "recipe_v2")
DE76_THRESHOLDS = (2.3, 5, 10)
DE2000_THRESHOLDS = (1, 2, 5)
DE2000_CHUNK_PX = 1_000_000
MAP_VMIN, MAP_VMAX = 0, 20
FIGURE_LONG_SIDE = 1200
FIGURE_COLUMN_INCHES = 5.0
FIGURE_DPI = 150
LEGEND_TEXT = "white dashed = leak box\ncyan dashed = margin\nbottom row = ΔE76 vs original\nscale 0–20, masked black"
CSV_COLUMNS = ("scan", "version", "inliers", "outside_pct", "de76_med", "de76_mean", "de76_gt2.3", "de76_gt5",
               "de76_gt10", "de2000_med", "de2000_mean", "de2000_gt1", "de2000_gt2", "de2000_gt5", "dL", "da", "db", "covered_pct")


def parse_box(text):
    try:
        x0, x1, y0, y1 = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"box must be x0,x1,y0,y1 (four integers), got '{text}'")
    if x1 <= x0 or y1 <= y0: raise argparse.ArgumentTypeError(f"box {text} must have x1 > x0 and y1 > y0")
    return (x0, x1, y0, y1)


def parse_version(text):
    label, separator, path = text.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"version must be <label>=<path>, got '{text}'")
    return label.strip(), path.strip()


def filename_safe(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "version"


def unique_union(boxes):
    return list(dict.fromkeys(tuple(int(v) for v in b) for b in boxes))


def label_leak_boxes(label):
    return unique_union(b for st in label.get("stages", []) if st.get("stage") in LEAK_STAGES for b in st.get("boxes", []))


def read_label_boxes(path):
    if not os.path.isfile(path): raise FileNotFoundError(f"boxes file not found: {path}")
    try:
        label = json.load(open(path))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e})")
    if not isinstance(label, dict) or not isinstance(label.get("stages"), list): raise ValueError(f"{path} has no 'stages' list")
    return label_leak_boxes(label)


def gray_u8(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def lowe_matches(original, version):
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kp_original, des_original = sift.detectAndCompute(gray_u8(original), None)
    kp_version, des_version = sift.detectAndCompute(gray_u8(version), None)
    if des_original is None or des_version is None or len(kp_original) < 2 or len(kp_version) < 2: return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pairs = cv2.BFMatcher().knnMatch(des_version, des_original, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < LOWE_RATIO * n.distance]
    return np.float32([kp_version[m.queryIdx].pt for m in good]).reshape(-1, 2), np.float32([kp_original[m.trainIdx].pt for m in good]).reshape(-1, 2)


def homography_to_original(original, version, label):
    src, dst = lowe_matches(original, version)
    if len(src) < MIN_MATCHES: raise ValueError(f"version '{label}': only {len(src)} SIFT matches with the original (need at least {MIN_MATCHES}); is it the same photo?")
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    if homography is None: raise ValueError(f"version '{label}': no homography found from {len(src)} matches")
    return homography, int(inliers.sum())


def corner_shift_px(homography, shape):
    H, W = shape[:2]
    corners = np.float64([[0, 0], [W, 0], [0, H], [W, H]]).reshape(-1, 1, 2)
    return float(np.abs(cv2.perspectiveTransform(corners, homography) - corners).max())


def is_identity_on(homography, version_shape, original_shape):
    return version_shape[:2] == original_shape[:2] and corner_shift_px(homography, original_shape) < IDENTITY_CORNER_TOLERANCE_PX


def eroded_coverage(coverage_u8, erode_px=COVERAGE_ERODE_PX):
    if erode_px <= 0: return coverage_u8 > 0
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    return cv2.erode(coverage_u8, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0


def coverage_mask(homography, version_shape, original_shape, erode_px=COVERAGE_ERODE_PX):
    H, W = original_shape[:2]
    warped_ones = cv2.warpPerspective(np.ones(version_shape[:2], np.uint8), homography, (W, H), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return eroded_coverage(warped_ones, erode_px)


def align_with_coverage(original, version, label):
    homography, inliers = homography_to_original(original, version, label)
    if is_identity_on(homography, version.shape, original.shape): return version.copy(), inliers, np.ones(original.shape[:2], bool)
    H, W = original.shape[:2]
    warped = cv2.warpPerspective(version, homography, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(warped, 0, 1), inliers, coverage_mask(homography, version.shape, original.shape)


def align_to_original(original, version, label):
    aligned, inliers, _ = align_with_coverage(original, version, label)
    return aligned, inliers


def padded_box(box, margin, shape):
    H, W = shape[:2]; x0, x1, y0, y1 = box
    return (max(0, x0 - margin), min(W, x1 + margin), max(0, y0 - margin), min(H, y1 + margin))


def outside_mask(shape, boxes, margin=DEFAULT_MARGIN, border=BORDER_PX):
    H, W = shape[:2]; mask = np.ones((H, W), bool)
    for box in boxes:
        x0, x1, y0, y1 = padded_box(box, margin, shape)
        mask[y0:y1, x0:x1] = False
    mask[:border] = False; mask[max(0, H - border):] = False; mask[:, :border] = False; mask[:, max(0, W - border):] = False
    return mask


def to_lab(rgb):
    return rgb2lab(np.clip(rgb, 0, 1))


def de2000_values(lab_original, lab_version):
    return np.concatenate([deltaE_ciede2000(lab_original[i:i + DE2000_CHUNK_PX], lab_version[i:i + DE2000_CHUNK_PX])
                           for i in range(0, len(lab_original), DE2000_CHUNK_PX)])


def percent_over(values, threshold):
    return float((values > threshold).mean() * 100)


def threshold_key(prefix, threshold):
    return f"{prefix}_gt{threshold:g}"


def covered_percent(mask, covered):
    return float((mask & covered).sum() / mask.sum() * 100)


def collateral_stats(lab_original, lab_version, mask, covered=None):
    if not mask.any(): raise ValueError("the boxes plus margin and border cover the whole frame; nothing is left outside to measure")
    covered = np.ones(mask.shape, bool) if covered is None else covered
    measured = mask & covered
    if not measured.any(): raise ValueError("the aligned version covers none of the outside area; nothing is left to measure")
    outside_original, outside_version = lab_original[measured], lab_version[measured]
    de76, de2000 = deltaE_cie76(outside_original, outside_version), de2000_values(outside_original, outside_version)
    signed = (outside_version - outside_original).mean(axis=0)
    stats = {"outside_pct": float(mask.mean() * 100), "de76_med": float(np.median(de76)), "de76_mean": float(de76.mean())}
    stats.update({threshold_key("de76", t): percent_over(de76, t) for t in DE76_THRESHOLDS})
    stats.update({"de2000_med": float(np.median(de2000)), "de2000_mean": float(de2000.mean())})
    stats.update({threshold_key("de2000", t): percent_over(de2000, t) for t in DE2000_THRESHOLDS})
    stats.update({"dL": float(signed[0]), "da": float(signed[1]), "db": float(signed[2]), "covered_pct": covered_percent(mask, covered)})
    return stats


def measure_collateral(original, aligned, boxes, margin=DEFAULT_MARGIN, covered=None):
    return collateral_stats(to_lab(original), to_lab(aligned), outside_mask(original.shape, boxes, margin), covered)


def de76_map(lab_original, lab_version):
    return deltaE_cie76(lab_original, lab_version)


def figure_scale(shape):
    return min(1.0, FIGURE_LONG_SIDE / max(shape[:2]))


def downscaled(image, scale):
    if scale >= 1: return image
    return cv2.resize(image.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def masked_map_preview(de_map, mask, scale):
    small_map = downscaled(np.where(mask, de_map, 0), scale)
    small_mask = downscaled(mask.astype(np.float32), scale) >= 0.5
    return np.ma.masked_where(~small_mask, small_map)


def csv_is_new(path):
    return not os.path.exists(path) or os.path.getsize(path) == 0


def check_csv_header(path):
    if csv_is_new(path): return
    with open(path, newline="") as f: header = tuple(next(csv.reader(f), []))
    if header != CSV_COLUMNS:
        missing = [c for c in CSV_COLUMNS if c not in header]
        raise ValueError(f"{path} has a different header ({'missing ' + ', '.join(missing) if missing else 'columns differ'}); it was written by an older "
                         f"leakfix.compare, so appending would misalign the rows. Use a new --csv or move the old file aside")


def write_csv_rows(path, rows):
    check_csv_header(path)
    is_new = csv_is_new(path)
    folder = os.path.dirname(os.path.abspath(path)); os.makedirs(folder, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new: writer.writeheader()
        for row in rows: writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def table_text(rows):
    header = f"{'version':<24} {'inliers':>7} {'out%':>6} {'dE76 med':>8} {'mean':>6} {'>2.3%':>6} {'>5%':>6} {'>10%':>6} {'dE00 med':>8} {'mean':>6} {'>1%':>6} {'>2%':>6} {'>5%':>6} {'dL':>6} {'da':>6} {'db':>6} {'cov%':>6}"
    lines = [header]
    for r in rows:
        lines.append(f"{r['version']:<24} {r['inliers']:>7} {r['outside_pct']:>6.1f} {r['de76_med']:>8.2f} {r['de76_mean']:>6.2f} {r['de76_gt2.3']:>6.1f} {r['de76_gt5']:>6.1f} {r['de76_gt10']:>6.1f} "
                     f"{r['de2000_med']:>8.2f} {r['de2000_mean']:>6.2f} {r['de2000_gt1']:>6.1f} {r['de2000_gt2']:>6.1f} {r['de2000_gt5']:>6.1f} {r['dL']:>6.2f} {r['da']:>6.2f} {r['db']:>6.2f} {r['covered_pct']:>6.1f}")
    return "\n".join(lines)


def draw_box_outlines(ax, boxes, margin, shape):
    from matplotlib.patches import Rectangle
    for box in boxes:
        x0, x1, y0, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linestyle="--", linewidth=1.2))
        px0, px1, py0, py1 = padded_box(box, margin, shape)
        ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fill=False, edgecolor="cyan", linestyle="--", linewidth=1.0))


def show_image(ax, image, shape, title):
    H, W = shape[:2]
    shown = ax.imshow(image, extent=(0, W, H, 0), interpolation="nearest")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    return shown


def render_figure(path, shape, boxes, margin, original_preview, panels):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import colormaps
    H, W = shape[:2]; columns = 1 + len(panels)
    fig = Figure(figsize=(FIGURE_COLUMN_INCHES * columns, 2 * FIGURE_COLUMN_INCHES * H / W + 1.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, columns, squeeze=False)
    show_image(axes[0, 0], original_preview, shape, "original"); draw_box_outlines(axes[0, 0], boxes, margin, shape)
    axes[1, 0].axis("off"); axes[1, 0].text(0.02, 0.5, LEGEND_TEXT, transform=axes[1, 0].transAxes, fontsize=10, va="center")
    cmap = colormaps["inferno"].with_extremes(bad="black")
    shown_map = None
    for column, panel in enumerate(panels, start=1):
        show_image(axes[0, column], panel["preview"], shape, panel["label"]); draw_box_outlines(axes[0, column], boxes, margin, shape)
        shown_map = axes[1, column].imshow(panel["map"], extent=(0, W, H, 0), cmap=cmap, vmin=MAP_VMIN, vmax=MAP_VMAX, interpolation="nearest")
        axes[1, column].set_title(f"{panel['label']}: ΔE76 median {panel['de76_med']:.1f} outside boxes", fontsize=9)
        axes[1, column].set_xticks([]); axes[1, column].set_yticks([])
    fig.colorbar(shown_map, ax=axes[1, 1:].tolist(), label="ΔE76", shrink=0.9)
    fig.savefig(path, dpi=FIGURE_DPI, pil_kwargs={"quality": 92})
    return path


def scan_name(original_path):
    return os.path.splitext(os.path.basename(original_path))[0]


def checked_versions(versions):
    if not versions: raise ValueError("give at least one --version <label>=<path>")
    labels = [label for label, _ in versions]; safe = [filename_safe(label) for label in labels]
    if len(set(safe)) < len(safe): raise ValueError(f"version labels must be distinct after making them filename-safe: {', '.join(labels)}")
    for label, path in versions:
        if not os.path.isfile(path): raise FileNotFoundError(f"version '{label}': file not found: {path}")
    return versions


def checked_boxes(boxes, shape):
    if not boxes: raise ValueError("give at least one leak box (--boxes <scan>.boxes.json with leak stages, or --box x0,x1,y0,y1)")
    return [core.clamp_box(b, shape) for b in unique_union(boxes)]


def load_version(label, path):
    try:
        return core.load(path)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"version '{label}': {e}")


def compare_versions(original_path, versions, boxes, margin=DEFAULT_MARGIN, csv_path=DEFAULT_CSV, out_dir=".", report=print):
    versions = checked_versions(list(versions))
    original = core.load(original_path); boxes = checked_boxes(boxes, original.shape)
    check_csv_header(csv_path)
    scan = scan_name(original_path); os.makedirs(out_dir, exist_ok=True)
    lab_original = to_lab(original); mask = outside_mask(original.shape, boxes, margin)
    scale = figure_scale(original.shape); rows, panels = [], []
    for label, path in versions:
        aligned, inliers, covered = align_with_coverage(original, load_version(label, path), label)
        report(f"{label}: {inliers} RANSAC inliers" + ("  WARNING: alignment unreliable (under 100 inliers)" if inliers < RELIABLE_INLIERS else ""))
        core.save(os.path.join(out_dir, f"{scan}_{filename_safe(label)}_aligned.jpg"), aligned)
        lab_aligned = to_lab(aligned)
        row = {"scan": scan, "version": label, "inliers": inliers, **collateral_stats(lab_original, lab_aligned, mask, covered)}
        report(f"{label}: covers {row['covered_pct']:.1f} % of the outside area" + (f"  WARNING: under {LOW_COVERAGE_PCT} %, the version is cropped or shifted" if row["covered_pct"] < LOW_COVERAGE_PCT else ""))
        rows.append(row)
        panels.append({"label": label, "de76_med": row["de76_med"], "preview": np.clip(downscaled(aligned, scale), 0, 1),
                       "map": masked_map_preview(de76_map(lab_original, lab_aligned), mask & covered, scale)})
    figure_path = render_figure(os.path.join(out_dir, f"{scan}_compare.jpg"), original.shape, boxes, margin, np.clip(downscaled(original, scale), 0, 1), panels)
    write_csv_rows(csv_path, rows)
    report(table_text(rows))
    report(f"appended {len(rows)} row(s) to {csv_path}; figure {figure_path}")
    return rows, figure_path


def gathered_boxes(a):
    boxes = list(a.box or [])
    for path in a.boxes or []: boxes += read_label_boxes(path)
    return boxes


def main(argv=None):
    p = argparse.ArgumentParser(prog="leakfix-compare", description="Measure how much each regenerated version changed the scan outside the leak boxes (ΔE76 / ΔE2000 after SIFT alignment).")
    p.add_argument("original", help="the raw scan the boxes were drawn on")
    p.add_argument("--version", action="append", type=parse_version, default=[], metavar="LABEL=PATH", help="a version to compare (repeatable, at least one)")
    p.add_argument("--boxes", action="append", default=[], metavar="BOXES_JSON", help="<scan>.boxes.json: union of leak, cool_tint, exposure and recipe_v2 boxes (repeatable)")
    p.add_argument("--box", action="append", type=parse_box, default=[], metavar="x0,x1,y0,y1", help="a leak box in full-res pixels (repeatable)")
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="pixels added around every leak box before measuring outside")
    p.add_argument("--csv", default=DEFAULT_CSV, help="CSV to append rows to (header written only when new)")
    p.add_argument("-o", "--out", default=".", help="folder for the figure and the aligned versions")
    a = p.parse_args(argv)
    try:
        if a.margin < 0: raise ValueError(f"--margin must be 0 or more, got {a.margin}")
        compare_versions(a.original, a.version, gathered_boxes(a), a.margin, a.csv, a.out)
    except (ValueError, OSError) as e:
        sys.exit(f"leakfix.compare: {e}")


if __name__ == "__main__":
    main()

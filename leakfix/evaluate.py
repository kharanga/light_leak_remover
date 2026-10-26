import argparse, csv, json, os
import numpy as np, cv2
from scipy.ndimage import gaussian_filter
from . import core

RESIDUAL_STAGES = ("leak",)
EDITED_STAGES = ("leak", "cool_tint", "exposure", "hsv")
GRAIN_SIGMA = 3.0
SHADOW_L_MAX = 20.0
EDGE_GAP_PX = 50
NEIGHBOUR_PX = 32
CSV_COLUMNS = ("scan", "arm", "residual_rms", "residual_dc", "de_median", "de_pct_over_5", "de_pct_over_10",
               "dL_mean", "da_mean", "db_mean", "black_point", "shadow_a", "shadow_b", "chroma", "grain",
               "black_point_delta", "shadow_a_delta", "shadow_b_delta", "chroma_delta", "grain_delta", "aligned_inliers")


def to_lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB).astype(np.float64)


def luma_levels(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) * 255


def box_mask(shape, boxes, margin=0):
    H, W = shape[:2]; mask = np.zeros((H, W), bool)
    for x0, x1, y0, y1 in boxes:
        mask[max(0, int(y0) - margin):min(H, int(y1) + margin), max(0, int(x0) - margin):min(W, int(x1) + margin)] = True
    return mask


def is_vertical(box):
    x0, x1, y0, y1 = box
    return (y1 - y0) >= (x1 - x0)


def chroma_profile(lab, box, neighbour_px=NEIGHBOUR_PX):
    x0, x1, y0, y1 = [int(v) for v in box]
    H, W = lab.shape[:2]
    ab = lab[..., 1:3]
    if is_vertical(box):
        inside = ab[y0:y1, x0:x1].mean(0)
        before, after = ab[y0:y1, max(0, x0 - neighbour_px):x0], ab[y0:y1, x1:min(W, x1 + neighbour_px)]
        start, stop = x0, x1
    else:
        inside = ab[y0:y1, x0:x1].mean(1)
        before, after = ab[max(0, y0 - neighbour_px):y0, x0:x1], ab[y1:min(H, y1 + neighbour_px), x0:x1]
        start, stop = y0, y1
    before_width, after_width = before.shape[1 if is_vertical(box) else 0], after.shape[1 if is_vertical(box) else 0]
    before_mean = before.reshape(-1, 2).mean(0) if before_width else None
    after_mean = after.reshape(-1, 2).mean(0) if after_width else None
    if before_mean is None and after_mean is None: before_mean = after_mean = inside.mean(0)
    if before_mean is None: before_mean = after_mean
    if after_mean is None: after_mean = before_mean
    positions = start + 0.5 + np.arange(stop - start)
    before_pos, after_pos = start - max(before_width, 1) / 2, stop + max(after_width, 1) / 2
    return positions, inside, (before_pos, before_mean), (after_pos, after_mean)


def neighbour_line(positions, before, after):
    (p0, v0), (p1, v1) = before, after
    t = ((positions - p0) / (p1 - p0))[:, None]
    return v0[None] * (1 - t) + v1[None] * t


def profile_rms(profile, line):
    return float(np.sqrt(((profile - line) ** 2).sum(1).mean()))


def strongest_edge(profile, gap_px=EDGE_GAP_PX):
    n = len(profile)
    gap = min(gap_px, n - 1)
    if gap < 1: return 0.0
    return float(np.sqrt(((profile[gap:] - profile[:-gap]) ** 2).sum(1)).max())


def residual_in_box(lab, box, neighbour_px=NEIGHBOUR_PX, gap_px=EDGE_GAP_PX):
    positions, profile, before, after = chroma_profile(lab, box, neighbour_px)
    return {"rms": profile_rms(profile, neighbour_line(positions, before, after)), "dc": strongest_edge(profile, gap_px), "columns": len(positions)}


def residual(rgb, boxes, neighbour_px=NEIGHBOUR_PX, gap_px=EDGE_GAP_PX):
    lab = to_lab(rgb)
    per_box = [residual_in_box(lab, b, neighbour_px, gap_px) for b in boxes]
    if not per_box: return {"rms": 0.0, "dc": 0.0, "boxes": []}
    weights = np.array([p["columns"] for p in per_box], float)
    rms = float(np.sqrt((np.array([p["rms"] ** 2 for p in per_box]) * weights).sum() / weights.sum()))
    return {"rms": rms, "dc": max(p["dc"] for p in per_box), "boxes": per_box}


def film_properties(rgb, mask):
    lab = to_lab(rgb); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    shadows = mask & (L < SHADOW_L_MAX)
    high_frequency = L - gaussian_filter(L, GRAIN_SIGMA)
    return {"black_point": float(np.percentile(luma_levels(rgb)[mask], 1)),
            "shadow_a": float(a[shadows].mean()) if shadows.any() else float("nan"),
            "shadow_b": float(b[shadows].mean()) if shadows.any() else float("nan"),
            "chroma": float(np.hypot(a, b)[mask].mean()),
            "grain": float(high_frequency[mask].std())}


def collateral(original, result, boxes, margin=80):
    mask = ~box_mask(original.shape, boxes, margin)
    lab0, lab1 = to_lab(original), to_lab(result)
    diff = (lab1 - lab0)[mask]
    de = np.sqrt((diff ** 2).sum(1))
    before, after = film_properties(original, mask), film_properties(result, mask)
    out = {"de_median": float(np.median(de)), "de_pct_over_5": float((de > 5).mean() * 100), "de_pct_over_10": float((de > 10).mean() * 100),
           "dL_mean": float(diff[:, 0].mean()), "da_mean": float(diff[:, 1].mean()), "db_mean": float(diff[:, 2].mean()), "pixels": int(mask.sum())}
    for key in before: out[key], out[key + "_delta"] = after[key], after[key] - before[key]
    return out


def align(reference, moving, min_matches=12):
    sift = cv2.SIFT_create()
    ref_u8, mov_u8 = (np.clip(reference, 0, 1) * 255).astype(np.uint8), (np.clip(moving, 0, 1) * 255).astype(np.uint8)
    kp_ref, des_ref = sift.detectAndCompute(cv2.cvtColor(ref_u8, cv2.COLOR_RGB2GRAY), None)
    kp_mov, des_mov = sift.detectAndCompute(cv2.cvtColor(mov_u8, cv2.COLOR_RGB2GRAY), None)
    if des_ref is None or des_mov is None: return moving, 0
    pairs = cv2.BFMatcher().knnMatch(des_mov, des_ref, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_matches: return moving, 0
    src = np.float32([kp_mov[m.queryIdx].pt for m in good]); dst = np.float32([kp_ref[m.trainIdx].pt for m in good])
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None: return moving, 0
    H, W = reference.shape[:2]
    warped = cv2.warpPerspective(moving, homography, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped, int(inliers.sum())


def stage_boxes(log, stages):
    return [tuple(int(v) for v in b) for st in log["stages"] if st["stage"] in stages for b in st["boxes"]]


def evaluate_arm(scan, arm, original, result, leak_boxes, edited_boxes, margin, inliers):
    res = residual(result, leak_boxes)
    col = collateral(original, result, edited_boxes, margin)
    row = {"scan": scan, "arm": arm, "residual_rms": res["rms"], "residual_dc": res["dc"], "aligned_inliers": inliers}
    row.update({k: col[k] for k in CSV_COLUMNS if k in col})
    return row


def arm_path(results_dir, stem, arm):
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidate = os.path.join(results_dir, f"{stem}_{arm}{ext}")
        if os.path.isfile(candidate): return candidate
    return None


def evaluate_scan(image_path, log_path, results_dir=None, arms=(), align_arms=(), margin=80):
    log = json.load(open(log_path))
    original = core.load(image_path)
    leak_boxes, edited_boxes = stage_boxes(log, RESIDUAL_STAGES), stage_boxes(log, EDITED_STAGES)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    rows = [evaluate_arm(stem, "original", original, original, leak_boxes, edited_boxes, margin, 0)]
    for arm in arms:
        path = arm_path(results_dir or os.path.dirname(image_path) or ".", stem, arm)
        if path is None: continue
        result, inliers = core.load(path), 0
        if arm in align_arms: result, inliers = align(original, result)
        rows.append(evaluate_arm(stem, arm, original, result, leak_boxes, edited_boxes, margin, inliers))
    return rows


def evaluate_folder(folder, labels_dir=None, results_dir=None, arms=(), align_arms=(), margin=80):
    labels_dir = labels_dir or folder
    rows = []
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".boxes.json"): continue
        log_path = os.path.join(labels_dir, name)
        image_path = os.path.join(folder, json.load(open(log_path))["image"])
        if not os.path.isfile(image_path): continue
        rows.extend(evaluate_scan(image_path, log_path, results_dir, arms, align_arms, margin))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows: writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="a folder of scans with their .boxes.json, or one scan image")
    p.add_argument("--boxes", default=None, help="boxes.json for a single scan (default: <scan>.boxes.json beside it)")
    p.add_argument("--labels", default=None, help="folder holding the .boxes.json files (default: the scan folder)")
    p.add_argument("--results", default=None, help="folder holding <scan>_<arm>.jpg results (default: the scan folder)")
    p.add_argument("--arms", default="fixed", help="comma-separated arm suffixes to look for")
    p.add_argument("--align", default="", help="comma-separated arms to SIFT+RANSAC align to the original first")
    p.add_argument("--margin", type=int, default=80, help="pixels added around the edited boxes before measuring collateral")
    p.add_argument("-o", "--out", default=None, help="CSV path (default: print rows)")
    a = p.parse_args()
    arms = tuple(s for s in a.arms.split(",") if s); align_arms = tuple(s for s in a.align.split(",") if s)
    if os.path.isdir(a.target):
        rows = evaluate_folder(a.target, a.labels, a.results, arms, align_arms, a.margin)
    else:
        log_path = a.boxes or os.path.splitext(a.target)[0] + ".boxes.json"
        rows = evaluate_scan(a.target, log_path, a.results, arms, align_arms, a.margin)
    if a.out:
        write_csv(rows, a.out); print(f"wrote {len(rows)} row(s) to {a.out}")
    else:
        for row in rows: print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in CSV_COLUMNS}))


if __name__ == "__main__":
    main()

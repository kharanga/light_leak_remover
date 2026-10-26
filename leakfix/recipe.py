import numpy as np, cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt
from . import core, leak_map

FIRST_GAIN = 1.3
SECOND_GAIN = 1.3
TINT_GAIN = 1.0
GENTLE_TINT_GAIN = 0.5
FULL_TINT_SHARE = 0.10
CYAN_MARGIN = 0.02
CYAN_REF_TILES = 3
MIN_REFERENCE_CORR = 0.6
MAX_REFERENCE_OVERLAP = 0.2
EXPOSURE_MAX_STOPS = 1.0
HSV_STRENGTH = 0.45
BLACK_POINT_PERCENTILE = 1
FADE_SCALE_FRACTION = 0.3
FADE_ZERO_BELOW = 0.15


def pad_box(box, pad, shape):
    H, W = shape[:2]; x0, x1, y0, y1 = box
    return (max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad), min(H, y1 + pad))


def padded_leak_boxes(boxes, tile, shape):
    return [core.clamp_box(pad_box(core.clamp_box(b, shape), 2 * tile, shape), shape) for b in boxes]


def luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def box_mask(shape, boxes):
    mask = np.zeros(shape[:2], bool)
    for x0, x1, y0, y1 in boxes: mask[y0:y1, x0:x1] = True
    return mask


def tiles_in_boxes(grid, boxes, tile):
    inside = np.zeros_like(grid, bool)
    for x0, x1, y0, y1 in boxes: inside[y0 // tile:-(-y1 // tile), x0 // tile:-(-x1 // tile)] = True
    return inside


def excess_grid(rgb, tile):
    return leak_map.excess_above_background(leak_map.tile_means(leak_map.warm_excess_map(rgb), tile))


def upsampled_tile_alpha(weight, tile, shape):
    H, W = shape[:2]; ny, nx = weight.shape
    alpha = cv2.resize(weight.astype(np.float32), (nx * tile, ny * tile), interpolation=cv2.INTER_LINEAR)[:H, :W]
    return np.clip(gaussian_filter(alpha, tile * 0.5), 0, 1)


def coolness_grid(rgb, tile):
    dom = np.clip(rgb, 0, 1) ** (1 / 2.2); coolness = dom[..., 2] - dom[..., 0]
    return leak_map.tile_means(coolness, tile)


def cyan_tiles(rgb, boxes, tile, ref_tiles=CYAN_REF_TILES, margin=CYAN_MARGIN):
    grid = coolness_grid(rgb, tile); ny, nx = grid.shape
    inside = tiles_in_boxes(grid, boxes, tile); cyan = np.zeros_like(inside)
    for ty in range(ny):
        cols = np.where(inside[ty])[0]
        if not len(cols): continue
        side_cols = list(range(cols.min() - ref_tiles, cols.min())) + list(range(cols.max() + 1, cols.max() + 1 + ref_tiles))
        neighbours = [tx for tx in side_cols if 0 <= tx < nx and not inside[ty, tx]]
        if not neighbours: continue
        cyan[ty, cols] = grid[ty, cols] > np.median(grid[ty, neighbours]) + margin
    return cyan, inside


def detint_every_box(rgb, boxes, gain, tile):
    out = rgb.copy()
    for b in boxes: out, _, _ = core.detint(out, b, gain=gain, second_pass=False, tile=tile)
    return out


def gated_blue_tint(rgb, boxes, tile, tint_gain=TINT_GAIN, gentle_gain=GENTLE_TINT_GAIN, full_share=FULL_TINT_SHARE):
    cyan, inside = cyan_tiles(rgb, boxes, tile); share = float(cyan.sum() / max(inside.sum(), 1))
    if share < full_share:
        return detint_every_box(rgb, boxes, gentle_gain, tile), {"mode": "gentle", "gain": gentle_gain, "share": share}
    full = detint_every_box(rgb, boxes, tint_gain, tile)
    alpha = upsampled_tile_alpha(cyan, tile, rgb.shape) * box_mask(rgb.shape, boxes)
    return rgb + alpha[..., None] * (full - rgb), {"mode": "cyan", "gain": tint_gain, "share": share}


def row_profile(rgb, box):
    x0, x1, y0, y1 = box
    return np.median(luminance(rgb[y0:y1, x0:x1]), axis=1)


def profile_correlation(reference, edge):
    if reference.std() < 1e-4 or edge.std() < 1e-4: return 1.0 if abs(reference.mean() - edge.mean()) < 0.1 else 0.0
    return float(np.corrcoef(reference, edge)[0, 1])


def side_strips(box, tile, width):
    x0, x1, y0, y1 = box
    return {"left": ((max(0, x0 - 2 * tile), x0, y0, y1), (x0, min(x1, x0 + tile), y0, y1)),
            "right": ((x1, min(width, x1 + 2 * tile), y0, y1), (max(x0, x1 - tile), x1, y0, y1))}


def matching_reference(rgb, box, tile, boxes, min_corr=MIN_REFERENCE_CORR):
    occupied = box_mask(rgb.shape, boxes); best, best_corr = None, min_corr
    for side, (ref, edge) in side_strips(box, tile, rgb.shape[1]).items():
        if ref[1] - ref[0] < tile or occupied[ref[2]:ref[3], ref[0]:ref[1]].mean() > MAX_REFERENCE_OVERLAP: continue
        corr = profile_correlation(row_profile(rgb, ref), row_profile(rgb, edge))
        if corr > best_corr: best, best_corr = (ref, side), corr
    return best


def matched_exposure(rgb, boxes, tile, max_stops=EXPOSURE_MAX_STOPS):
    out = rgb.copy(); per_box = []
    for b in boxes:
        found = matching_reference(out, b, tile, boxes)
        if not found: per_box.append({"side": "skip", "lift": 0.0}); continue
        ref, side = found
        out, lift = core.exposure_match(out, b, ref, max_stops=max_stops); per_box.append({"side": side, "lift": lift})
    return out, per_box


def hsv_in_padded_boxes(rgb, boxes, tile, strength=HSV_STRENGTH):
    for b in boxes: rgb, _ = core.hsv_neutralize(rgb, pad_box(b, tile, rgb.shape), strength=strength)
    return rgb


def black_point_floor(original, rgb, boxes, percentile=BLACK_POINT_PERCENTILE):
    outside = ~box_mask(original.shape, boxes); lum_before = luminance(original)
    bp = float(np.percentile(lum_before[outside], percentile)) if outside.any() else float(np.percentile(lum_before, percentile))
    lum_after = luminance(rgb); floor = np.minimum(lum_before, bp)
    keep = np.where(lum_after < floor, np.clip((lum_before - floor) / np.maximum(lum_before - lum_after, 1e-6), 0, 1), 1.0)
    return original + keep[..., None] * (rgb - original), bp


def limit_mask(shape, boxes, taper):
    return np.clip(distance_transform_edt(box_mask(shape, boxes)) / taper, 0, 1)


def scale_from_peak(grid, allowed, scale_fraction):
    return float(max(scale_fraction * grid[allowed].max(), 1e-6))


def fade_scale(original, boxes, tile, scale_fraction=FADE_SCALE_FRACTION):
    grid = excess_grid(original, tile)
    return scale_from_peak(grid, tiles_in_boxes(grid, boxes, tile), scale_fraction)


def tile_fade_alpha(original, boxes, tile, scale_fraction=FADE_SCALE_FRACTION, zero_below=FADE_ZERO_BELOW):
    grid = excess_grid(original, tile); allowed = tiles_in_boxes(grid, boxes, tile)
    scale = scale_from_peak(grid, allowed, scale_fraction)
    weight = np.where(allowed, np.clip(grid / scale, 0, 1), 0); weight[weight < zero_below] = 0
    return upsampled_tile_alpha(weight, tile, original.shape) * limit_mask(original.shape, boxes, tile)


def run_recipe(original, leak_boxes, second_gain=SECOND_GAIN, tint_gain=TINT_GAIN, tile=None):
    if not leak_boxes: raise ValueError("the recipe needs at least one leak box")
    tile = core.resolve_tile(tile, original.shape); boxes = padded_leak_boxes(leak_boxes, tile, original.shape)
    rgb = original.copy()
    for b in boxes: rgb, _, _ = core.deleak(rgb, b, gain=FIRST_GAIN, second_pass=False, tile=tile)
    for b in boxes: rgb, _, _ = core.deleak(rgb, b, gain=second_gain, second_pass=False, tile=tile)
    rgb, tint = gated_blue_tint(rgb, boxes, tile, tint_gain=tint_gain)
    rgb, exposure = matched_exposure(rgb, boxes, tile)
    rgb = hsv_in_padded_boxes(rgb, boxes, tile)
    rgb, bp = black_point_floor(original, rgb, boxes)
    alpha = tile_fade_alpha(original, boxes, tile)
    final = original + alpha[..., None] * (rgb - original)
    return final, {"boxes": boxes, "tile": tile, "tint": tint, "exposure": exposure, "bp": bp}


def describe(info):
    tint = info["tint"]; exposure = ", ".join("skip" if e["side"] == "skip" else f"{e['side']} {e['lift']:+.2f}" for e in info["exposure"])
    return f"tile {info['tile']} | tint {tint['mode']} {tint['gain']:g} (cyan {tint['share']:.0%}) | exposure {exposure} | black point {info['bp'] * 255:.0f}"

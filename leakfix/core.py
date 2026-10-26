import os, numpy as np, cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d

GAMMA = 2.2
TILE_AUTO_THRESHOLDS = ((700, 64), (300, 32), (0, 16))
EXPOSURE_FEATHER_PX = 8
MIN_BOX_PX = 8
MIN_TILE_PX = 8


def clamp_box(box, shape):
    H, W = shape[:2]
    x0, x1, y0, y1 = [int(v) for v in box]
    clamped = (min(max(0, x0), W), min(max(0, x1), W), min(max(0, y0), H), min(max(0, y1), H))
    if clamped[1] - clamped[0] < MIN_BOX_PX or clamped[3] - clamped[2] < MIN_BOX_PX:
        raise ValueError(f"box {(x0, x1, y0, y1)} is empty or narrower than {MIN_BOX_PX} px after clamping to the {W}x{H} frame")
    return clamped


def fit_tile(tile, box):
    x0, x1, y0, y1 = box
    while tile > MIN_TILE_PX and min(x1 - x0, y1 - y0) < tile:
        tile //= 2
    return max(MIN_TILE_PX, tile)


def auto_tile(shape):
    short_side = min(shape[0], shape[1])
    return next(tile for threshold, tile in TILE_AUTO_THRESHOLDS if short_side >= threshold)


def resolve_tile(tile, shape):
    return int(tile) if tile else auto_tile(shape)


def confinement_mask(shape, box, feather_px):
    H, W = shape[:2]
    x0, x1, y0, y1 = [int(v) for v in box]
    x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0: return np.zeros((H, W))
    ramp = lambda n: np.minimum(1, (np.arange(n) + 0.5) / feather_px) if feather_px > 0 else np.ones(n)
    across = np.minimum(ramp(x1 - x0), ramp(x1 - x0)[::-1])
    down = np.minimum(ramp(y1 - y0), ramp(y1 - y0)[::-1])
    mask = np.zeros((H, W)); mask[y0:y1, x0:x1] = down[:, None] * across[None, :]
    return mask


def apply_inside_box(rgb, corrected, mask):
    out = rgb.copy(); inside = mask.reshape(mask.shape[:2]) > 0
    out[inside] = corrected[inside]
    return out


def tile_fix(rgb, box, mode="warm", tile=64, gain=1.3, pct=2.0, ref_tiles=2, tone_weight=True, force_dir=None):
    H, W, _ = rgb.shape
    box = clamp_box(box, rgb.shape); tile = fit_tile(int(tile), box)
    dom = rgb ** (1 / GAMMA)
    den = gaussian_filter(dom, (2, 2, 0))
    R, G, B = den[..., 0], den[..., 1], den[..., 2]
    feats = np.stack([R - B, G - B, B], -1) if mode == "warm" else np.stack([B - R, G - R, R], -1)

    ny, nx = int(np.ceil(H / tile)), int(np.ceil(W / tile))
    F = np.zeros((ny, nx, 3))
    for ty in range(ny):
        for tx in range(nx):
            F[ty, tx] = np.percentile(feats[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile].reshape(-1, 3), pct, axis=0)

    x0, x1, y0, y1 = [int(v) for v in box]
    tx0, tx1 = x0 // tile, int(np.ceil(x1 / tile)); ty0, ty1 = y0 // tile, int(np.ceil(y1 / tile))
    vertical = (force_dir == "V") if force_dir else ((y1 - y0) >= (x1 - x0))
    E = np.zeros_like(F)
    if vertical:
        for ty in range(ty0, ty1):
            L = F[ty, max(0, tx0 - ref_tiles):tx0]; Rr = F[ty, tx1:tx1 + ref_tiles]
            if len(L) == 0 and len(Rr) == 0: continue
            lv = L.mean(0) if len(L) else Rr.mean(0); rv = Rr.mean(0) if len(Rr) else L.mean(0)
            n = tx1 - tx0; base = np.linspace(0, 1, n)[:, None] * (rv - lv)[None] + lv[None]
            E[ty, tx0:tx1] = np.clip(F[ty, tx0:tx1] - base, 0, None)
    else:
        for tx in range(tx0, tx1):
            T = F[max(0, ty0 - ref_tiles):ty0, tx]; Bt = F[ty1:ty1 + ref_tiles, tx]
            if len(T) == 0 and len(Bt) == 0: continue
            tv = T.mean(0) if len(T) else Bt.mean(0); bv = Bt.mean(0) if len(Bt) else T.mean(0)
            n = ty1 - ty0; base = np.linspace(0, 1, n)[:, None] * (bv - tv)[None] + tv[None]
            E[ty0:ty1, tx] = np.clip(F[ty0:ty1, tx] - base, 0, None)

    k = np.unravel_index(np.argmax(E[..., 0]), E.shape[:2])
    af = float(np.clip(E[k][2] / (E[k][0] + E[k][2] + 1e-6), 0, 0.5))
    main = E[..., 0] / (1 - af); anc = af * main; sec = E[..., 1] + anc
    leak = np.stack([main, sec, anc], -1) if mode == "warm" else np.stack([anc, sec, main], -1)
    leak = gaussian_filter(leak, (1, 1, 0))
    field = cv2.resize(leak, (nx * tile, ny * tile), interpolation=cv2.INTER_CUBIC)[:H, :W]
    field = gaussian_filter(np.clip(field, 0, None), (tile * 0.5, tile * 0.5, 0))

    if tone_weight:
        Lm = 0.299 * dom[..., 0] + 0.587 * dom[..., 1] + 0.114 * dom[..., 2]
        w = np.clip(1.15 * np.sin(np.pi * np.clip((Lm - 0.05) / 0.9, 0, 1)) + 0.15, 0.15, 1)[..., None]
    else:
        w = 1.0
    confine = confinement_mask(rgb.shape, box, min(8, tile // 4))[..., None]
    corrected = np.clip(dom - gain * field * w * confine, 0, 1) ** GAMMA
    return apply_inside_box(rgb, corrected, confine), field, af


def second_pass_tile(tile):
    return max(8, tile // 2)


def deleak(rgb, box, gain=1.3, second_pass=True, force_dir=None, tile=None):
    tile = resolve_tile(tile, rgb.shape)
    out, f, af = tile_fix(rgb, box, "warm", tile, gain, force_dir=force_dir)
    if second_pass:
        out, _, _ = tile_fix(out, box, "warm", second_pass_tile(tile), gain * 0.75, force_dir=force_dir)
    return out, f, af


def detint(rgb, box, gain=0.9, second_pass=True, force_dir=None, tile=None):
    tile = resolve_tile(tile, rgb.shape)
    out, f, af = tile_fix(rgb, box, "cool", tile, gain, force_dir=force_dir)
    if second_pass:
        out, _, _ = tile_fix(out, box, "cool", second_pass_tile(tile), gain * 0.65, force_dir=force_dir)
    return out, f, af


def srgb_to_lin(x): return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
def lin_to_srgb(x): x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
def luminance(lin): return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def exposure_match(rgb, box, ref, max_stops=1.0, smooth=25, feather=EXPOSURE_FEATHER_PX):
    H, W, _ = rgb.shape
    x0, x1, y0, y1 = clamp_box(box, rgb.shape); rx0, rx1, ry0, ry1 = clamp_box(ref, rgb.shape)
    lin = srgb_to_lin(rgb); lum = luminance(lin)
    ratio = np.ones(H)
    for y in range(y0, y1):
        ry = min(max(ry0, y), ry1 - 1)
        b = np.median(lum[y, x0:x1]); r = np.median(lum[ry, rx0:rx1])
        ratio[y] = np.clip(r / (b + 1e-6), 1.0, 2.0 ** max_stops)
    ratio[y0:y1] = gaussian_filter1d(ratio[y0:y1], smooth)
    confine = confinement_mask(rgb.shape, box, feather)
    gain = 1 + (ratio[:, None] - 1) * confine
    lifted = lin_to_srgb(np.clip(lin * gain[..., None], 0, 1))
    return apply_inside_box(rgb, lifted, confine[..., None]), float(np.log2(ratio[y0:y1]).mean())


def hsv_neutralize(rgb, box=None, strength=0.45, hue_max=25, sat_min=40, val_min=180, blur=31):
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([hue_max, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if box is not None:
        x0, x1, y0, y1 = clamp_box(box, rgb.shape); m = np.zeros_like(mask); m[y0:y1, x0:x1] = 255; mask = cv2.bitwise_and(mask, m)
    k = blur | 1; alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float64) / 255.0
    if box is not None: alpha *= m > 0
    out = rgb.copy(); out[..., 0] = rgb[..., 0] * (1 - alpha * strength) + rgb[..., 1] * (alpha * strength)
    return np.clip(out, 0, 1), float((mask > 0).mean())


def audit(before, after, boxes):
    H, W = before.shape[:2]; m = np.zeros((H, W), bool)
    for x0, x1, y0, y1 in boxes: m[int(y0):int(y1), int(x0):int(x1)] = True
    d = (before - after) * 255
    inside = d[m].mean(0) if m.any() else np.zeros(3)
    outside = d[~m].mean(0) if (~m).any() else np.zeros(3)
    return inside, outside


def lab_ab(rgb, box):
    x0, x1, y0, y1 = [int(v) for v in box]
    lab = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    return float(lab[y0:y1, x0:x1, 1].mean() - 128), float(lab[y0:y1, x0:x1, 2].mean() - 128)


def load(path):
    if not os.path.isfile(str(path)): raise FileNotFoundError(f"image file not found: {path}")
    bgr = cv2.imread(str(path))
    if bgr is None: raise ValueError(f"not a readable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

def save(path, rgb, quality=93):
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])

import os
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, binary_dilation
from . import core
from .leak_map import warm_excess_map, tile_means, excess_above_background

try:
    import torch
except ImportError:
    torch = None

CLIP_LEVEL = 0.97
MAX_MASK_FRACTION = 0.15
SEAM_SIGMA = 6
RING_PX = 61
GRAIN_STRENGTH = 0.35
GRAIN_SIGMA = 1.5
TELEA_RADIUS = 3
DEFAULT_WEIGHTS = "./big-lama.pt"
WEIGHTS_ENV = "LEAKFIX_LAMA_WEIGHTS"


def box_only_mask(shape, box):
    H, W = shape[:2]; x0, x1, y0, y1 = [int(v) for v in box]
    mask = np.zeros((H, W), bool); mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    return mask


def residue_mask(original, box, clip_level=CLIP_LEVEL):
    clipped_red = (original[..., 0] >= clip_level) & (original[..., 2] < clip_level)
    return clipped_red & box_only_mask(original.shape, box)


def score_mask(original, box, tile=64, min_excess=0.05):
    excess = excess_above_background(tile_means(warm_excess_map(original), tile))
    H, W = original.shape[:2]
    per_pixel = cv2.resize(excess, (excess.shape[1] * tile, excess.shape[0] * tile), interpolation=cv2.INTER_NEAREST)[:H, :W]
    return (per_pixel > min_excess) & box_only_mask(original.shape, box)


MASK_MODES = {"clip": residue_mask, "score": score_mask, "box": lambda original, box: box_only_mask(original.shape, box)}


def mask_fraction(mask):
    return float(mask.mean())


def ring_around(mask, ring_px=RING_PX):
    radius = ring_px // 2
    return binary_dilation(mask, iterations=radius) & ~mask


def lab_float(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def rgb_from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB).astype(np.float64), 0, 1)


def local_color_match(filled, reference, mask, ring_px=RING_PX):
    ring = ring_around(mask, ring_px)
    if not ring.any() or not mask.any(): return filled
    lab = lab_float(filled); target = lab_float(reference)
    for channel in (1, 2):
        lab[..., channel][mask] += target[..., channel][ring].mean() - lab[..., channel][mask].mean()
    out = filled.copy(); out[mask] = rgb_from_lab(lab)[mask]
    return out


def grain_layer(source, mask, sigma=GRAIN_SIGMA, ring_px=RING_PX):
    luma = core.luminance(source)
    high_frequency = luma - gaussian_filter(luma, sigma)
    shift = ring_px // 2
    neighbour_grain = np.roll(high_frequency, (shift, shift), axis=(0, 1))
    return neighbour_grain


def soft_light(base, blend):
    low = base - (1 - 2 * blend) * base * (1 - base)
    d = np.where(base <= 0.25, ((16 * base - 12) * base + 4) * base, np.sqrt(np.clip(base, 0, 1)))
    high = base + (2 * blend - 1) * (d - base)
    return np.where(blend <= 0.5, low, high)


def regrain(filled, source, mask, strength=GRAIN_STRENGTH):
    grain = np.clip(0.5 + grain_layer(source, mask), 0, 1)[..., None]
    grained = soft_light(np.clip(filled, 0, 1), grain)
    out = filled.copy(); out[mask] = filled[mask] + strength * (grained[mask] - filled[mask])
    return np.clip(out, 0, 1)


def feather_alpha(mask, sigma=SEAM_SIGMA):
    reach = int(np.ceil(3 * sigma))
    support = binary_dilation(mask, iterations=reach)
    alpha = cv2.GaussianBlur(mask.astype(np.float64), (0, 0), sigma)
    alpha = np.maximum(alpha, mask) * support
    return np.clip(alpha, 0, 1)


def inpaint_telea(rgb, mask, radius=TELEA_RADIUS):
    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    filled = cv2.inpaint(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def weights_path():
    return os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)


def lama_available():
    return torch is not None and os.path.isfile(weights_path())


def pad_to_multiple(array, multiple=8):
    H, W = array.shape[:2]
    pad_h, pad_w = (-H) % multiple, (-W) % multiple
    widths = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (array.ndim - 2)
    return np.pad(array, widths, mode="reflect"), (H, W)


def inpaint_lama(rgb, mask):
    model = torch.jit.load(weights_path(), map_location="cpu").eval()
    image, (H, W) = pad_to_multiple(np.clip(rgb, 0, 1).astype(np.float32))
    hole, _ = pad_to_multiple(mask.astype(np.float32))
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
    mask_tensor = torch.from_numpy(hole)[None, None]
    with torch.no_grad():
        result = model(image_tensor, mask_tensor)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(result[:H, :W].astype(np.float64), 0, 1)


def choose_method(method):
    if method == "auto": return "lama" if lama_available() else "telea"
    if method == "lama" and not lama_available(): raise RuntimeError(f"LaMa requested but torch or the weights ({weights_path()}) are missing")
    if method not in ("lama", "telea"): raise ValueError(f"unknown inpaint method {method!r}; expected auto, lama or telea")
    return method


def inpaint_residue(rgb, original, box, method="auto", mask_mode="clip", max_mask_fraction=MAX_MASK_FRACTION):
    if mask_mode not in MASK_MODES: raise ValueError(f"unknown mask_mode {mask_mode!r}; expected one of {', '.join(MASK_MODES)}")
    mask = MASK_MODES[mask_mode](original, box)
    fraction = mask_fraction(mask)
    info = {"method": None, "mask_mode": mask_mode, "mask_fraction": fraction, "applied": False}
    if not mask.any(): return rgb, info
    if fraction > max_mask_fraction:
        print(f"inpaint refused: mask covers {fraction * 100:.1f}% of the frame, limit {max_mask_fraction * 100:.0f}%")
        return rgb, info
    used = choose_method(method)
    filled = inpaint_lama(rgb, mask) if used == "lama" else inpaint_telea(rgb, mask)
    filled = regrain(local_color_match(filled, rgb, mask), rgb, mask)
    alpha = feather_alpha(mask)[..., None]
    out = rgb * (1 - alpha) + filled * alpha
    info.update(method=used, applied=True)
    return np.clip(out, 0, 1), info

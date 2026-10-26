import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, label, binary_dilation
from .core import GAMMA

EIGHT_CONNECTED = np.ones((3, 3), bool)


def warm_excess_map(rgb):
    dom = gaussian_filter(rgb ** (1 / GAMMA), (2, 2, 0))
    red_over_blue = dom[..., 0] - dom[..., 2]
    green_over_blue = dom[..., 1] - dom[..., 2]
    return np.clip(red_over_blue - 0.5 * green_over_blue, 0, None)


def tile_means(values, tile):
    H, W = values.shape
    ny, nx = int(np.ceil(H / tile)), int(np.ceil(W / tile))
    grid = np.zeros((ny, nx))
    for ty in range(ny):
        for tx in range(nx):
            grid[ty, tx] = values[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile].mean()
    return grid


def excess_above_background(grid, background_tiles=9):
    local_median = median_filter(grid, size=background_tiles, mode="nearest")
    background = np.minimum(local_median, np.median(grid))
    return np.clip(grid - background, 0, None)


def component_containing(mask, seed):
    labels, _ = label(mask, structure=EIGHT_CONNECTED)
    return labels == labels[seed]


def grow_components(excess, min_excess, peak_fraction=0.3, tail_fraction=0.25):
    assigned = np.zeros(excess.shape, bool)
    components = []
    for flat_index in np.argsort(excess, axis=None)[::-1]:
        seed = np.unravel_index(flat_index, excess.shape)
        if excess[seed] <= min_excess: break
        if assigned[seed]: continue
        core_tiles = component_containing((excess >= peak_fraction * excess[seed]) & ~assigned, seed)
        tail_tiles = binary_dilation(core_tiles, EIGHT_CONNECTED) & (excess > tail_fraction * min_excess) & ~assigned
        member = core_tiles | tail_tiles
        assigned |= member
        components.append(member)
    return components


def component_box(member, excess, tile, shape):
    H, W = shape[:2]
    rows, cols = np.nonzero(member)
    x0, x1 = int(cols.min()) * tile, min(W, (int(cols.max()) + 1) * tile)
    y0, y1 = int(rows.min()) * tile, min(H, (int(rows.max()) + 1) * tile)
    return (x0, x1, y0, y1, float(excess[member].max()))


def propose_boxes(rgb, tile=64, min_tiles=4, min_excess=0.05, background_tiles=9):
    excess = excess_above_background(tile_means(warm_excess_map(rgb), tile), background_tiles)
    components = [m for m in grow_components(excess, min_excess) if m.sum() >= min_tiles]
    return sorted((component_box(m, excess, tile, rgb.shape) for m in components), key=lambda b: -b[4])

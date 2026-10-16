"""Shared particle diameters, independent of empty domain area and display units."""

from itertools import product
import numpy as np


def particle_spacing(particles):
    """Reuse estimates only inside one prepared multi-panel update."""
    cache = getattr(particles, "_spacingCache", None)
    if cache is None:
        return _estimate_particle_spacing(particles)
    if "spacing" not in cache:
        cache["spacing"] = _estimate_particle_spacing(particles)
    return cache["spacing"]


def _estimate_particle_spacing(particles):
    """Conservative population spacing in world units.

    A low volume-spacing percentile handles mixed resolutions. A bounded set of
    local spatial-bin queries also detects compression without an all-pairs
    distance matrix or another dependency. Exact coincident points are ignored:
    changing marker size cannot separate their centers.
    """
    pos = particles.positions.detach().cpu().numpy()
    pos = pos[np.isfinite(pos).all(axis=1)]
    if not len(pos):
        return 0.0
    dim = pos.shape[1]
    volume = (particles.masses / particles.densities).detach().cpu().numpy()
    volume = volume[np.isfinite(volume) & (volume > 0)]
    spacing = float(np.quantile(volume, 0.1) ** (1 / dim)) if len(volume) else 0.0
    extent = np.ptp(pos, axis=0)
    if spacing <= 0:
        spans = extent[extent > 0]
        spacing = (
            float((np.prod(spans) / len(pos)) ** (1 / len(spans)))
            if len(spans)
            else 1.0
        )
    if len(pos) < 2:
        return spacing
    # Flatten integer cells only when their range safely fits an int64 key.
    shape = np.floor(extent / spacing) + 3
    if not np.isfinite(shape).all() or np.prod(shape.astype(float)) > 2**60:
        return spacing
    shape = shape.astype(np.int64)
    stride = np.cumprod(np.r_[1, shape[:-1]])
    cells = np.floor((pos - pos.min(axis=0)) / spacing).astype(np.int64) + 1
    keys = cells @ stride
    order = np.argsort(keys)
    sorted_keys = keys[order]
    sample = np.linspace(0, len(pos) - 1, min(512, len(pos)), dtype=int)
    offsets = np.array(list(product((-1, 0, 1), repeat=dim))) @ stride
    query = keys[sample, None] + offsets
    lo = np.searchsorted(sorted_keys, query, side="left")
    hi = np.searchsorted(sorted_keys, query, side="right")
    # At most 16 candidates per bin, including endpoints for dense cells.
    count = hi - lo
    index = lo[..., None] + (
        np.maximum(count - 1, 0)[..., None] * np.linspace(0, 1, 16)
    ).astype(int)
    candidates = pos[order[np.minimum(index, len(pos) - 1)]]
    dist2 = np.sum((candidates - pos[sample, None, None, :]) ** 2, axis=-1)
    valid = (count[..., None] > 0) & (dist2 > 0)
    nearest = np.sqrt(np.where(valid, dist2, np.inf).min(axis=(1, 2)))
    nearest = nearest[np.isfinite(nearest)]
    return min(spacing, float(np.quantile(nearest, 0.1))) if len(nearest) else spacing


def validate_marker_options(options):
    if options.markerSize is not None:
        explicit_pixel_size(options.markerSize)
    elif (
        not np.isfinite(float(options.markerSizeScale))
        or float(options.markerSizeScale) <= 0
    ):
        raise ValueError("markerSizeScale must be finite and positive")


def particle_diameter(particles, options):
    """Automatic world diameter; explicit markerSize is resolved by the renderer."""
    scale = float(options.markerSizeScale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("markerSizeScale must be finite and positive")
    return scale * particle_spacing(particles)


def explicit_pixel_size(marker_size):
    """Match Matplotlib's normalized marker path: sqrt(s) points across."""
    size = float(marker_size)
    if not np.isfinite(size) or size < 0:
        raise ValueError("markerSize must be finite and nonnegative")
    return max(0.1, np.sqrt(size) * 96 / 72)

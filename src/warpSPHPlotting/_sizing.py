"""Shared particle diameters, independent of empty domain area and display units."""

from itertools import product
import numpy as np


def particle_spacing(particles, mode="neighbors"):
    """Reuse estimates only inside one prepared multi-panel update."""
    if mode not in ("volume", "neighbors"):
        raise ValueError("markerSpacing must be 'volume' or 'neighbors'")
    estimate = (
        _volume_particle_spacing if mode == "volume" else _estimate_particle_spacing
    )
    cache = getattr(particles, "_spacingCache", None)
    if cache is None:
        return estimate(particles)
    key = f"spacing:{mode}"
    if key not in cache:
        cache[key] = estimate(particles)
    return cache[key]


def _volume_particle_spacing(particles):
    """Nominal resolution stays independent of rearrangement and close pairs."""
    if not len(particles.positions):
        return 0.0
    volume = (particles.masses / particles.densities).detach().cpu().numpy()
    volume = volume[np.isfinite(volume) & (volume > 0)]
    if len(volume):
        return float(np.quantile(volume, 0.1) ** (1 / particles.positions.shape[1]))
    # Geometry is the remaining resolution cue when no usable volumes exist.
    return _estimate_particle_spacing(particles)


def _estimate_particle_spacing(particles):
    """Conservative population spacing in world units.

    A low volume-spacing percentile handles mixed resolutions. A bounded set of
    local spatial-bin queries also detects compression without an all-pairs
    distance matrix or another dependency. Exact coincident points are ignored:
    changing marker size cannot separate their centers.
    """
    # Coordinate-major storage makes reductions contiguous. Reducing millions
    # of short rows (N, 2/3) is much slower than reducing two/three long arrays.
    # Keep every finite position and the existing query/candidate ordering.
    coordinates = np.ascontiguousarray(particles.positions.detach().cpu().numpy().T)
    finite = np.isfinite(coordinates).all(axis=0)
    if not finite.all():
        coordinates = np.ascontiguousarray(coordinates[:, finite])
    pos = coordinates.T
    if not len(pos):
        return 0.0
    dim = pos.shape[1]
    volume = (particles.masses / particles.densities).detach().cpu().numpy()
    volume = volume[np.isfinite(volume) & (volume > 0)]
    spacing = float(np.quantile(volume, 0.1) ** (1 / dim)) if len(volume) else 0.0
    minimum = coordinates.min(axis=1)
    extent = coordinates.max(axis=1) - minimum
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
    keys = np.zeros(len(pos), dtype=np.int64)
    for axis in range(dim):
        cells = (
            np.floor((coordinates[axis] - minimum[axis]) / spacing).astype(np.int64) + 1
        )
        keys += cells * stride[axis]
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
    if options.markerSpacing not in ("volume", "neighbors"):
        raise ValueError("markerSpacing must be 'volume' or 'neighbors'")
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
    return scale * particle_spacing(particles, options.markerSpacing)


def explicit_pixel_size(marker_size):
    """Match Matplotlib's normalized marker path: sqrt(s) points across."""
    size = float(marker_size)
    if not np.isfinite(size) or size < 0:
        raise ValueError("markerSize must be finite and nonnegative")
    return max(0.1, np.sqrt(size) * 96 / 72)

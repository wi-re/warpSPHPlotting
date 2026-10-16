"""Compare legacy empty-as-zero grids with coverage-aware transparency.

Uses real saved frontend buffers, preserving all physical interpolation values.
"""

import argparse
from pathlib import Path
import torch
from warpSPHCore import DomainDescription
from warpSPHPlotting import (
    visualize,
    PlottingOptions,
    GridVisualization,
    VisualizeOptions,
    Mapping,
)
from stress_replay import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["matplotlib", "pyvista", "vispy"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain-scale", type=float, default=1)
    parser.add_argument("--resolution", type=int, default=192)
    fields = parser.add_mutually_exclusive_group()
    fields.add_argument(
        "--zero",
        action="store_true",
        help="Show a genuine zero field within the same particle coverage.",
    )
    fields.add_argument(
        "--streamlines",
        action="store_true",
        help="Plot the actual velocity field and its streamlines.",
    )
    args = parser.parse_args()
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()
    torch.set_num_threads(2)
    p, v, d = load(args.input)
    if args.domain_scale != 1:
        center = (d.min + d.max) * 0.5
        half = (d.max - d.min) * 0.5 * args.domain_scale
        d = DomainDescription(
            center - half, center + half, torch.zeros_like(d.periodic), 2
        )
    q = (
        torch.zeros_like(p.densities)
        if args.zero
        else (v if args.streamlines else p.densities)
    )
    # Compare opacity with an identical, useful scale on both panels. Particle
    # bounds can be much narrower than the SPH grid estimator's actual range.
    from warpSPHPlotting.backends._render_util import prepare_frame
    from warpSPHPlotting.grid import mapFrameToGrid, gridDisplayBounds

    scale_options = PlottingOptions(
        gridVisualization=GridVisualization(resolution=args.resolution),
        boundaryVisualization=VisualizeOptions.Passive,
        mapping=Mapping.L2 if args.streamlines else Mapping.none,
    )
    frame = prepare_frame(p, d, q, scale_options)
    _, values, _, extent = mapFrameToGrid(frame, frame.particles.quantities)
    _, norm = gridDisplayBounds(values, extent, scale_options)
    lo, hi = (
        (-1.0, 1.0)
        if args.zero
        else ((float(norm.vmin), float(norm.vmax)) if norm is not None else (0.0, 1.0))
    )
    opts = {
        key: PlottingOptions(
            gridVisualization=GridVisualization(
                resolution=args.resolution, maskEmpty=mask, streamLines=args.streamlines
            ),
            boundaryVisualization=VisualizeOptions.Passive,
            mapping=Mapping.L2 if args.streamlines else Mapping.none,
            vMin=lo,
            vMax=hi,
            plotTitle=title,
        )
        for key, mask, title in [
            ("A", False, "Empty cells colored as zero"),
            ("B", True, "Empty cells transparent"),
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot = visualize(
        p,
        d,
        q,
        opts,
        mosaic="AB",
        figsize=(11, 5),
        backend=args.backend,
        backendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if args.backend == "vispy"
            else {}
        ),
        figTitle=f'{args.input.parent.name} · {"genuine zero field" if args.zero else "speed" if args.streamlines else "density"} · {args.backend}',
    )
    try:
        plot.export(str(args.output), dpi=120)
    finally:
        plot.close()


if __name__ == "__main__":
    main()

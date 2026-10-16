"""Render saved frontend states through each backend for visual comparison.

This replay needs warpSPHCore and plotting extras, but does not need warpSPH.
The NPZ fixtures are produced by verify_frontends.py's actual simulations.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="matplotlib", choices=["matplotlib", "pyvista", "vispy"]
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domain-scale",
        type=float,
        default=1.0,
        help="Scale only the displayed domain; useful for sparse-view stress checks.",
    )
    parser.add_argument("--title")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=None,
        help="Fixed scatter area in points squared; omit for automatic sizing.",
    )
    args = parser.parse_args()
    data = np.load(args.input)
    p = ParticleState(
        **{
            name: torch.from_numpy(data[name].copy())
            for name in ("positions", "supports", "masses", "densities", "kinds")
        }
    )
    d = DomainDescription(
        torch.from_numpy(data["domain_min"]),
        torch.from_numpy(data["domain_max"]),
        torch.from_numpy(data["periodic"]),
        2,
    )
    if args.domain_scale != 1:
        center = (d.min + d.max) * 0.5
        half = (d.max - d.min) * 0.5 * args.domain_scale
        d = DomainDescription(
            center - half, center + half, torch.zeros_like(d.periodic), 2
        )
    v = torch.from_numpy(data["velocities"])
    opts = {
        "A": PlottingOptions(
            mapping=Mapping.L2,
            plotTitle="Speed",
            markerSize=args.marker_size,
            boundaryVisualization=VisualizeOptions.Passive,
        ),
        "B": PlottingOptions(
            plotTitle="Density",
            markerSize=args.marker_size,
            boundaryVisualization=VisualizeOptions.Visualize,
            quantityScaling=PlotScaling.Symmetric,
            midPoint="median",
        ),
    }
    beopts = (
        {"jupyter_backend": "image", "app_backend": "egl"}
        if args.backend == "vispy"
        else {}
    )
    s = visualize(
        p,
        d,
        {"A": v, "B": p.densities},
        opts,
        mosaic="AB",
        figsize=(11, 5),
        backend=args.backend,
        backendOptions=beopts,
        figTitle=args.title or f"{args.input.parent.name} · {args.backend}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        s.export(str(args.output), dpi=120)
    finally:
        s.close()


if __name__ == "__main__":
    main()

"""Periodic Taylor-Green field: scatter, grid, streamlines, updates and export.

This is an analytical plotting example. For actual time integration use
verify_frontends.py. Run from the repository root with the desired backend::

    python examples/quickstart.py --backend matplotlib
    python examples/quickstart.py --backend pyvista
    EGL_PLATFORM=surfaceless python examples/quickstart.py --backend vispy
"""

import argparse
from pathlib import Path
from _artifacts import development_dir
import math
import torch
import warp as wp
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import (
    visualize,
    PlottingOptions,
    GridVisualization,
    Mapping,
    export_particles_vtp,
    waitForExports,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="matplotlib", choices=["matplotlib", "pyvista", "vispy"]
    )
    parser.add_argument(
        "--output", type=Path, default=development_dir() / "output/quickstart"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    n = 24
    x = (torch.arange(n) + 0.5) / n
    positions = torch.stack(torch.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    count = len(positions)
    p = ParticleState(
        positions,
        torch.full((count,), 0.18),
        torch.full((count,), 1 / n**2),
        torch.zeros(count, dtype=torch.int32),
        torch.ones(count),
    )
    d = DomainDescription(
        torch.zeros(2), torch.ones(2), torch.ones(2, dtype=torch.bool), 2
    )
    velocity = torch.stack(
        (
            torch.sin(2 * math.pi * positions[:, 0])
            * torch.cos(2 * math.pi * positions[:, 1]),
            -torch.cos(2 * math.pi * positions[:, 0])
            * torch.sin(2 * math.pi * positions[:, 1]),
        ),
        dim=-1,
    )
    options = {
        "A": PlottingOptions(
            mapping=Mapping.L2, plotTitle="Particle speed", markerSize=8
        ),
        "B": PlottingOptions(
            mapping=Mapping.L2,
            plotTitle="Interpolated speed and streamlines",
            gridVisualization=GridVisualization(resolution=32, streamLines=True),
        ),
    }
    backend_options = (
        {"jupyter_backend": "image", "app_backend": "egl"}
        if args.backend == "vispy"
        else {}
    )
    plot = visualize(
        p,
        d,
        velocity,
        options,
        mosaic="AB",
        figsize=(11, 5),
        backend=args.backend,
        backendOptions=backend_options,
        figTitle="Periodic Taylor–Green field · initial",
    )
    try:
        plot.export(str(args.output / f"{args.backend}_initial.png"))
        plot.updateQuantities(velocity * 0.6, newParticleState=p)
        plot.updateTitle("Periodic Taylor–Green field · amplitude × 0.6")
        plot.export(str(args.output / f"{args.backend}_updated.png"))
        export_particles_vtp(
            args.output / "particles.vtp",
            p,
            {"velocity": velocity, "speed": torch.linalg.vector_norm(velocity, dim=1)},
            d,
            {"source": "analytical Taylor-Green"},
        )
        waitForExports()
    finally:
        plot.close()


if __name__ == "__main__":
    main()

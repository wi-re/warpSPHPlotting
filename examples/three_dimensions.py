"""Analytical 3D Taylor-Green field: particles, volume, and movable slices.

This is visualization input, not an integrated fluid simulation. The optional
spherical crop leaves empty space to verify the volume's coverage mask.
"""

import argparse
import json
from pathlib import Path
from _artifacts import development_dir
import numpy as np
import torch
import warp as wp
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import (
    visualize,
    PlottingOptions,
    GridVisualization,
    Mapping,
)


def field(nx=32, dim=3, device="cpu", crop=False):
    axis = (torch.arange(nx, device=device) + 0.5) * (2 * torch.pi / nx)
    positions = torch.stack(torch.meshgrid(*([axis] * dim), indexing="ij"), -1).reshape(
        -1, dim
    )
    if crop:
        positions = positions[
            torch.linalg.vector_norm(positions - torch.pi, dim=1) < 0.86 * torch.pi
        ]
    n = len(positions)
    velocity = torch.zeros_like(positions)
    x, y = positions[:, 0], positions[:, 1]
    zfactor = torch.cos(positions[:, 2]) if dim == 3 else 1
    velocity[:, 0] = torch.sin(x) * torch.cos(y) * zfactor
    velocity[:, 1] = -torch.cos(x) * torch.sin(y) * zfactor
    dx = 2 * torch.pi / nx
    particles = ParticleState(
        positions=positions,
        supports=torch.full((n,), dx * 2.2, device=device),
        masses=torch.full((n,), dx**dim, device=device),
        densities=torch.ones(n, device=device),
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
    )
    domain = DomainDescription(
        torch.zeros(dim, device=device),
        torch.full((dim,), 2 * torch.pi, device=device),
        torch.full((dim,), not crop, device=device, dtype=torch.bool),
        dim,
    )
    return particles, domain, velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=development_dir() / "output/3d")
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--backend", choices=["auto", "pyvista", "vispy"], default="auto"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()
    args.output.mkdir(parents=True, exist_ok=True)
    p, d, v = field(args.nx, device=args.device, crop=True)
    opts = PlottingOptions(
        mapping=Mapping.L2,
        vMin=0,
        vMax=1,
        plotTitle=f"3D particles · {len(v):,} samples",
    )
    beopts = (
        {"jupyter_backend": "image", "app_backend": "egl"}
        if args.backend == "vispy"
        else {}
    )
    s = visualize(
        p, d, v, opts, backend=args.backend, backendOptions=beopts, figsize=(8, 6)
    )
    resolved = s.backend
    try:
        s.export(str(args.output / f"{resolved}_particles.png"))
        if resolved == "pyvista":
            for mode in ("surface", "volume"):
                s.updateQuantities(
                    v,
                    newOptions={
                        "A": {
                            "gridVisualization": GridVisualization(
                                resolution=args.resolution, renderMode=mode
                            ),
                            "plotTitle": f"3D {mode} · SPH speed",
                        }
                    },
                    redraw=False,
                )
                s.export(str(args.output / f"pyvista_{mode}.png"))
            # Orbit the native 3D camera, then prove a quantity update retains it.
            s.fig.camera.azimuth += 55
            camera = tuple(tuple(x) for x in s.fig.camera_position)
            s.updateQuantities(v * 0.5, redraw=False)
            assert tuple(tuple(x) for x in s.fig.camera_position) == camera
            s.export(str(args.output / "pyvista_volume_orbit.png"))
    finally:
        s.close()

    # The physical 3D kernel is sampled at the plane, not a projected cloud.
    # Use the selected backend explicitly so the slice round trip uses its API.
    s = visualize(
        p,
        d,
        v,
        PlottingOptions(
            mapping=Mapping.L2,
            vMin=0,
            vMax=1,
            gridVisualization=GridVisualization(
                resolution=72, sliceAxis=2, slicePosition=torch.pi
            ),
            plotTitle="XY slice · Z = 3.142",
        ),
        backend=resolved,
        backendOptions=beopts,
        figsize=(8, 6),
    )
    try:
        s.export(str(args.output / f"{resolved}_slice_mid.png"))
        s.updateQuantities(
            v,
            newOptions={
                "A": {
                    "gridVisualization": GridVisualization(
                        resolution=72, sliceAxis=2, slicePosition=torch.pi / 2
                    ),
                    "plotTitle": "XY slice · Z = 1.571 (analytic velocity node)",
                }
            },
            redraw=False,
        )
        s.export(str(args.output / f"{resolved}_slice_quarter.png"))
        # Also expose both representations in a mixed mosaic, with default
        # camera sharing: incompatible planar/3D cameras stay independent.
    finally:
        s.close()
    s = visualize(
        p,
        d,
        v,
        {
            "A": PlottingOptions(
                mapping=Mapping.L2, vMin=0, vMax=1, plotTitle="3D particles"
            ),
            "B": PlottingOptions(
                mapping=Mapping.L2,
                vMin=0,
                vMax=1,
                gridVisualization=GridVisualization(resolution=72, sliceAxis=2),
                plotTitle="XY slice through the center",
            ),
        },
        mosaic="AB",
        backend=resolved,
        backendOptions=beopts,
        figsize=(12, 5),
    )
    try:
        s.export(str(args.output / f"{resolved}_mixed.png"))
    finally:
        s.close()
    (args.output / f"{resolved}_example.json").write_text(
        json.dumps(
            {
                "field": "analytical 3D Taylor-Green, spherical crop",
                "simulation_integrated": False,
                "particles": len(v),
                "device": args.device,
                "backend": resolved,
                "grid_resolution": args.resolution,
                "camera_update_preserved": resolved == "pyvista",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

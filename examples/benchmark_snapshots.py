"""Replay real frontend states and benchmark the plotting pipeline separately.

Run with PYTHONPATH pointing to another checkout to compare implementations.
Use the same environment, samples, canvas size, and thread count for both.
"""

import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import visualize, PlottingOptions, VisualizeOptions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--backend", choices=["matplotlib", "pyvista", "vispy"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--mosaic", default="A")
    p.add_argument(
        "--marker-spacing", choices=["volume", "neighbors"], default="volume"
    )
    p.add_argument(
        "--independent-sizing",
        action="store_true",
        help="Benchmark the previous independent per-panel spacing estimates.",
    )
    p.add_argument(
        "--marker-size",
        type=float,
        default=None,
        help="Fixed scatter size; omit to benchmark automatic sizing.",
    )
    p.add_argument(
        "--count", type=int, help="Repeat samples to test a larger particle population."
    )
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.independent_sizing:
        import warpSPHPlotting._sizing as sizing

        def independent_spacing(particles, mode="neighbors"):
            estimate = (
                sizing._volume_particle_spacing
                if mode == "volume"
                else sizing._estimate_particle_spacing
            )
            return estimate(particles)

        sizing.particle_spacing = independent_spacing
    source = np.load(args.input)
    n = len(source["positions"])
    ids = np.arange(args.count or n) % n
    values = {
        name: torch.from_numpy(source[name][ids].copy())
        for name in ("positions", "supports", "masses", "densities", "kinds")
    }
    particles = ParticleState(**values)
    domain = DomainDescription(
        torch.from_numpy(source["domain_min"]),
        torch.from_numpy(source["domain_max"]),
        torch.from_numpy(source["periodic"]),
        2,
    )
    q = particles.densities
    opts = PlottingOptions(
        markerSize=args.marker_size,
        markerSpacing=args.marker_spacing,
        plotTitle="Density",
        boundaryVisualization=VisualizeOptions.Passive,
    )
    beopts = (
        {"jupyter_backend": "static"}
        if args.backend == "pyvista"
        else (
            {"jupyter_backend": "image", "app_backend": "egl"}
            if args.backend == "vispy"
            else {}
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    s = visualize(
        particles,
        domain,
        q,
        opts,
        backend=args.backend,
        mosaic=args.mosaic,
        backendOptions=beopts,
        figsize=(7, 5),
    )
    create = time.perf_counter() - start
    times = []
    try:
        for i in range(args.frames + 3):
            start = time.perf_counter()
            s.updateQuantities(
                q * (1 + 0.0001 * i), newParticleState=particles, redraw=False
            )
            updated = time.perf_counter()
            # render() is the underlying output of each backend; exclude PNG
            # compression to distinguish renderer time from image encoding.
            if args.backend == "matplotlib":
                s.fig.canvas.draw()
            elif args.backend == "pyvista":
                s.fig.render()
                s.fig.screenshot(return_img=True)
            else:
                s.fig.render()
            end = time.perf_counter()
            if i >= 3:
                times.append(
                    {
                        "prepare_ms": 1000 * (updated - start),
                        "render_ms": 1000 * (end - updated),
                        "total_ms": 1000 * (end - start),
                    }
                )
        s.export(str(args.output.with_suffix(".png")), dpi=120)
    finally:
        if hasattr(s, "close"):
            s.close()
        elif args.backend == "pyvista":
            s.fig.close()
        elif args.backend == "vispy":
            s.fig.close()
    summary = {
        "backend": args.backend,
        "particles": len(ids),
        "marker_size": args.marker_size,
        "marker_spacing": args.marker_spacing,
        "mosaic": args.mosaic,
        "independent_sizing": args.independent_sizing,
        "input": str(args.input),
        "create_s": create,
        "samples": times,
        "median_ms": {
            key: float(np.median([r[key] for r in times])) for key in times[0]
        },
        "p95_total_ms": float(np.percentile([r["total_ms"] for r in times], 95)),
    }
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}))


if __name__ == "__main__":
    main()

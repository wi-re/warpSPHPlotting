"""Measure full-population 2D/3D scatter updates, synchronized render/readback.

No particle sampling, duplicated positions, file encoding in the measured loop,
or simulation stepping. Separate preparation and renderer timings expose CPU
costs remaining when a GPU backend is selected.
"""

import argparse
import json
from pathlib import Path
import time
import platform
import numpy as np
import torch
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import PlottingOptions, visualize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend", choices=["auto", "matplotlib", "pyvista", "vispy"], required=True
    )
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument(
        "--counts", type=int, nargs="+", default=[10_000, 100_000, 1_000_000]
    )
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--marker-spacing", choices=["volume", "neighbors"], default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--reference-sizing",
        type=Path,
        help="Load an archived spacing estimator for a controlled before/after comparison.",
    )
    args = p.parse_args()
    spacing_mode = args.marker_spacing or (
        "neighbors" if args.reference_sizing else "volume"
    )
    if args.reference_sizing and spacing_mode != "neighbors":
        p.error(
            "--reference-sizing compares the archived neighbor estimator; use --marker-spacing neighbors"
        )
    if args.samples < 1 or any(n < 1 for n in args.counts):
        p.error("samples and particle counts must be positive")
    if args.reference_sizing:
        import importlib.util
        import warpSPHPlotting._sizing as sizing

        spec = importlib.util.spec_from_file_location(
            "reference_sizing", args.reference_sizing
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        sizing._estimate_particle_spacing = reference._estimate_particle_spacing
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for count in args.counts:
        gen = torch.Generator().manual_seed(31415)
        pos = torch.rand((count, args.dim), generator=gen)
        state = ParticleState(
            positions=pos,
            supports=torch.full((count,), 0.02),
            masses=torch.full((count,), 1 / count),
            densities=torch.ones(count),
            kinds=torch.zeros(count, dtype=torch.int32),
        )
        domain = DomainDescription(
            torch.zeros(args.dim),
            torch.ones(args.dim),
            torch.zeros(args.dim, dtype=torch.bool),
            args.dim,
        )
        q = 0.5 + 0.5 * torch.sin(pos[:, 0] * 2 * torch.pi) * torch.cos(
            pos[:, 1] * 2 * torch.pi
        )
        beopts = (
            {"jupyter_backend": "image", "app_backend": "egl"}
            if args.backend == "vispy"
            else {}
        )
        start = time.perf_counter()
        s = visualize(
            state,
            domain,
            q,
            PlottingOptions(
                vMin=0,
                vMax=1,
                plotTitle=f"{count:,} particles · {args.dim}D",
                markerSpacing=spacing_mode,
            ),
            backend=args.backend,
            backendOptions=beopts,
            figsize=(8, 6),
        )
        # PyVista static mode initializes its window on first capture outside
        # notebooks. Include that initialization in creation, not a sample.
        if s.backend == "pyvista":
            try:
                s.fig.render()
                s.fig.screenshot(return_img=True)
            finally:
                s.fig.render_window.ReleaseCurrent()
        elif s.backend == "matplotlib":
            s.fig.canvas.draw()
        else:
            s.fig.render()
        created = time.perf_counter() - start
        samples = []
        renderer = None
        try:
            if s.backend == "pyvista":
                s.fig.render_window.MakeCurrent()
                renderer = s.fig.render_window.ReportCapabilities()
                s.fig.render_window.ReleaseCurrent()
            elif s.backend == "vispy":
                from vispy import gloo
                from warpSPHPlotting.backends.vispy_backend import (
                    _release_current_gl_context,
                )

                s.fig.set_current()
                renderer = str(gloo.gl.glGetParameter(gloo.gl.GL_RENDERER))
                _release_current_gl_context(s.fig)
            for i in range(args.samples + 3):
                start = time.perf_counter()
                s.updateQuantities(
                    q * (1 - 0.001 * i), newParticleState=state, redraw=False
                )
                updated = time.perf_counter()
                if s.backend == "matplotlib":
                    s.fig.canvas.draw()
                elif s.backend == "pyvista":
                    try:
                        s.fig.render()
                        s.fig.screenshot(return_img=True)
                    finally:
                        s.fig.render_window.ReleaseCurrent()
                else:
                    s.fig.render()
                end = time.perf_counter()
                if i >= 3:
                    samples.append(
                        {
                            "prepare_ms": 1000 * (updated - start),
                            "render_readback_ms": 1000 * (end - updated),
                            "total_ms": 1000 * (end - start),
                        }
                    )
            s.export(str(args.output / f"{s.backend}_{args.dim}d_{count}.png"))
            from PIL import Image

            pixels = np.asarray(
                Image.open(args.output / f"{s.backend}_{args.dim}d_{count}.png")
            )[:, :, :3].astype(float)
            colored = int((np.ptp(pixels, axis=2) > 20).sum())
            if colored < 50:
                raise RuntimeError(
                    "Benchmark export is blank or lacks the colored particle field"
                )
            results.append(
                {
                    "requested_backend": args.backend,
                    "reference_sizing": (
                        str(args.reference_sizing) if args.reference_sizing else None
                    ),
                    "backend": s.backend,
                    "dimension": args.dim,
                    "particles": count,
                    "marker_spacing": spacing_mode,
                    "size_pixels": [768, 576],
                    "threads": 2,
                    "python": platform.python_version(),
                    "renderer": renderer,
                    "render_window": (
                        type(s.fig.render_window).__name__
                        if s.backend == "pyvista"
                        else (s.fig.app.backend_name if s.backend == "vispy" else "Agg")
                    ),
                    "colored_pixels": colored,
                    "create_s": created,
                    "samples": samples,
                    "median_ms": {
                        k: float(np.median([r[k] for r in samples])) for k in samples[0]
                    },
                    "p95_total_ms": float(
                        np.percentile([r["total_ms"] for r in samples], 95)
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in results[-1].items()
                        if k not in ("renderer", "samples")
                    }
                ),
                flush=True,
            )
        finally:
            s.close()
    (args.output / f"{args.backend}_{args.dim}d.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()

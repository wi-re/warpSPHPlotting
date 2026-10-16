"""Exercise actual warpSPH frontend cases and save plotting evidence.

Requires the optional sibling warpSPH frontend installation. Run each case in
its own process to keep JIT and graphics initialization explicit::

    python examples/verify_frontends.py --case tgv --backend pyvista --steps 40
    python examples/verify_frontends.py --case sod --backend matplotlib --steps 40
    python examples/verify_frontends.py --case marrone31 --backend pyvista --steps 120
    python examples/verify_frontends.py --case khh --backend pyvista --steps 40

These are visualization/short-trajectory checks, not physical convergence tests.
"""

import argparse
from dataclasses import replace
from pathlib import Path
from _artifacts import evidence_dir
import json
import time
import numpy as np
from _verification_metrics import resident_bytes, scene_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["tgv", "sod", "marrone31", "khh"], required=True
    )
    parser.add_argument(
        "--backend", choices=["matplotlib", "pyvista", "vispy"], default="matplotlib"
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--nx", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-offset", type=int, default=0)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--plot-interval", type=int)
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Save temporal NPZ states for renderer stress replay.",
    )
    parser.add_argument("--compact-diagnostics", action="store_true")
    parser.add_argument(
        "--native-hooks",
        action="store_true",
        help="Use the original frontend plotting hooks.",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=128,
        help="KHH grid resolution for bounded CPU native-hook checks.",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--sparse-view",
        action="store_true",
        help="Expand only the plotting viewport around real Sod particles.",
    )
    parser.add_argument("--output", type=Path, default=evidence_dir("frontend"))
    args = parser.parse_args()
    from warpSPHBootstrap import bootstrap

    bootstrap(precision="float32")
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()
    import torch

    torch.set_num_threads(args.threads)
    from warpSPH.runner import run
    from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase
    from warpSPH.cases.compressible.sodND import sod2dCase
    from warpSPH.cases.freeSurface.dambreak import dambreakCase
    from warpSPH.cases.compressible.kelvinHelmholtz import kelvinHelmholtzCase
    from warpSPHPlotting import (
        visualize,
        PlottingOptions,
        Mapping,
        VisualizeOptions,
        PlotScaling,
    )

    cases = {
        "tgv": tgvWeaklyCompressibleCase,
        "sod": sod2dCase,
        "marrone31": dambreakCase,
        "khh": kelvinHelmholtzCase,
    }
    case = cases[args.case]
    out = args.output.resolve() / f"{args.case}-{args.backend}"
    out.mkdir(parents=True, exist_ok=True)
    measurements = {
        "case": args.case,
        "backend": args.backend,
        "device": args.device,
        "steps_requested": args.steps,
        "threads": args.threads,
        "sparse_view": args.sparse_view,
        "frames": [],
    }
    plotters = []

    def save_snapshot(path, particles, domain):
        np.savez_compressed(
            path,
            **{
                name: getattr(particles, name).detach().cpu().numpy()
                for name in (
                    "positions",
                    "velocities",
                    "densities",
                    "masses",
                    "supports",
                    "kinds",
                )
            },
            domain_min=domain.min.cpu().numpy(),
            domain_max=domain.max.cpu().numpy(),
            periodic=domain.periodic.cpu().numpy(),
        )

    def quantities(state):
        return {"A": state.state.velocities, "B": state.state.densities}

    def options():
        return {
            "A": PlottingOptions(
                mapping=Mapping.L2,
                plotTitle="Speed",
                boundaryVisualization=VisualizeOptions.Passive,
            ),
            "B": PlottingOptions(
                plotTitle="Density",
                boundaryVisualization=VisualizeOptions.Visualize,
                quantityScaling=PlotScaling.Symmetric,
                midPoint="median",
            ),
        }

    def frame(ctx, state, step):
        from warpSPHCore import DomainDescription

        display_domain = ctx.config.domain
        if args.sparse_view:
            center = (display_domain.min + display_domain.max) * 0.5
            half = (display_domain.max - display_domain.min) * (
                0.75 + 3 * step / max(1, args.steps)
            )
            display_domain = DomainDescription(
                center - half,
                center + half,
                torch.zeros_like(display_domain.periodic),
                2,
            )
        tick = time.perf_counter()
        if not plotters:
            beopts = (
                {"jupyter_backend": "static"}
                if args.backend == "pyvista"
                else (
                    {"jupyter_backend": "image", "app_backend": "egl"}
                    if args.backend == "vispy"
                    else {}
                )
            )
            plotters.append(
                visualize(
                    state.state,
                    display_domain,
                    quantities(state),
                    options(),
                    mosaic="AB",
                    figsize=(11, 5),
                    figTitle=f"{args.case} — step {step}, t={state.t:.5g}",
                    backend=args.backend,
                    backendOptions=beopts,
                )
            )
        else:
            plotters[0].updateQuantities(
                quantities(state),
                newParticleState=state.state,
                newDomain=display_domain,
                redraw=False,
            )
            plotters[0].updateTitle(f"{args.case} — step {step}, t={state.t:.5g}")
        prepared = time.perf_counter()
        plotters[0].export(str(out / f"frame_{step:05d}.png"), dpi=120)
        end = time.perf_counter()
        measurements["frames"].append(
            {
                "step": step,
                "time": float(state.t),
                "prepare_ms": 1000 * (prepared - tick),
                "export_ms": 1000 * (end - prepared),
                "particles": len(state.state.positions),
                "finite": bool(
                    torch.isfinite(state.state.positions).all()
                    and torch.isfinite(state.state.densities).all()
                ),
                "domain_min": display_domain.min.tolist(),
                "domain_max": display_domain.max.tolist(),
                "rss_bytes": resident_bytes(),
                "scene_counts": scene_counts(plotters[0]),
            }
        )
        if args.snapshots:
            snapshots = out / "snapshots"
            snapshots.mkdir(exist_ok=True)
            save_snapshot(
                snapshots / f"state_{step:05d}.npz", state.state, display_domain
            )
        (out / "progress.json").write_text(json.dumps(measurements, indent=2))
        return plotters[0]

    def setup(ctx, state):
        return frame(ctx, state, ctx.spec.resumeStepOffset)

    def update(ctx, state, plotter, step):
        frame(ctx, state, step)

    # Use the real frontend construction, integrator, diagnostics and output
    # scheduler. Replace only plot hooks to compare exactly the same panels.
    def progress(ctx, state, step):
        if case.postStep is not None:
            case.postStep(ctx, state, step)
        if step % 100 == 0:
            print(json.dumps({"step": step, "time": float(state.t)}), flush=True)

    tested = replace(case, setupPlot=setup, updatePlot=update, postStep=progress)
    if args.native_hooks:
        tested = replace(case, postStep=progress)
        if args.case == "khh":
            from warpSPH.cases.compressible.kelvinHelmholtz import (
                KELVIN_HELMHOLTZ_FIELDS,
            )
            from warpSPH.cases.common.plotting import particlePlot

            fields = [
                (
                    replace(f, gridResolution=args.grid_resolution)
                    if f.gridResolution
                    else f
                )
                for f in KELVIN_HELMHOLTZ_FIELDS
            ]
            native_setup, native_update = particlePlot(fields, dpi=120)
            tested = replace(tested, setupPlot=native_setup, updatePlot=native_update)
    measurements["native_hooks"] = args.native_hooks
    kw = dict(
        device=args.device,
        nx=args.nx or (67 if args.case == "marrone31" else 24),
        nSteps=args.steps,
        plot=True,
        plotBackend=args.backend,
        plotBackendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if args.backend == "vispy"
            else ({"jupyter_backend": "static"} if args.backend == "pyvista" else {})
        ),
        show=False,
        holdPlot=False,
        asyncPlot=False,
        plotInterval=args.plot_interval or max(1, args.steps // 4),
        store=False,
        storeInterval=args.steps + 1,
        resumeFrom=str(args.resume.resolve()) if args.resume else None,
        resumeStepOffset=args.resume_offset,
        video=False,
        progress=False,
        quiet=True,
        execution="eager",
        cudaGraph=False,
        pipelineOutputs=False,
        exportRoot=str(out / "runner"),
    )
    if args.case == "marrone31":
        kw.update(
            scheme="sun2017DeltaSPH",
            L=1.0,
            periodic=False,
            params=dict(
                W=3.2196,
                fillRatio=0.6,
                fluidWidth=1.2 / 3.2196,
                gravityMagnitude=9.81,
                referenceVelocity=1.95 * (9.81 * 0.6) ** 0.5,
                machTarget=1.95 / 40,
                shifting=False,
                wallBC="freeSlip",
            ),
        )
    start = time.perf_counter()
    try:
        result = run(tested, **kw)
        measurements.update(
            export_path=result.exportPath,
            actual_backend=result.ctx.scratch.get("plotBackend", args.backend),
            steps=result.nSteps,
            diverged=result.diverged,
            stop_reason=result.stopReason,
            wall_s=time.perf_counter() - start,
            simulation_time=float(result.state.t),
        )
        keys = {
            "step",
            "t",
            "stepTime_ms",
            "maxVelocity",
            "kineticEnergy",
            "minDensity",
            "maxDensity",
            "nPenetrating",
            "maxPenetrationDx",
        }
        measurements["diagnostics"] = (
            [{k: v for k, v in row.items() if k in keys} for row in result.trajectory]
            if args.compact_diagnostics
            else result.trajectory
        )
        if args.checkpoint:
            import h5py
            from warpSPH.io.hdf5 import dumpState

            with h5py.File(out / "checkpoint.h5", "w") as file:
                file.attrs["time"] = float(result.state.t)
                dumpState(result.state.state, file.create_group("state"))
        save_snapshot(
            out / "final_state.npz", result.state.state, result.ctx.config.domain
        )
    except Exception as exc:
        measurements.update(
            error=f"{type(exc).__name__}: {exc}", wall_s=time.perf_counter() - start
        )
        raise
    finally:
        for p in plotters:
            p.close()
        (out / "measurements.json").write_text(
            json.dumps(measurements, indent=2, default=float)
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in measurements.items()
                if k not in ("diagnostics", "frames")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

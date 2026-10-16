"""Replay temporal frontend snapshots and measure renderer lifecycle stability.

Example: python examples/stress_replay.py --backend pyvista --input reports/long_frontend --frames 2000
This measures repeated rendering, not new physical time integration.
"""

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from _artifacts import evidence_dir
import time
import numpy as np
import torch
from PIL import Image
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import (
    visualize,
    PlottingOptions,
    GridVisualization,
    Mapping,
    VisualizeOptions,
    PlotScaling,
    waitForExports,
)
from _verification_metrics import resident_bytes, scene_counts


def load(path):
    with np.load(path) as data:
        p = ParticleState(
            **{
                n: torch.from_numpy(data[n].copy())
                for n in ("positions", "supports", "masses", "densities", "kinds")
            }
        )
        v = torch.from_numpy(data["velocities"].copy())
        d = DomainDescription(
            torch.from_numpy(data["domain_min"].copy()),
            torch.from_numpy(data["domain_max"].copy()),
            torch.from_numpy(data["periodic"].copy()),
            2,
        )
    return p, v, d


def camera_state(plot):
    if plot.backend == "matplotlib":
        return [
            tuple(s.axis.get_xlim()) + tuple(s.axis.get_ylim())
            for s in plot.plotStates.values()
        ]
    if plot.backend == "pyvista":
        return [
            tuple(r.camera.position)
            + tuple(r.camera.focal_point)
            + (r.camera.parallel_scale,)
            + tuple(r.camera.GetWindowCenter())
            for r in plot.fig.renderers
        ]
    return [
        (
            s.view.camera.rect.left,
            s.view.camera.rect.right,
            s.view.camera.rect.bottom,
            s.view.camera.rect.top,
        )
        for s in plot.plotStates.values()
    ]


def render(plot):
    if plot.backend == "matplotlib":
        plot.fig.canvas.draw()
        return np.asarray(plot.fig.canvas.buffer_rgba()).copy()
    if plot.backend == "pyvista":
        plot.fig.render()
        return plot.fig.screenshot(return_img=True)
    pixels = plot.fig.render()
    if plot._backend_instance._refresh_axes():
        pixels = plot.fig.render()
    return pixels


def visible_colorbars(plot):
    if plot.backend == "matplotlib":
        return tuple(
            any(
                result is not None and result[1] is not None
                for result in (
                    s.gridResult,
                    s.fluidScatterResult,
                    s.boundaryScatterResult,
                )
            )
            for s in plot.plotStates.values()
        )
    return tuple(s.colorbar is not None for s in plot.plotStates.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["matplotlib", "pyvista", "vispy"], required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Reuse the exact source ordering from a previous measurements.json.",
    )
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument(
        "--trace-memory",
        action="store_true",
        help="Trace Python allocations; timings include tracing overhead.",
    )
    parser.add_argument("--output", type=Path, default=evidence_dir("stress"))
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("--frames must exceed --warmup")
    torch.set_num_threads(2)
    trace_start = None
    if args.trace_memory:
        import tracemalloc

        tracemalloc.start(10)
    import warp as wp

    wp.init()
    files = sorted(args.input.glob("**/snapshots/state_*.npz"))
    if not files:
        parser.error(
            "No temporal snapshots found; run verify_frontends.py --snapshots first."
        )
    # Temporal blocks exercise value-only updates as well as case/domain changes.
    cases = defaultdict(list)
    for f in files:
        cases[f.parent.parent.name].append(f)
    ordered = [
        p
        for i in range(0, max(map(len, cases.values())), 25)
        for v in cases.values()
        for p in v[i : i + 25]
    ]
    if args.source_manifest:
        ordered = [
            Path(p) for p in json.loads(args.source_manifest.read_text())["sources"]
        ]
    out = args.output / args.backend
    (out / "frames").mkdir(parents=True, exist_ok=True)
    p, v, d = load(ordered[0])
    opts = {
        "A": PlottingOptions(
            mapping=Mapping.L2,
            markerSize=2,
            plotTitle="Speed",
            boundaryVisualization=VisualizeOptions.Passive,
        ),
        "B": PlottingOptions(
            markerSize=2,
            plotTitle="Density",
            quantityScaling=PlotScaling.Symmetric,
            midPoint="median",
        ),
    }
    beopts = (
        {"jupyter_backend": "image", "app_backend": "egl"}
        if args.backend == "vispy"
        else ({"jupyter_backend": "static"} if args.backend == "pyvista" else {})
    )
    plot = visualize(
        p,
        d,
        {"A": v, "B": p.densities},
        opts,
        mosaic="AB",
        figsize=(8, 4),
        backend=args.backend,
        backendOptions=beopts,
        figTitle="Temporal renderer stress replay",
    )
    frames, samples, errors = [], [], []
    camera_checks = 0
    counts = defaultdict(list)
    previous_domain = previous_mode = None
    started = time.perf_counter()
    try:
        for i in range(args.frames):
            path = ordered[i % len(ordered)]
            p, v, d = load(path)
            # A full cycle creates/removes grids, streamlines and bars, and
            # alternates active/passive walls. State is reused throughout.
            mode = (i // 125) % 4
            grid_a = (
                GridVisualization(resolution=24, streamLines=True)
                if mode == 1
                else None
            )
            grid_b = GridVisualization(resolution=24) if mode == 1 else None
            new_opts = {
                "A": dict(
                    gridVisualization=grid_a,
                    showColorBar=mode != 2,
                    boundaryVisualization=VisualizeOptions.Passive,
                ),
                "B": dict(
                    gridVisualization=grid_b,
                    showColorBar=mode != 2,
                    boundaryVisualization=(
                        VisualizeOptions.Passive
                        if mode == 3
                        else VisualizeOptions.Visualize
                    ),
                ),
            }
            domain_key = (
                tuple(d.min.tolist()),
                tuple(d.max.tolist()),
                tuple(d.periodic.tolist()),
            )
            before = camera_state(plot)
            bars_before = visible_colorbars(plot)
            tick = time.perf_counter()
            plot.updateQuantities(
                {"A": v, "B": p.densities},
                newParticleState=p,
                newDomain=d,
                newOptions=new_opts,
                redraw=False,
            )
            prepared = time.perf_counter()
            after = camera_state(plot)
            bars_after = visible_colorbars(plot)
            if (
                previous_domain == domain_key
                and previous_mode == mode
                and bars_before == bars_after
            ):
                camera_checks += 1
                if not np.allclose(before, after, atol=1e-7):
                    errors.append(dict(frame=i, issue="camera_changed_on_value_update"))
            pixels = render(plot)
            ended = time.perf_counter()
            frame = dict(
                frame=i,
                source=str(path),
                mode=mode,
                prepare_ms=1000 * (prepared - tick),
                render_ms=1000 * (ended - prepared),
                total_ms=1000 * (ended - tick),
            )
            frames.append(frame)
            if i >= args.warmup:
                counts[(path.parent.parent.name, mode, bars_after)].append(
                    scene_counts(plot)
                )
            if i % 100 == 0 or i == args.frames - 1:
                gc.collect()
                sample = dict(
                    frame=i, rss_bytes=resident_bytes(), scene_counts=scene_counts(plot)
                )
                samples.append(sample)
                if args.trace_memory:
                    from collections import Counter

                    sample["python_vispy_types"] = dict(
                        Counter(
                            type(o).__name__
                            for o in gc.get_objects()
                            if str(type(o).__module__).startswith("vispy.")
                        )
                    )
                    if trace_start is None and i >= args.warmup:
                        trace_start = tracemalloc.take_snapshot()
                print(json.dumps(sample), flush=True)
                target = out / "frames" / f"frame_{i:05d}.png"
                if args.backend == "vispy":
                    plot.export(str(target), background=True)
                else:
                    plot.export(str(target))
                (out / "progress.json").write_text(
                    json.dumps(dict(samples=samples, frames_completed=i + 1), indent=2)
                )
            previous_domain, previous_mode = domain_key, mode
        waitForExports()
        for path in sorted((out / "frames").glob("*.png")):
            with Image.open(path) as im:
                im.verify()
        # Patch counts naturally vary with streamline arrow geometry. Axes,
        # collections, actors and scene nodes must remain bounded per mode/case.
        count_ranges = {}
        for key, values in counts.items():
            ranges = {
                name: [min(v[name] for v in values), max(v[name] for v in values)]
                for name in values[0]
            }
            count_ranges[str(key)] = ranges
            for name, (lo, hi) in ranges.items():
                # Streamline geometry may legitimately be empty for a field;
                # its optional PyVista actor is counted separately.
                dynamic = ("patches", "actors", "streamline_actors")
                if name not in dynamic and hi > lo:
                    errors.append(
                        dict(
                            case_mode=str(key),
                            issue="scene_count_growth",
                            kind=name,
                            bounds=[lo, hi],
                        )
                    )
        warm = [
            s
            for s in samples
            if s["frame"] >= args.warmup and s["rss_bytes"] is not None
        ]
        growth = (
            (warm[-1]["rss_bytes"] - warm[0]["rss_bytes"]) / 2**20 if warm else None
        )
        slope = (
            float(
                np.polyfit(
                    [s["frame"] for s in warm],
                    [s["rss_bytes"] / 2**20 for s in warm],
                    1,
                )[0]
                * 1000
            )
            if len(warm) > 1
            else None
        )
        result = dict(
            backend=args.backend,
            frames=args.frames,
            warmup=args.warmup,
            sources=[str(p) for p in ordered],
            cases=list(cases),
            wall_s=time.perf_counter() - started,
            median_ms=float(np.median([f["total_ms"] for f in frames[args.warmup :]])),
            p95_ms=float(
                np.percentile([f["total_ms"] for f in frames[args.warmup :]], 95)
            ),
            rss_growth_after_warmup_mib=growth,
            rss_slope_mib_per_1000_frames=slope,
            count_ranges=count_ranges,
            errors=errors,
            camera_checks=camera_checks,
            samples=samples,
            frame_timings=frames,
        )
        if args.trace_memory and trace_start is not None:
            diff = tracemalloc.take_snapshot().compare_to(trace_start, "lineno")
            result["python_traced_growth_mib"] = sum(d.size_diff for d in diff) / 2**20
            result["python_allocation_diff"] = [str(d) for d in diff[:30]]
        (out / "measurements.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("sources", "samples", "frame_timings", "count_ranges")
                },
                indent=2,
            )
        )
    finally:
        waitForExports()
        plot.close()


if __name__ == "__main__":
    main()

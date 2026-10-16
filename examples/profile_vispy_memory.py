"""Isolate Vispy marker/grid memory trends with a fixed real frontend state."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from warpSPHPlotting import visualize, PlottingOptions, GridVisualization
from stress_replay import load, render
from _verification_metrics import resident_bytes, scene_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=["markers", "grid"], required=True)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("--frames must exceed --warmup")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()
    torch.set_num_threads(2)
    p, _, d = load(args.input)
    q = p.densities
    grid = GridVisualization(resolution=32) if args.mode == "grid" else None
    plot = visualize(
        p,
        d,
        q,
        PlottingOptions(gridVisualization=grid, markerSize=2),
        mosaic="AB",
        figsize=(8, 4),
        backend="vispy",
        backendOptions={"jupyter_backend": "image", "app_backend": "egl"},
    )
    samples = []
    trace_start = None
    if args.trace_memory:
        import tracemalloc
    started = time.perf_counter()
    try:
        for i in range(args.frames):
            # Change ranges without changing geometry, separating transform and
            # colorbar allocations from legitimate new simulation content.
            plot.updateQuantities(q * (1 + 1e-5 * (i % 50)), redraw=False)
            render(plot)
            if i == args.warmup and args.trace_memory:
                gc.collect()
                tracemalloc.start(10)
                trace_start = tracemalloc.take_snapshot()
            if i % 100 == 0 or i == args.frames - 1:
                gc.collect()
                types = Counter(
                    type(o).__module__ + "." + type(o).__name__
                    for o in gc.get_objects()
                    if str(type(o).__module__).startswith("vispy.")
                )
                sample = dict(
                    frame=i,
                    rss_bytes=resident_bytes(),
                    scene_counts=scene_counts(plot),
                    vispy_types=dict(types),
                )
                samples.append(sample)
                print(
                    json.dumps({k: v for k, v in sample.items() if k != "vispy_types"}),
                    flush=True,
                )
        result = dict(
            mode=args.mode,
            input=str(args.input),
            frames=args.frames,
            warmup=args.warmup,
            elapsed_s=time.perf_counter() - started,
            samples=samples,
        )
        warm = next(s for s in samples if s["frame"] >= args.warmup)
        result["rss_growth_after_warmup_mib"] = (
            samples[-1]["rss_bytes"] - warm["rss_bytes"]
        ) / 2**20
        result["object_growth"] = {
            k: v - warm["vispy_types"].get(k, 0)
            for k, v in samples[-1]["vispy_types"].items()
            if v != warm["vispy_types"].get(k, 0)
        }
        if trace_start is not None:
            gc.collect()
            stats = tracemalloc.take_snapshot().compare_to(trace_start, "lineno")
            result["python_growth_top"] = [str(s) for s in stats[:25]]
        plot.export(str(args.output.with_suffix(".png")))
    finally:
        plot.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

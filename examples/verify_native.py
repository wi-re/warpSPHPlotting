"""Exercise native window mapping, resizing, camera changes and live updates.

Run each backend in a separate process on a host with a native display.
"""

import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["matplotlib", "pyvista", "vispy"], required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.backend == "matplotlib":
        import matplotlib

        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt

        plt.ion()
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()
    torch.set_num_threads(2)
    from warpSPHCore import DomainDescription
    from warpSPHPlotting import (
        visualize,
        PlottingOptions,
        GridVisualization,
        Mapping,
        VisualizeOptions,
    )
    from stress_replay import load, render

    p, v, d = load(args.input)
    for name in ("positions", "supports", "masses", "densities", "kinds"):
        setattr(p, name, getattr(p, name).to(args.device))
    v = v.to(args.device)
    d = DomainDescription(
        d.min.to(args.device), d.max.to(args.device), d.periodic.to(args.device), 2
    )
    options = (
        {"jupyter_backend": "none", "off_screen": False}
        if args.backend == "pyvista"
        else (
            {"jupyter_backend": "native", "app_backend": "glfw"}
            if args.backend == "vispy"
            else {}
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot = visualize(
        p,
        d,
        v,
        PlottingOptions(
            mapping=Mapping.L2, boundaryVisualization=VisualizeOptions.Passive
        ),
        backend=args.backend,
        backendOptions=options,
        figsize=(7, 5),
        figTitle=f"Native {args.backend} verification",
    )
    try:

        def events():
            if args.backend == "matplotlib":
                plt.show(block=False)
                plot.fig.canvas.flush_events()
            elif args.backend == "vispy":
                plot.fig.app.process_events()
            else:
                plot.fig.update()

        events()
        initial = render(plot).astype(float)
        plot.export(str(args.output.with_name(f"native_{args.backend}_before.png")))
        if args.backend == "matplotlib":
            mapped = bool(plot.fig.canvas.manager.window.winfo_ismapped())
            plot.fig.set_size_inches(9, 6, forward=True)
            for ax in plot.axes.values():
                lo, hi = ax.get_xlim()
                c = (lo + hi) / 2
                half = (hi - lo) / 3
                ax.set_xlim(c - half, c + half)
        elif args.backend == "pyvista":
            mapped = bool(plot.fig.render_window.GetMapped())
            plot.fig.window_size = (900, 600)
            plot.fig.camera.zoom(1.5)
        else:
            import glfw

            mapped = bool(glfw.get_window_attrib(plot.fig.native._id, glfw.VISIBLE))
            plot.fig.size = (900, 600)
            plot.plotStates["A"].view.camera.zoom(2 / 3)
        assert mapped
        plot.updateQuantities(
            v * 1.25,
            newOptions={
                "A": {
                    "gridVisualization": GridVisualization(
                        resolution=64, streamLines=True
                    )
                }
            },
            redraw=True,
        )
        events()
        final = render(plot)
        plot.export(str(args.output.with_name(f"native_{args.backend}_after.png")))
        assert final.shape[:2] != initial.shape[:2]
        assert (final[..., :3] < 200).any()
        result = dict(
            backend=args.backend,
            data_device=str(p.positions.device),
            window_mapped=mapped,
            initial_shape=list(initial.shape),
            final_shape=list(final.shape),
            resize_camera_and_grid_update=True,
            manual_mouse_keyboard_verified=False,
        )
        if args.backend == "pyvista":
            result["renderer"] = plot.fig.render_window.ReportCapabilities()
        elif args.backend == "vispy":
            from vispy import gloo

            result["renderer"] = str(gloo.gl.glGetParameter(gloo.gl.GL_RENDERER))
        args.output.write_text(json.dumps(result, indent=2))
        print(
            json.dumps({k: v for k, v in result.items() if k != "renderer"}), flush=True
        )
    finally:
        plot.close()
        if args.backend == "matplotlib":
            plt.ioff()


if __name__ == "__main__":
    main()

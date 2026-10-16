"""Compare fixed/automatic markers and replay saved frontend snapshots.

python examples/particle_sizes.py --backend pyvista \
  --input reports/long_frontend/marrone31-pyvista --output reports/particle_sizes --video
No simulation is rerun: all images use the saved physical particle states.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from warpSPHPlotting import (
    visualize,
    PlottingOptions,
    Mapping,
    VisualizeOptions,
    PlotScaling,
)
from stress_replay import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["matplotlib", "pyvista", "vispy"], required=True
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Frontend run directory containing final_state.npz and snapshots/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    p, v, d = load(args.input / "final_state.npz")
    beopts = (
        {"jupyter_backend": "image", "app_backend": "egl"}
        if args.backend == "vispy"
        else {}
    )
    opts = {
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
    plot = visualize(
        p,
        d,
        {"A": v, "B": p.densities},
        opts,
        mosaic="AB",
        figsize=(11, 5),
        backend=args.backend,
        backendOptions=beopts,
        figTitle=f"{args.backend} · fixed markerSize=36",
    )
    try:
        # sqrt(36) points * 96 / 72 = 8 display pixels.
        plot.updateQuantities(
            {"A": v, "B": p.densities},
            newOptions={key: {"markerSize": 36} for key in opts},
            redraw=False,
        )
        plot.export(str(args.output / f"{args.backend}_fixed.png"))
        plot.updateQuantities(
            {"A": v, "B": p.densities},
            newOptions={key: {"markerSize": None} for key in opts},
            redraw=False,
        )
        plot.updateTitle(f"{args.backend} · automatic particle sizes")
        plot.export(str(args.output / f"{args.backend}_auto.png"))
        plot.updateQuantities(
            {"A": v, "B": p.densities},
            newOptions={key: {"plotTitle": None} for key in opts},
            redraw=False,
        )
        # Zoom into the right-hand wall and lower free surface using real camera
        # controls. Particle positions and the physical domain are unchanged.
        center = (d.min + (d.max - d.min) * torch.tensor([0.95, 0.30])).numpy()
        for panel in plot.plotStates.values():
            if args.backend == "matplotlib":
                panel.axis.set_xlim(center[0] - 0.45, center[0] + 0.45)
                panel.axis.set_ylim(center[1] - 0.4, center[1] + 0.4)
            elif args.backend == "pyvista":
                plot.fig.subplot(panel.row, panel.col)
                camera = plot.fig.camera
                focal = np.r_[center, 0.0]
                offset = np.asarray(camera.position) - np.asarray(camera.focal_point)
                camera.focal_point = focal
                camera.position = focal + offset
                camera.parallel_scale = 0.45
            else:
                panel.view.camera.set_range(
                    x=(center[0] - 0.45, center[0] + 0.45),
                    y=(center[1] - 0.4, center[1] + 0.4),
                    margin=0,
                )
        plot.updateTitle(
            f"{args.backend} · wall detail · speed (left), density (right)"
        )
        plot.export(str(args.output / f"{args.backend}_zoom.png"))
    finally:
        plot.close()

    if not args.video:
        return
    measurements = json.loads((args.input / "measurements.json").read_text())
    times = {f["step"]: f["time"] for f in measurements["frames"]}
    paths = sorted((args.input / "snapshots").glob("state_*.npz"))
    frames = args.output / f"{args.backend}_frames"
    frames.mkdir(exist_ok=True)
    plot = None
    try:
        for i, path in enumerate(paths):
            p, v, d = load(path)
            step = int(path.stem.split("_")[-1])
            title = f"Marrone 3.1 · t = {times[step]:.4f} · automatic particle sizes"
            if plot is None:
                plot = visualize(
                    p,
                    d,
                    {"A": v, "B": p.densities},
                    opts,
                    mosaic="AB",
                    figsize=(11, 5),
                    backend=args.backend,
                    backendOptions=beopts,
                    figTitle=title,
                )
            else:
                plot.updateQuantities(
                    {"A": v, "B": p.densities},
                    newParticleState=p,
                    newDomain=d,
                    redraw=False,
                )
                plot.updateTitle(title)
            plot.export(str(frames / f"frame_{i:05d}.png"))
    finally:
        if plot is not None:
            plot.close()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Frames saved in {frames}; install ffmpeg to encode the movie."
        )
    movie = args.output / f"{args.backend}_auto.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-framerate",
            "12",
            "-i",
            str(frames / "frame_%05d.png"),
            "-frames:v",
            str(len(paths)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
    )
    # Decode the entire clip, checking for corrupt frames as well as its count.
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(movie),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    check = json.loads(result.stdout)
    assert int(check["streams"][0]["nb_read_frames"]) == len(paths)
    (args.output / f"{args.backend}_video_check.json").write_text(
        json.dumps(check, indent=2)
    )


if __name__ == "__main__":
    main()

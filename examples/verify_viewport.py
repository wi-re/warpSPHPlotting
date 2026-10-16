"""Check camera retention with actual Sod and Marrone frontend recordings."""

import argparse
import json
from pathlib import Path
from _artifacts import evidence_dir

import numpy as np
import torch
import warp as wp

from verify_notebook_camera import load_frame
from warpSPHPlotting import GridVisualization, Mapping, PlottingOptions, visualize


def camera(plot):
    if plot.backend == "matplotlib":
        ax = plot.axes["A"]
        return np.array([*ax.get_xlim(), *ax.get_ylim()])
    if plot.backend == "vispy":
        r = plot.plotStates["A"].view.camera.rect
        return np.array([r.left, r.bottom, r.width, r.height])
    return np.r_[
        np.asarray(list(plot.fig.camera_position)).ravel(),
        plot.fig.camera.parallel_scale,
        plot.fig.camera.GetWindowCenter(),
    ]


def zoom(plot):
    if plot.backend == "matplotlib":
        ax = plot.axes["A"]
        for get, set_ in ((ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)):
            lo, hi = get()
            mid, half = (lo + hi) / 2, (hi - lo) / 3
            set_(mid - half, mid + half)
    elif plot.backend == "vispy":
        plot.plotStates["A"].view.camera.zoom(2 / 3)
    else:
        plot.fig.camera.zoom(1.5)


def verify(args):
    wp.init()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for backend in ("matplotlib", "pyvista", "vispy"):
        extra = (
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        )
        for case, frames in (("sod", (0, 11, 40)), ("marrone31", (0, 25, 50))):
            p, domain, v = load_frame(args.root, case, 0)
            plot = visualize(
                p,
                domain,
                v,
                PlottingOptions(
                    mapping=Mapping.L2,
                    plotTitle="Speed",
                    gridVisualization=GridVisualization(
                        resolution=96, streamLines=case == "marrone31"
                    ),
                ),
                backend=backend,
                backendOptions=extra,
                figsize=(9, 5),
                figTitle=f"{case} · camera retention",
            )
            try:
                zoom(plot)
                moved = camera(plot)
                for frame in frames:
                    p, actual, v = load_frame(args.root, case, frame)
                    # Sod recordings animate a stress-test display viewport.
                    # Match the notebook's fixed initial viewport by default.
                    source = domain if case == "sod" else actual
                    plot.updateQuantities(
                        v,
                        newParticleState=p,
                        newDomain=source,
                        redraw=False,
                        yieldSeconds=0,
                    )
                    np.testing.assert_array_equal(camera(plot), moved)
                    image = f"{backend}-{case}-{frame:04d}-retained.png"
                    plot.export(str(args.output / image))
                    records.append(
                        dict(
                            backend=backend,
                            case=case,
                            frame=frame,
                            retained=True,
                            camera=moved.tolist(),
                            image=image,
                        )
                    )
                # A new case with identical bounds must still support a refit.
                plot.updateQuantities(v, resetView=True, redraw=False, yieldSeconds=0)
                assert not np.allclose(camera(plot), moved)
                plot.export(str(args.output / f"{backend}-{case}-refit.png"))
                zoom(plot)
                moved = camera(plot)
                plot.updateQuantities(
                    v, dynamicBounds=True, redraw=False, yieldSeconds=0
                )
                assert not np.allclose(camera(plot), moved)
                records.append(
                    dict(
                        backend=backend,
                        case=case,
                        explicit_refit=True,
                        dynamic_refit=True,
                    )
                )
            finally:
                plot.close()
    (args.output / "verification.json").write_text(json.dumps(records, indent=2) + "\n")
    print("18 retained-camera frame checks and 12 requested refits passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, default=evidence_dir("viewport/replay"))
    verify(parser.parse_args())

"""Compare physical and neighbor-limited sizing over the real Marrone recording."""

import argparse
import json
from pathlib import Path
from _artifacts import evidence_dir, recording_paths

import numpy as np
import torch
import warp as wp

from verify_notebook_camera import load_frame
from warpSPHPlotting import Mapping, PlottingOptions, VisualizeOptions, visualize


def diameters(plot, key):
    panel = plot.plotStates[key]
    if plot.backend == "matplotlib":
        return {
            "fluid": panel.fluidScatterResult[0].world_diameter,
            "boundary": panel.boundaryScatterResult[0].world_diameter,
        }
    if plot.backend == "pyvista":
        return dict(panel.particleDiameters)
    return {
        "fluid": float(panel.markers_fluid._data["a_size"][0]),
        "boundary": float(panel.markers_boundary._data["a_size"][0]),
    }


def verify(args):
    wp.init()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    count = len(recording_paths("marrone31", root=args.root))
    if not count:
        raise FileNotFoundError("Missing Marrone frontend recording")
    records, summaries = [], []
    for backend in ("matplotlib", "pyvista", "vispy"):
        options = {
            key: PlottingOptions(
                mapping=Mapping.L2,
                markerSpacing=mode,
                boundaryVisualization=VisualizeOptions.Passive,
                plotTitle=title,
                vMin=0,
                vMax=5,
            )
            for key, mode, title in (
                ("A", "volume", "Physical volume (default)"),
                ("B", "neighbors", "Close-neighbor limit (optional)"),
            )
        }
        p, domain, v = load_frame(args.root, "marrone31", 0)
        extra = (
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        )
        plot = visualize(
            p,
            domain,
            v,
            options,
            mosaic="AB",
            figsize=(12, 5),
            backend=backend,
            backendOptions=extra,
            figTitle=f"Marrone 3.1 · {backend}",
        )
        initial = {k: diameters(plot, k) for k in options}
        try:
            final = None
            for frame in range(count):
                p, d, v = load_frame(args.root, "marrone31", frame)
                plot.updateQuantities(
                    v, newParticleState=p, newDomain=d, redraw=False, yieldSeconds=0
                )
                sizes = {k: diameters(plot, k) for k in options}
                for kind in ("fluid", "boundary"):
                    assert abs(sizes["A"][kind] / initial["A"][kind] - 1) < 0.001
                records.append(dict(backend=backend, frame=frame, sizes=sizes))
                if frame in (0, 25, count - 1):
                    plot.updateTitle(f"Marrone 3.1 · {backend} · frame {frame}")
                    plot.export(str(args.output / f"{backend}-frame-{frame:04d}.png"))
                final = sizes
            assert 0.25 < final["B"]["fluid"] / initial["B"]["fluid"] < 0.3
            # Revisit both ends: sizes depend on the current physical state,
            # rather than accumulating shrinkage in the renderer.
            for frame, expected in ((0, initial), (count - 1, final)):
                p, d, v = load_frame(args.root, "marrone31", frame)
                plot.updateQuantities(
                    v, newParticleState=p, newDomain=d, redraw=False, yieldSeconds=0
                )
                for key in options:
                    for kind, value in diameters(plot, key).items():
                        assert value == expected[key][kind]
            summaries.append(
                dict(
                    backend=backend,
                    frames=count,
                    physical_fluid_ratio=final["A"]["fluid"] / initial["A"]["fluid"],
                    neighbor_fluid_ratio=final["B"]["fluid"] / initial["B"]["fluid"],
                    physical_boundary_ratio=final["A"]["boundary"]
                    / initial["A"]["boundary"],
                    round_trip_exact=True,
                )
            )
        finally:
            plot.close()
    result = dict(summaries=summaries, frames=records)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    data = [r for r in records if r["backend"] == "pyvista"]
    for key, kind, label, style in (
        ("A", "fluid", "Fluid · physical volume", "-"),
        ("B", "fluid", "Fluid · neighbor limit", "-"),
        ("A", "boundary", "Boundary · physical volume", "--"),
    ):
        values = np.array([r["sizes"][key][kind] for r in data])
        ax.plot([r["frame"] for r in data], values / values[0], style, label=label)
    ax.set(
        xlabel="Saved Marrone frame",
        ylabel="Diameter / initial diameter",
        ylim=(0, 1.1),
    )
    ax.legend()
    fig.savefig(args.output / "diameter-history.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output", type=Path, default=evidence_dir("physical-sizing/replay")
    )
    verify(parser.parse_args())

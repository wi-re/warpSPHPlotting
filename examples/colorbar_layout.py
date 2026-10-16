"""Render narrow density, logarithmic, and signed symmetric-log colorbars."""

import argparse
from pathlib import Path
from _artifacts import evidence_dir
import torch
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import visualize, PlottingOptions, PlotScaling, ColorMap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="pyvista", choices=["matplotlib", "pyvista", "vispy"]
    )
    parser.add_argument(
        "--output", type=Path, default=evidence_dir("colorbar_layout.png")
    )
    parser.add_argument("--rotation", type=float)
    parser.add_argument(
        "--orientation", choices=["vertical", "horizontal"], default="vertical"
    )
    args = parser.parse_args()
    x = (torch.arange(8) + 0.5) / 8
    points = torch.stack(torch.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    p = ParticleState(
        points,
        torch.ones(64),
        torch.ones(64) / 64,
        torch.zeros(64, dtype=torch.int32),
        torch.ones(64),
    )
    domain = DomainDescription(
        torch.zeros(2), torch.ones(2), torch.zeros(2, dtype=torch.bool), 2
    )
    quantities = {
        "A": torch.linspace(0.997, 1.003, 64),
        "B": torch.logspace(-3, 3, 64),
        "C": torch.linspace(-10, 10, 64),
    }
    options = {
        "A": PlottingOptions(
            quantityScaling=PlotScaling.Symmetric,
            midPoint=1,
            plotTitle="Density around 1",
            markerSize=20,
        ),
        "B": PlottingOptions(
            quantityScaling=PlotScaling.Logarithmic,
            plotTitle="Logarithmic field",
            markerSize=20,
        ),
        "C": PlottingOptions(
            quantityScaling=PlotScaling.SymmetricLog,
            colorMap=ColorMap.RdBu,
            plotTitle="Signed symmetric-log field",
            markerSize=20,
        ),
    }
    backend_options = (
        {
            "scalar_bar": {
                "orientation": args.orientation,
                **(
                    {"label_rotation": args.rotation}
                    if args.rotation is not None
                    else {}
                ),
                "n_labels": 3,
                "font_size": 12,
            }
        }
        if args.backend == "pyvista"
        else (
            {"jupyter_backend": "image", "app_backend": "egl"}
            if args.backend == "vispy"
            else {}
        )
    )
    s = visualize(
        p,
        domain,
        quantities,
        options,
        mosaic="ABC",
        figsize=(14, 5),
        backend=args.backend,
        backendOptions=backend_options,
        figTitle="Physical colorbar labels · density, logarithmic and signed fields",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        s.export(str(args.output))
        if args.backend == "pyvista":
            s.fig.window_size = (1100, 420)
        elif args.backend == "vispy":
            s.fig.size = (1100, 420)
        else:
            s.fig.set_size_inches(11, 4.2)
        s.updateTitle("Resized figure · updated physical labels")
        s.updateQuantities({k: v * 1.001 for k, v in quantities.items()}, redraw=False)
        s.export(str(args.output.with_name(args.output.stem + "_resized.png")))
    finally:
        s.close()


if __name__ == "__main__":
    main()

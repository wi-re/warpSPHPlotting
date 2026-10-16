"""Compare CPU/CUDA interpolation and coverage on saved frontend states."""

import argparse
import json
from pathlib import Path
import torch
from warpSPHCore import DomainDescription, ParticleState
from warpSPHPlotting import PlottingOptions, GridVisualization, VisualizeOptions
from warpSPHPlotting.backends._render_util import prepare_frame
from warpSPHPlotting.grid import mapFrameToGrid
from stress_replay import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    checks = []
    for case in ("tgv", "sod", "marrone31", "khh"):
        p, _, d = load(args.input / f"{case}-pyvista/final_state.npz")
        if case == "sod":
            c = (d.min + d.max) / 2
            h = (d.max - d.min) / 2 * 7.5
            d = DomainDescription(c - h, c + h, torch.zeros_like(d.periodic), 2)
        outputs = []
        for device in ("cpu", "cuda"):
            particles = ParticleState(
                **{
                    name: getattr(p, name).to(device)
                    for name in (
                        "positions",
                        "supports",
                        "masses",
                        "densities",
                        "kinds",
                    )
                }
            )
            domain = DomainDescription(
                d.min.to(device), d.max.to(device), d.periodic.to(device), 2
            )
            frame = prepare_frame(
                particles,
                domain,
                particles.densities,
                PlottingOptions(
                    gridVisualization=GridVisualization(resolution=192),
                    boundaryVisualization=VisualizeOptions.Passive,
                ),
            )
            _, values, _, extent = mapFrameToGrid(frame, frame.particles.quantities)
            outputs.append((values.cpu(), extent["coverage"].cpu()))
        cpu, gpu = outputs
        torch.testing.assert_close(cpu[0], gpu[0], rtol=2e-4, atol=2e-5)
        diff = (cpu[0] - gpu[0]).abs()
        mask_diff = cpu[1] != gpu[1]
        if mask_diff.any():
            # Positive density fields can differ at a roundoff-level kernel
            # cutoff. Significant contributions must have identical coverage.
            assert (
                torch.maximum(cpu[0][mask_diff].abs(), gpu[0][mask_diff].abs()).max()
                < 1e-10
            )
        checks.append(
            dict(
                case=case,
                max_absolute_difference=float(diff.max()),
                mean_absolute_difference=float(diff.mean()),
                coverage_differences=int(mask_diff.sum()),
                samples=len(diff),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(gpu=torch.cuda.get_device_name(), checks=checks), indent=2)
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

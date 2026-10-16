"""Isolate Vispy font-resource churn during repeated colorbar hide/show cycles.

Run in a fresh process; compare measurements across revisions on the same host.
"""

import argparse
import gc
import json
from pathlib import Path
from _artifacts import evidence_dir
import torch
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import visualize, PlottingOptions
from _verification_metrics import resident_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument(
        "--output", type=Path, default=evidence_dir("stress_colorbar_after.json")
    )
    args = parser.parse_args()
    x = torch.linspace(0.1, 0.9, 8)
    p = ParticleState(
        torch.cartesian_prod(x, x),
        torch.ones(64),
        torch.ones(64),
        torch.zeros(64, dtype=torch.int32),
        torch.ones(64),
    )
    d = DomainDescription(
        torch.zeros(2), torch.ones(2), torch.zeros(2, dtype=torch.bool), 2
    )
    q = torch.linspace(0, 1, 64)
    plot = visualize(
        p,
        d,
        q,
        PlottingOptions(),
        mosaic="AB",
        backend="vispy",
        backendOptions={"jupyter_backend": "image", "app_backend": "egl"},
    )
    samples = []
    try:
        for i in range(args.cycles + 1):
            for show in (False, True):
                plot.updateQuantities(
                    q,
                    newOptions={key: {"showColorBar": show} for key in "AB"},
                    redraw=False,
                )
                plot.fig.render()
            if i % 5 == 0 or i == args.cycles:
                gc.collect()
                sample = dict(cycle=i, rss_bytes=resident_bytes())
                samples.append(sample)
                print(json.dumps(sample), flush=True)
    finally:
        plot.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()

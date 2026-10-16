"""Compare marker estimates and full preparation against an archived reference.

Pass --reference examples/_reference_sizing.py. This validates the numerical
calculation over full populations; it does not accept a speed-up from sampling
fewer particles. Both algorithms execute in the same process, alternating order.
"""

import argparse
import cProfile
import importlib.util
import io
import json
from pathlib import Path
from _artifacts import evidence_dir
import pstats
import time
from types import SimpleNamespace
import numpy as np
import torch
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import PlottingOptions, visualize
import warpSPHPlotting._sizing as sizing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("_reference_sizing.py"),
    )
    parser.add_argument("--output", type=Path, default=evidence_dir("notebook"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("reference_sizing", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    old = reference._estimate_particle_spacing
    new = sizing._estimate_particle_spacing
    rng = np.random.default_rng(419)
    equality = []
    for dim in (2, 3):
        for dtype in (np.float32, np.float64):
            for count in (0, 1, 17, 4096, 100000):
                pos = rng.random((count, dim)).astype(dtype)
                for variant in (
                    "regular",
                    "compressed",
                    "invalid",
                    "coincident",
                    "strided",
                ):
                    points = pos.copy()
                    if variant == "compressed":
                        points[:, 0] *= 0.04
                    if variant == "invalid" and count > 1:
                        points[0, 0] = np.nan
                        points[-1, -1] = np.inf
                    if variant == "coincident":
                        points[:] = 0.5
                    if variant == "strided":
                        storage = np.zeros((count, dim * 2), dtype=dtype)
                        storage[:, ::2] = points
                        points = storage[:, ::2]
                    p = SimpleNamespace(
                        positions=torch.from_numpy(points),
                        masses=torch.full(
                            (count,),
                            1 / max(1, count),
                            dtype=(
                                torch.float64 if dtype == np.float64 else torch.float32
                            ),
                        ),
                        densities=torch.ones(count),
                    )
                    a, b = old(p), new(p)
                    if a != b:
                        raise AssertionError((dim, str(dtype), count, variant, a, b))
                    equality.append(
                        {
                            "dim": dim,
                            "dtype": str(dtype),
                            "count": count,
                            "variant": variant,
                            "spacing": b,
                            "exact": a == b,
                        }
                    )
    for path in sorted(evidence_dir("long_frontend").glob("*/final_state.npz")):
        with np.load(path) as data:
            for kind in np.unique(data["kinds"]):
                mask = data["kinds"] == kind
                p = SimpleNamespace(
                    **{
                        n: torch.from_numpy(data[n][mask].copy())
                        for n in ("positions", "masses", "densities")
                    }
                )
                a, b = old(p), new(p)
                assert a == b, (str(path), int(kind), a, b)
                equality.append(
                    {
                        "source": str(path),
                        "kind": int(kind),
                        "spacing": b,
                        "exact": True,
                    }
                )
    (args.output / "sizing_equality.json").write_text(json.dumps(equality, indent=2))
    n = 1000000
    pos = torch.rand((n, 2), generator=torch.Generator().manual_seed(31415))
    p = ParticleState(
        positions=pos,
        supports=torch.full((n,), 0.02),
        masses=torch.full((n,), 1 / n),
        densities=torch.ones(n),
        kinds=torch.zeros(n, dtype=torch.int32),
    )
    d = DomainDescription(
        torch.zeros(2), torch.ones(2), torch.zeros(2, dtype=torch.bool), 2
    )
    q = 0.5 + 0.5 * torch.sin(pos[:, 0] * 2 * torch.pi) * torch.cos(
        pos[:, 1] * 2 * torch.pi
    )
    plot = visualize(
        p,
        d,
        q,
        PlottingOptions(vMin=0, vMax=1, markerSpacing="neighbors"),
        backend="pyvista",
    )
    timings = {"before": [], "after": []}
    try:
        for i in range(23):
            for name, fn in (
                [("before", old), ("after", new)]
                if i % 2 == 0
                else [("after", new), ("before", old)]
            ):
                sizing._estimate_particle_spacing = fn
                start = time.perf_counter()
                plot.updateQuantities(q, newParticleState=p, redraw=False)
                elapsed = 1000 * (time.perf_counter() - start)
                if i >= 3:
                    timings[name].append(elapsed)
        sizing._estimate_particle_spacing = new
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(12):
            plot.updateQuantities(q, newParticleState=p, redraw=False)
        pr.disable()
        pr.dump_stats(str(args.output / "preparation_after.prof"))
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumtime").print_stats(45)
        (args.output / "preparation_after.txt").write_text(buf.getvalue())
    finally:
        sizing._estimate_particle_spacing = new
        plot.close()
    summary = {
        "particles": n,
        "dimension": 2,
        "threads": 2,
        "samples_ms": timings,
        "median_ms": {k: float(np.median(v)) for k, v in timings.items()},
        "exact_spacing_cases": len(equality),
        "all_exact": all(r["exact"] for r in equality),
    }
    (args.output / "preparation_comparison.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(summary["median_ms"], len(equality))


if __name__ == "__main__":
    main()

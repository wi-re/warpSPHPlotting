"""Regenerate the four recorded frontend trajectories used by review examples.

Requires the optional sibling warpSPH frontend and plotting dependencies.
These coarse trajectories check visualization; they do not establish physical
convergence. Each case runs in a separate process. --dry-run prints commands.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from _artifacts import evidence_dir

CASES = {
    "tgv": (24, 2000, 25),
    "sod": (24, 400, 10),
    "khh": (48, 1200, 30),
    "marrone31": (33, 5000, 100),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--backend", choices=["matplotlib", "pyvista", "vispy"], default="pyvista"
    )
    parser.add_argument("--output", type=Path, default=evidence_dir("long_frontend"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    driver = Path(__file__).with_name("verify_frontends.py")
    for case in args.cases:
        nx, steps, interval = CASES[case]
        command = [
            sys.executable,
            str(driver),
            "--case",
            case,
            "--backend",
            args.backend,
            "--device",
            args.device,
            "--nx",
            str(nx),
            "--steps",
            str(steps),
            "--plot-interval",
            str(interval),
            "--snapshots",
            "--checkpoint",
            "--compact-diagnostics",
            "--output",
            str(args.output),
        ]
        if case == "sod":
            command.append("--sparse-view")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

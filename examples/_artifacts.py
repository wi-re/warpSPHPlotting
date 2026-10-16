"""Resolve development evidence outside the source checkout without creating it."""

import os
from pathlib import Path


def development_dir(root=None):
    repo = Path(root or Path(__file__).resolve().parents[1]).resolve()
    shared = os.environ.get("WARPSPH_DEVELOPMENT_ROOT")
    base = Path(shared).expanduser() if shared else repo.parent / "warpSPHDevelopment"
    return base.resolve() / "plotting"


def evidence_dir(*parts, root=None):
    return development_dir(root) / "reports" / Path(*parts)


def recording_paths(case, root=None):
    return sorted(
        evidence_dir("long_frontend", f"{case}-pyvista", "snapshots", root=root).glob(
            "state_*.npz"
        )
    )

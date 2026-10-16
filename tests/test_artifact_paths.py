"""Keep generated development evidence outside the source checkout."""

import importlib
from pathlib import Path
import subprocess
import sys


def test_external_evidence_paths_and_recording_discovery(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    artifacts = importlib.import_module("_artifacts")
    monkeypatch.delenv("WARPSPH_DEVELOPMENT_ROOT", raising=False)
    repo = tmp_path / "checkout"
    expected = tmp_path / "warpSPHDevelopment" / "plotting"
    assert artifacts.development_dir(repo) == expected
    assert not expected.exists()  # Import/discovery must not generate outputs.
    shared = tmp_path / "relocated archive"
    monkeypatch.setenv("WARPSPH_DEVELOPMENT_ROOT", str(shared))
    snapshots = shared / "plotting/reports/long_frontend/sod-pyvista/snapshots"
    snapshots.mkdir(parents=True)
    for frame in (10, 0):
        (snapshots / f"state_{frame:05d}.npz").touch()
    assert artifacts.recording_paths("sod", root=repo) == [
        snapshots / "state_00000.npz",
        snapshots / "state_00010.npz",
    ]
    assert (
        artifacts.evidence_dir("new-run", root=repo)
        == shared / "plotting/reports/new-run"
    )
    assert not repo.exists()


def test_frontend_regeneration_dry_run_does_not_create_artifacts(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    shared = tmp_path / "archive"
    monkeypatch.setenv("WARPSPH_DEVELOPMENT_ROOT", str(shared))
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "examples/regenerate_frontends.py"),
            "--dry-run",
            "--cases",
            "sod",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(shared / "plotting/reports/long_frontend") in result.stdout
    assert "--steps 400" in result.stdout and "--sparse-view" in result.stdout
    assert not shared.exists()

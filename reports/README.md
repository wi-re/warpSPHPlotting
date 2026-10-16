# Verification notes and reproducible evidence

This directory contains written findings, measured summaries and limitations.
Generated screenshots, videos, simulation states/checkpoints, profiles, raw
measurements, package builds and run logs are excluded from Git.

The original evidence was moved with SHA-256 verification to the sibling
`../warpSPHDevelopment/plotting/` directory. Its `reports/` and `output/` trees
retain the original relative paths; `manifests/migration-2026-10-16.json` records
the migration. This development archive is managed and backed up separately.

Examples and the review notebook now default to that external project directory.
Set `WARPSPH_DEVELOPMENT_ROOT` to relocate the shared archive; tools append
`plotting` themselves. Resolving a path does not create directories or run a
simulation. Required `--input`/`--output` flags continue to accept explicit paths.
Historical figure and raw-data references below are archive-relative filenames,
not missing files to download from this repository.

## Reproduce

Install the plotting dependencies from the repository root. Real trajectories
also require the optional sibling warpSPH frontend and its bootstrap environment.
Without recordings, the interactive notebook offers an explicitly labeled
analytical Taylor–Green fallback. Use this command to regenerate the four
recordings used by the replay checks:

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
pip install -e '.[dev,plot-all]'
python examples/regenerate_frontends.py --dry-run
python examples/regenerate_frontends.py
```

The regeneration script records the case resolutions, step counts and snapshot
intervals used in the long-run reports. These coarse simulations check plotting;
they do not establish physical convergence or experimental agreement. Regenerated
outputs on another revision/runtime are new evidence, not byte-identical archival
results. Keep significant historical runs under distinct output directories.

Renderer-only examples and regressions need no saved frontend data:

```bash
python examples/quickstart.py --backend matplotlib
python examples/colorbar_layout.py --backend pyvista
python examples/three_dimensions.py --backend pyvista
pytest -q tests
```

After generating or restoring recordings:

```bash
python examples/verify_viewport.py
python examples/verify_marrone_sizing.py
python examples/verify_vispy_notebook.py
python examples/verify_notebook_camera.py --with-vispy
python examples/verify_sizing_preparation.py
```

Notebook renderers need their widget/trame dependencies and a working graphics
context. Native GLFW tests explicitly skip when no display is available. Use
`EGL_PLATFORM=surfaceless` for headless Vispy image-mode examples. Tests construct
their own small inputs and do not depend on the external archive. The retained
`examples/_reference_sizing.py` is the small historical estimator needed to
reproduce the documented before/after preparation comparison.

## Migration verification

The archive migration preserves 1,522 files (107,619,747 bytes), including the
original executed notebooks. Generated files were removed only after source and
destination hashes matched. The source checkout retains notebook code and prose,
with embedded outputs cleared. Only Markdown reports are tracked here; generated
output paths are ignored.

After relocation, the full suite passes **132 tests on NVIDIA** and **130 with
two explicit display-dependent skips in the CPU sandbox**. The notebook discovers
all four archived recordings; the replay check completes 18 retained-camera
updates and 12 requested refits using the relocated data. Fresh logs and renders
are in the separate archive's `reports/archive-migration/` directory.

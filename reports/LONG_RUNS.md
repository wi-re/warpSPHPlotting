# Extended simulation and renderer stress verification

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Current 3D support and large-particle performance are covered in the
[3D follow-up](THREE_DIMENSIONS.md), including automatic backend selection,
volume rendering, corrected depth handling and the 105-test suite.

Date: 2026-10-16. CPU, EGL/Mesa llvmpipe; CUDA and a working native display were unavailable.

The license file is now MIT, matching the existing package metadata, at the
copyright holder's direction. The wheel contains the same MIT file.

The particle-sizing follow-up replays these saved states with automatic markers.
See the [new comparison and performance report](PARTICLE_SIZING.md) and
updated Marrone movie (local evidence: `particle_sizes/pyvista_auto.mp4`). The original recordings
below retain their earlier marker defaults.

## Actual frontend integration

These runs use the real warpSPH case builder, integrator, boundaries, diagnostics
and scheduling. The verification driver supplies controlled speed/density plot
hooks, with PyVista rendering during integration. All four completed without
reported divergence. Positions, velocities, densities, masses and supports in
the saved final states were independently checked for finite values.

| Case | nx | Total particles | Steps | Final physical time | Plot calls | Wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| tgv | 24 | 576 | 2000 | 2 | 81 | 79.1 |
| sod | 24 | 600 | 400 | 3.65622 | 41 | 36.8 |
| khh | 48 | 2304 | 1200 | 2.3124 | 41 | 415.6 |
| marrone31 | 33 | 4400 | 5000 | 0.984309 | 51 | 1195.1 |

- Taylor–Green ran to t=2 through 2,000 steps of periodic motion and decay.
- Sod ran through 400 physical steps while the displayed domain progressively
  expanded. This is the frontend's periodic slab solver with an expanded
  visualization domain; it is not a vacuum-boundary expansion experiment.
- Marrone 3.1 used nx=33 to reach wall impact and a rising wall jet on this CPU.
  It is a coarse visualization stress run with the stated tank geometry,
  not a converged experiment/pressure-trace comparison.
- KHH reached t≈2.31 at nx=48 with an evolving two-density shear interface.
  The coarse run shows interface deformation rather than clearly resolved
  rolled-up billows. Higher-resolution KHH roll-up remains a separate check.

Early, middle and final states for each case (local evidence: `long_frontend/evolution.png`)

Playback clips: Taylor–Green (local evidence: `long_frontend/tgv.mp4`),
Sod sparse view (local evidence: `long_frontend/sod.mp4`), KHH (local evidence: `long_frontend/khh.mp4`),
Marrone impact (local evidence: `long_frontend/marrone31.mp4`). Each retained image was opened and
verified, and ffprobe independently decoded and counted the video frames.
The clips use accelerated, fixed-frame-rate playback. The initial setup image
and first step share a filename in this driver, so retained-image counts are
one below plot-call counts. Physical times remain printed on the frames.

Final impact-state backend renders: Matplotlib (local evidence: `long_frontend/marrone31_matplotlib.png`),
PyVista (local evidence: `long_frontend/marrone31_pyvista.png`), and
Vispy (local evidence: `long_frontend/marrone31_vispy.png`). Final Taylor–Green, Sod, and KHH
replays are retained alongside these images for each backend.

## Sustained rendering and bugs fixed

Each backend completed **2,000 temporal replay updates**, including particle
count/domain changes, scatter/grid switching, streamlines, active/passive walls,
and colorbar hide/show. The first 500 updates warm caches. Replay loads saved
states and does not integrate additional physical time. Sources are recorded
in each backend's measurements.json; they were captured while the longer
integrations were still progressing and therefore cover subsets of those runs.

1. **PyVista grid re-entry crash:** grid → particles → the same grid geometry
   selected a cleared mesh via a surviving geometry signature. Mesh reuse now
   requires an existing mesh. Repeated same-resolution transitions are tested.
2. **Vispy resource churn:** hiding/revealing colorbars rebuilt text/font
   resources, and quantity-limit changes also rebuilt shaders and LUT textures.
   Hidden widgets now persist per panel, unchanged colormaps retain their
   shader/texture, and LUT textures survive range and layout changes.
3. **Vispy colormap update ordering:** a changed palette could leave the old
   LUT bound until a later geometry/range refresh. The new palette is installed
   before updating limits; an exported-pixel regression verifies its new color
   in the same frame.

The controlled two-panel hide/show probe retained approximately
71.6 MiB before and 7.0 MiB after the fixes across 40 additional cycles.
The original growth leveled off in that probe; it is evidence of avoidable
resource/cache churn, not proof of an unbounded leak.

| Backend | Median update + render (ms) | p95 (ms) | RSS growth after warmup (MiB) | Value-only camera checks |
|---|---:|---:|---:|---:|
| matplotlib | 75.6 | 228.8 | 3.4 | 1405 |
| pyvista | 24.5 | 250.3 | 3.8 | 1481 |
| vispy | 13.2 | 183.7 | 15.2 | 1411 |

Resident memory changes (local evidence: `stress_memory.png`)

Scene counts stayed bounded per case/mode. Streamline arrows and a PyVista
stream actor legitimately vary with integration geometry; optional stream actors
are counted separately. No unexpected camera changes were found. Vispy's image
exports also exercised asynchronous writes, which were drained before decoding.
RSS still rises during the measured window, particularly in Vispy; these finite
runs do not establish zero memory drift over arbitrarily long sessions. Native
GL/allocator caches are included in RSS. Continued Vispy memory profiling remains
worthwhile. The resource improvements reduce churn without a zero-leak claim.

Timing runs overlapped other verification work, differ in raster coverage and
include CPU streamline integration in grid modes. They are stress observations,
not a controlled backend speed ranking. Retained raw evidence:
`stress/*/measurements.json`, `stress_vispy_before.json`, and
`stress_colorbar_before.json` / `stress_colorbar_after.json`.

The complete local regression suite passes **86 tests**. A wheel builds. Remote
CI and physical convergence are not established by these headless checks.
Subsequent native-window, CUDA and trame-startup checks are recorded separately
in GRID_COVERAGE.md.

## Reproduction

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
python examples/verify_frontends.py --case marrone31 --backend pyvista \
  --nx 33 --steps 5000 --plot-interval 100 --snapshots --checkpoint \
  --compact-diagnostics --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend"
python examples/verify_frontends.py --case khh --backend pyvista \
  --nx 48 --steps 1200 --plot-interval 30 --snapshots --checkpoint \
  --compact-diagnostics --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend"
python examples/stress_replay.py --backend pyvista --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend" \
  --frames 2000 --warmup 500 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/stress"
python examples/stress_replay.py --backend vispy --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend" \
  --source-manifest "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/stress_vispy_before.json" --frames 2000 \
  --warmup 500 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/stress"
python examples/stress_colorbars.py --cycles 40 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/stress_colorbar_after.json"
```

Use the dependency/cache environment described in VERIFICATION.md; add
EGL_PLATFORM=surfaceless for headless Vispy. Matplotlib uses the same replay
command with --backend matplotlib. The two lower-cost frontend runs use nx=24,
with tgv --steps 2000 --plot-interval 25 and sod --steps 400 --plot-interval 10
--sparse-view. Checkpoints and temporal NPZ states are retained in each case's
output directory.

The [grid coverage and resource follow-up](GRID_COVERAGE.md)
adds transparent unsupported cells, shared sizing work and measured Vispy
transform reuse, with examples and another 3,000 temporal replay updates.

Subsequent host checks accessed the native display and CUDA outside the sandbox,
verified native updates and short CUDA frontend runs, and fixed a GLX/EGL
handoff failure. Trame startup and dependency installation were also checked.
See the grid coverage follow-up for exact scope and remaining browser checks.

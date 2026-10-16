# Automatic particle sizing

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

The [physical sizing follow-up](PHYSICAL_PARTICLE_SIZING.md) changes the default
to volume-based spacing. The neighbor-distance cap measured in this earlier
report remains available with `markerSpacing="neighbors"`.

Current 3D support and large-particle performance are covered in the
[3D follow-up](THREE_DIMENSIONS.md), including automatic backend selection,
volume rendering, corrected depth handling and the 105-test suite.

Verified 2026-10-16 with Python 3.13, CPU rendering and EGL/Mesa llvmpipe.

The default `markerSize=None` now adapts to particle spacing in Matplotlib,
PyVista and Vispy. PyVista previously used a fixed 8-pixel default, which merged
neighboring particles in the Marrone impact recording. Vispy estimated a panel
size before layout and used a minimum 2-pixel marker. Matplotlib sized before
final layout, used only nominal volume spacing and added a marker outline.

## Behavior

- The automatic diameter defaults to **0.65 times estimated spacing**. Set
  `PlottingOptions(markerSizeScale=0.5)` for smaller automatic markers.
- Fluid and boundary populations get separate estimates. The estimate starts
  with the tenth percentile of `(mass / density) ** (1 / dimension)` and is
  capped by the tenth percentile of sampled local neighbor distances. This
  catches closely spaced particle pairs without letting one outlier shrink
  every marker. Invalid volume data falls back to occupied-position geometry.
- Local searches use spatial bins, at most 512 query particles and 16 candidate
  points per adjacent bin. They avoid an all-pairs matrix and add no dependency.
  This is a conservative population estimate, not an exact per-particle
  non-overlap guarantee.
- Empty domain area and total boundary/ghost counts do not inflate fluid sizes.
  Zoom, resize and final subplot layout determine the projected diameter.
  Matplotlib also resolves size at the actual export DPI. Camera-only changes
  reuse the existing spacing estimate.
- Explicit `markerSize` retains fixed Matplotlib scatter size in points squared.
  The GPU conversion now matches its normalized marker path, `sqrt(s)` points
  across at 96 dpi; the previous literal-circle-area conversion made GPU
  markers wider. GPU `backendOptions['point_size']` remains a fixed pixel
  override when `markerSize` is absent.

Matplotlib resolves sizes immediately before drawing and removes the outline
on automatic colored markers. PyVista updates point properties at render time
from the active renderer/camera projection. Vispy uses scene-space marker sizes,
so its camera handles zoom and perspective without resubmitting particle data.

## Actual frontend examples

These images replay the saved real frontend simulations from
[the extended runs](LONG_RUNS.md); no additional physical integration was needed.
For each backend the comparison shows fixed `markerSize=36`, automatic sizing,
and an automatic-size wall detail using the camera controls. At 96 dpi,
`sqrt(36)` points corresponds to an 8-pixel marker path.

Rows: Matplotlib, PyVista, Vispy. Columns: fixed, automatic, wall detail. (local evidence: `particle_sizes/backend_comparison.png`)

Full-resolution automatic views and wall details:

| Backend | Whole domain | Wall detail |
|---|---|---|
| Matplotlib | Automatic (local evidence: `particle_sizes/matplotlib_auto.png`) | Zoom (local evidence: `particle_sizes/matplotlib_zoom.png`) |
| PyVista | Automatic (local evidence: `particle_sizes/pyvista_auto.png`) | Zoom (local evidence: `particle_sizes/pyvista_zoom.png`) |
| Vispy | Automatic (local evidence: `particle_sizes/vispy_auto.png`) | Zoom (local evidence: `particle_sizes/vispy_zoom.png`) |

At the final Marrone state, nominal fluid and wall spacing is about 0.0302.
The sampled lower fluid-neighbor spacing is about 0.00824, while wall spacing
remains about 0.0301. Treating these populations separately exposes close fluid
pairs and preserves visible gaps between boundary layers when zoomed in.

The updated PyVista Marrone movie (local evidence: `particle_sizes/pyvista_auto.mp4`) contains all
51 retained snapshots through t=0.984309, including wall impact and the rising
jet. The complete clip decoded successfully as 51 frames at 1056×480; see
ffprobe output (local evidence: `particle_sizes/pyvista_video_check.json`). The frames were also
reviewed in a contact sheet (local evidence: `particle_sizes/movie_contact.png`). Playback is
accelerated at 12 fps, with physical time printed on each frame.

The final Taylor–Green, expanded Sod and KHH states were also rendered through
all three backends. Rows below follow that order; columns are Matplotlib,
PyVista and Vispy. The Sod view expands only the displayed domain by 7.5×.

Automatic marker sizing across the other frontend cases (local evidence: `particle_sizes/case_gallery.png`)

## Performance

A synthetic uniform **100,000-particle** fixture, one panel, 7×5-inch figure,
two CPU threads, three warm-up frames and twelve measured updates. Fixed mode
uses `markerSize=2`; automatic mode uses the default. Each update supplies new
particle buffers. Timings exclude PNG encoding. These are local software-renderer
measurements, not GPU results or a new simulation benchmark.

| Backend | Fixed preparation (ms) | Auto preparation (ms) | Fixed total (ms) | Auto total (ms) |
|---|---:|---:|---:|---:|
| Matplotlib | 7.9 | 17.7 | 393.5 | 187.0 |
| PyVista | 4.0 | 14.1 | 32.3 | 35.7 |
| Vispy | 6.7 | 17.2 | 24.7 | 35.2 |

The spacing calculation costs about 10 ms at 100k particles on this CPU.
Rendering costs also change with marker coverage and outline width, particularly
in Matplotlib; the total times do not isolate the sizing algorithm. Raw samples
are `particle_sizes/benchmark_{backend}_{fixed,auto}.json`; the compressed
synthetic input (local evidence: `particle_sizes/uniform_100k.npz`) is retained. Separate
spacing measurements (local evidence: `particle_sizes/spacing_performance.json`) cover 1k, 10k
and 100k populations. These measurements precede the multi-panel sharing follow-up. Multi-panel
views now share estimates within each update; see [the follow-up report](GRID_COVERAGE.md).

## Regression checks and limits

The complete suite passes **86 tests with no skips**; raw output (local evidence: `pytest.txt`).
The wheel builds and includes the shared sizing module and the MIT license.
The sizing stage added twelve checks covering close spacing, mixed resolution, empty/invalid-volume
inputs, 3D volume spacing, coincident positions, independent boundary sizing,
manual/backend overrides, invalid options before panel mutation, and export
DPI/legend support. Pixel tests on all three real renderers verify that 1.5×
zoom and 1.25× resize enlarge automatic markers while fixed markers stay fixed.

Very small projected spacings eventually reach pixel/rasterizer limits: zoom in
to inspect the fluid in the expanded Sod view. Exact coincident particles cannot
be separated by changing their diameter. In native 3D PyVista, sizes use the
camera focal plane as a representative depth; Vispy scales with each point's
actual depth. The initial sizing checks were headless. Subsequent native-window, CUDA and
trame-startup checks are documented in GRID_COVERAGE.md; browser input remains
unverified.

## Reproduce

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
# Fixed/auto/zoom images; add --video for the complete saved Marrone sequence.
EGL_PLATFORM=surfaceless python examples/particle_sizes.py \
  --backend pyvista --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend/marrone31-pyvista" \
  --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/particle_sizes" --video
# Repeat without --video for --backend matplotlib and --backend vispy.

# Other actual frontend snapshots, defaulting to automatic sizing.
EGL_PLATFORM=surfaceless python examples/render_gallery.py \
  --backend vispy --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend/khh-pyvista/final_state.npz" \
  --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/particle_sizes/khh_vispy.png"

EGL_PLATFORM=surfaceless python examples/benchmark_snapshots.py \
  --backend pyvista --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/particle_sizes/uniform_100k.npz" \
  --frames 12 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/particle_sizes/benchmark_pyvista_auto.json"
# Add --marker-size 2 for the fixed-size comparison.

EGL_PLATFORM=surfaceless MPLBACKEND=Agg pytest -q tests
```

The [grid coverage and resource follow-up](GRID_COVERAGE.md)
adds transparent unsupported cells, shared sizing work and measured Vispy
transform reuse, with examples and another 3,000 temporal replay updates.

Subsequent host checks accessed the native display and CUDA outside the sandbox,
verified native updates and short CUDA frontend runs, and fixed a GLX/EGL
handoff failure. Trame startup and dependency installation were also checked.
See the grid coverage follow-up for exact scope and remaining browser checks.

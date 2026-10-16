# Implementation and verification report

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Current 3D support and large-particle performance are covered in the
[3D follow-up](THREE_DIMENSIONS.md), including automatic backend selection,
volume rendering, corrected depth handling and the 105-test suite.
The subsequent [notebook follow-up](NOTEBOOK_INTERACTIVITY.md) covers browser
input, faster preparation and the current 117-test host suite.

Date: 2026-10-16. Baseline: plotting commit `bc24ad6` (0.6.0).

## Result

Implemented the correctness and backend lifecycle work, PyVista colorbar layout,
shared frame preparation, 2D streamlines across all three renderers, planar 3D
slices, VTK XML export, examples, and regression/rendering tests. The local suite
passed **86 tests with no skips** at the initial checkpoint. A wheel builds successfully. CI is configured;
no remote CI result is claimed.

Verification includes actual frontend time integration, frontend-owned plotting
hooks, static image inspection, independent VTK reads, and replay timings.
This initial checkpoint did not verify native GUI interaction, trame widgets,
CUDA rendering, or physical convergence. Later host checks and current 3D
results are linked above.

The [extended-run report](LONG_RUNS.md) adds longer frontend trajectories,
Marrone wall impact, 6,000 replay updates, videos, and resource/camera checks.
It records the grid re-entry crash fix and Vispy resource improvements found
by the stress runs. The license file now matches the MIT package metadata.

## Rendered examples

Actual frontend snapshots (local evidence: `frontend_gallery.png`)

The gallery uses real simulation outputs, rendered again through the final
backend. Full-size images: Taylor–Green (local evidence: `tgv_pyvista.png`),
Sod sparse view (local evidence: `sod_pyvista.png`), Marrone 3.1 (local evidence: `marrone_pyvista.png`),
KHH (local evidence: `khh_pyvista.png`). The sparse Sod view deliberately keeps the domain in view
rather than automatically zooming into the small particle cloud.

The self-contained analytical example compares particles, interpolation, and
streamlines, and updates the field amplitude:

Grid and streamline update (local evidence: `quickstart/pyvista_updated.png`)

Corresponding Matplotlib and Vispy initial/updated renders are in `quickstart/`.
`demo.ipynb` executes the standalone example from a fresh kernel and displays the
saved figures. All code cells were executed successfully in a clean namespace.

## Numerical and state fixes

- Symmetric-log colors retain negative values; symmetric scaling works with the
  default zero center. Median centers are recalculated without mutating options.
- Invalid samples are masked; constant, empty, logarithmic, and explicit-bound
  cases have defined behavior. Mapping and per-kind tuple inputs are validated.
- Shared tensors update all panels; keyed updates work regardless of the input
  form. Domain changes propagate, while value-only updates preserve the camera.
- Particle and quantity snapshots accept non-leaf tensors and release autograd
  graphs. Plotting does not mutate the caller's state.
- Creation/updates share rotation, wrapping, operations, mapping, and filtering.
- Hide/show/passive, changing particle counts, scatter/grid switches, colorbars,
  titles, outlines, grids, and streamlines have explicit lifecycles.
- Domain outlines and actors do not accumulate. Export renders the current frame
  even when the update used `redraw=False`.
- Background image failures remain observable and draining waits for all prior
  work before reporting an error.
- Grid generation validates thin domains and edge spacing. Scalar/vector
  interpolation shares grid/search data within an immutable prepared frame;
  moving-particle updates construct a fresh search, preventing stale neighbors.

## PyVista frame and colorbar correction

Visual review identified two remaining layout defects: the plot axes coincided
with the blue domain outline at the data plane, and rotated colorbar labels
remained detached from the narrow strip. The previous screenshots did not
establish a satisfactory layout.

The 2D plot now has an explicit four-sided gray frame padded outside the domain
and visible boundary particles. The blue physical domain outline is drawn just
above the data plane, keeping it visible over interpolated surfaces. Frame axes
use those padded bounds instead of following the last mesh. All four frame and
domain edges are checked in rendered pixels for both particles and grids.

Default vertical colorbars follow the projected plot frame with a 12-pixel gap,
instead of using a fixed sidebar position far from square plots. Explicit
positions remain respected. Colorbar ticks default to upright text, left-aligned
five pixels from the strip.
VTK's unused internal text reservation is disabled so the strip fills its
requested width. Measured text widths reserve enough room for long values;
render callbacks reflow labels when the window changes, including without a
quantity update. Tick/title actors persist across updates. Optional rotation,
numeric format, font sizes, label count, orientation, and explicit bar bounds
remain supported. Default fonts follow panel height; explicit sizes remain
fixed. Figure titles use a separate white band, and camera framing reserves
room for colorbars so wide tanks do not lie underneath the strip.

Normalized display arrays provide correct nonlinear colors and fast lookup;
separate mesh arrays retain physical scalar values. Tick labels use the inverse
norm, preserving physical units.

The streamline overlay also exposed a VTK MSAA artifact: thin black lines over
the colored grid acquired white fringes. Isolating the grid and lines and
disabling lighting did not remove those fringes; disabling multisampling did.
While a 2D streamline overlay is present, the window uses single-sample
rendering and unlit black lines. Removing the overlay restores its previous
multisampling request, subject to the graphics driver's supported sample count.
A rendered regression compares the same field with and without lines and checks
that adding black lines only darkens pixels, never adds white highlights.

Density, log and signed bars (local evidence: `colorbar_layout.png`)

Horizontal presentation (local evidence: `colorbar_horizontal.png`)

Resized/updated vertical example (local evidence: `colorbar_layout_resized.png`) and
resized horizontal example (local evidence: `colorbar_horizontal_resized.png`) exercise viewport
changes and updated limits, with upright labels kept inside each panel. The
density endpoints remain distinguishable around
1; signed values and logarithmic decades retain their meaning. Pixel comparisons
also protect against stale exported frames. Offscreen static output was visually
inspected; native-window/trame layout has not been tested here.

## Matplotlib and Vispy follow-up

The same analytical vortex, narrow density/log/signed colorbars, and four saved
frontend final states were inspected through both other backends. These are
replays of the completed simulations above, not new integration runs.

- Matplotlib already rendered black streamlines and upright bar labels. Its
  colorbar padding is now 2.5% instead of 5%. Constrained layout is the default
  unless explicitly overridden, fixing clipped tick labels after resizing.
  Crowded symmetric-log ticks near zero are thinned according to their screen
  separation while preserving zero and outer decades. Physical normalization
  is unchanged; ordinary scalar formatter offsets remain supported.
- Vispy previously drew the grid over its streamlines. Explicit scene order
  now puts black streamlines and domain outlines above the image, with depth
  testing disabled for those 2D overlays. Endpoint labels are upright, placed
  beside a 14-pixel strip with a 12-pixel gap from the view frame; their column
  grows to fit formatted values. Titles have their own row above the view.
  Axis lines/ticks are thinner. Programmatic EGL resizing now updates widgets
  and transforms before rendering, preventing the last mosaic panel cropping.
- Pixel regressions verify that adding the black streamlines visibly darkens
  the field without white highlights. Resize regressions check label bounds,
  both panels, bar spacing, upright ticks, and symlog label overlap. All 58
  tests passed at that stage; the extended suite now has 86. These checks cover headless rendering, not native GUI
  event handling or remote notebook transport.

Matplotlib vortex (local evidence: `quickstart/matplotlib_updated.png`)

Vispy vortex (local evidence: `quickstart/vispy_updated.png`)

Colorbar examples: Matplotlib (local evidence: `colorbar_matplotlib.png`),
Matplotlib resized (local evidence: `colorbar_matplotlib_resized.png`),
Vispy (local evidence: `colorbar_vispy.png`), Vispy resized (local evidence: `colorbar_vispy_resized.png`).

Actual frontend snapshot galleries: Matplotlib (local evidence: `frontend_gallery_matplotlib.png`)
and Vispy (local evidence: `frontend_gallery_vispy.png`). Each includes Taylor–Green, expanded
Sod display, Marrone 3.1, and KHH.

## Actual frontend simulations

The shared `warpSPH.runner` constructed and integrated the real cases. For
controlled comparisons, one driver supplied identical speed/density panels.
Separate runs also exercised the frontend's own plotting hooks.

| Case | Particle count | Completed integration | Final simulated time | Result |
|---|---:|---:|---:|---|
| Periodic 2D Taylor–Green | 576 | 40 steps | 0.0400 | finite, no divergence |
| Sod 2D slab | 600 | 40 steps | 0.3656 | finite, no divergence |
| Marrone 3.1 geometry | 10,272 including ghosts | 300 + checkpointed 300 steps | 0.0583 | finite, no divergence |
| KHH two-density layers | 576 | 40 steps | 0.1542 | finite, no divergence |

Position, velocity, density, mass, and support buffers in all four final-state
fixtures were independently checked for finite values. Marrone has 3,240 fluid,
3,516 boundary, and 3,516 ghost particles; ghosts are hidden in the plots.

- **Taylor–Green:** periodic wrapping, evolving velocities, density limits,
  value/state updates, and visible axes. Matplotlib, PyVista, and Vispy runs.
- **Sod:** the actual solver remains a periodic slab. Only the display domain is
  expanded, up to 7.35× each original extent, to stress an increasingly empty
  viewport and changing bounds. This changes no solver boundary conditions.
- **Marrone:** section 3.1 tank proportions (H=0.6, tank width 3.2196, column width
  1.2), nx=67, g=9.81, free-slip walls, Sun-2017 delta-SPH, and target
  c0=40√(gH). The frontend's time integration defaults were retained. The run is
  early-time; it does not reach impact or validate experimental pressure traces.
  The later [nx=33 extended run](LONG_RUNS.md) reaches impact and a wall jet.
  A longer initial process was terminated by the execution environment; the
  reported result comes from completed shorter runs and a full-state checkpoint.
- **KHH:** the frontend's compressible two-density-layer `sampleKHH` case
  (rho=1/2, counter-streaming velocities), rather than an asserted test of every
  immiscible multiphase model. Particle fields and the native grid panel were
  exercised.

The frontend-owned hooks completed for all four cases: Taylor–Green (4 steps),
Sod profiles (4), KHH (4), and Marrone from checkpoint (2). The KHH native-hook
check used a **128 grid instead of its default 1024** to bound CPU cost. Actual
backend selection was recorded and did not fall back to another renderer.
Native-hook frames, run configuration, and measurements are retained under
`native_frontend/` and `native_frontend_final/`.

Raw controlled-run data: `frontend/*/measurements.json`, final NPZ fixtures,
and `marrone_stage1/` / `marrone_stage2/`. The checkpoint is
`marrone_stage1/marrone31-pyvista/checkpoint.h5`. These bounded runs verify
visualization behavior; they do not establish reference-solution accuracy or
resolution convergence. In particular, the KHH run is early evolution and the
Sod time reflects the explicitly requested fixed step count.

## Replay performance

Environment: Linux, Python 3.13.14, Torch 2.13.0, Warp 1.17.0, Matplotlib 3.11.1,
PyVista 0.49.1 / VTK 9.7.1, Vispy 0.16.2. Rendering used **Mesa llvmpipe/EGL;
CUDA was unavailable**. See environment.json (local evidence: `environment.json`).

Serial backend comparisons used the same saved Taylor–Green state, viewport,
marker setting, two Torch threads, and two Mesa/VTK worker threads. Larger
populations repeat real samples to test throughput; they are not higher-resolution
simulations. Three frames warm up the renderer, followed by 12 measured frames.

Matplotlib and Vispy timings were repeated after their layout/overlay fixes.
The Matplotlib measurements include automatic constrained layout. PyVista's
measurements follow its initial frame/colorbar correction and predate the later
12-pixel gap and streamline MSAA correction. These replay plots have no
streamlines. The renders differ in layout, rasterization, and pixel coverage;
timings are end-to-end host observations rather than isolated algorithm costs.
Median preparation plus offscreen rendering/readback, excluding PNG compression:

| Backend | 30k samples (ms) | 100k samples (ms) |
|---|---:|---:|
| matplotlib | 147.3 | 398.0 |
| pyvista | 22.4 | 38.6 |
| vispy | 21.6 | 31.0 |

Replay timings (local evidence: `performance.png`)

The original commit was checked out read-only into a temporary directory and
run through the same replay script/environment:

| PyVista sample count | Original total (ms) | Updated total (ms) | Observed speedup |
|---|---:|---:|---:|
| 30k | 47.9 | 22.4 | 2.14× |
| 100k | 79.0 | 38.6 | 2.05× |

The original path spent roughly 20 ms unconditionally sleeping outside notebooks.
The updated path removes that pause, shares resolved norms, preserves actors,
and uses normalized LUT arrays. Layout/label rendering also changed, so the total
speedup is an end-to-end observation rather than attribution to one change.
Raw frame samples and p95 values are in `benchmark_*.json` and
`baseline_pyvista_*.json`. These host-specific offscreen numbers do not predict
native GUI, trame transport, PNG encoding, or CUDA throughput.

## Reproduction

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
pip install -e '.[dev,plot-all]'
MPLBACKEND=Agg pytest -q tests/test_correctness.py
EGL_PLATFORM=surfaceless MPLBACKEND=Agg pytest -q tests

python examples/quickstart.py --backend pyvista
EGL_PLATFORM=surfaceless python examples/quickstart.py --backend vispy
python examples/colorbar_layout.py --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/colorbar_layout.png"
python examples/colorbar_layout.py --orientation horizontal --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/colorbar_horizontal.png"

# Requires the optional warpSPH frontend installation.
python examples/verify_frontends.py --case tgv --backend pyvista --steps 40
python examples/verify_frontends.py --case sod --backend pyvista --steps 40 --sparse-view
python examples/verify_frontends.py --case khh --backend pyvista --steps 40
python examples/verify_frontends.py --case khh --backend pyvista --steps 4 --native-hooks --grid-resolution 128
python examples/verify_frontends.py --case marrone31 --backend pyvista --steps 300 --checkpoint --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/marrone_stage1"
python examples/verify_frontends.py --case marrone31 --backend pyvista --steps 300 --resume-offset 300 \
  --resume "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/marrone_stage1/marrone31-pyvista/checkpoint.h5" --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/marrone_stage2"

python examples/benchmark_snapshots.py --backend pyvista --count 100000 \
  --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/frontend/tgv-pyvista/final_state.npz" --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/benchmark.json"
```

Use a writable Warp/Matplotlib/Numba cache in restricted environments. Verification
here used temporary caches and optional renderer packages under `/tmp`, without
changing the system Python installation. The frontend checkout was read-only.

## Limits and remaining decisions

- Native window updates, CUDA data/frontends and trame startup were subsequently
  checked outside the sandbox; browser input remains unverified. See GRID_COVERAGE.md.
- Three-dimensional streamline integration is unsupported; planar 3D slices
  with in-plane streamlines are supported. PyVista provides native 3D grid
  surfaces. Single-axis camera linking in PyVista is rejected explicitly;
  full-camera linking and independent cameras are available.
- Frame-local search reuse is implemented. Persistent caches across moving
  simulation frames remain deliberately conservative; no stale spatial cache
  is carried across an update.
- Raw grid interpolation retains the underlying SPH estimator, including zeros
  outside support. Rendering now masks unsupported cells by default; it does
  not introduce a new physical reconstruction or free-surface estimator.
- The copyright holder confirmed MIT; `LICENSE` and package metadata are now
  aligned. The built wheel's bundled license is checked during verification.
- CI has been added, but only the local test/build evidence is reported here.

The [automatic particle-sizing report](PARTICLE_SIZING.md) adds fixed/auto/zoom
examples for every backend, the updated Marrone movie, and 100k-particle timings.

The [grid coverage and resource follow-up](GRID_COVERAGE.md)
adds transparent unsupported cells, shared sizing work and measured Vispy
transform reuse, with examples and another 3,000 temporal replay updates.

Subsequent host checks accessed the native display and CUDA outside the sandbox,
verified native updates and short CUDA frontend runs, and fixed a GLX/EGL
handoff failure. Trame startup and dependency installation were also checked.
See the grid coverage follow-up for exact scope and remaining browser checks.

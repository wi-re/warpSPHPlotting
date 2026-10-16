# Grid coverage, shared sizing and Vispy resource verification

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Current 3D support and large-particle performance are covered in the
[3D follow-up](THREE_DIMENSIONS.md), including automatic backend selection,
volume rendering, corrected depth handling and the 105-test suite.

Verified 2026-10-16. Python 3.13. CPU/EGL-Mesa measurements and separate
NVIDIA/native-display/CUDA validation are distinguished below.

Grid plots now make unsupported regions transparent by default. Supported zeros
retain their proper color. Automatic particle sizing shares its spacing estimate
across panels within each update. Vispy grid updates reuse the image transform,
reducing the retained-memory growth reproduced in a fixed-state probe.

## Empty regions and genuine zeros

`GridVisualization(maskEmpty=True)` is the default. Set `maskEmpty=False` to
restore the earlier appearance. Coverage comes from interpolating a unit field
through the same kernel, support scheme, particle-kind selection and neighbor
search as the plotted field. A sample is covered when that contribution is
finite and positive. Fluid and boundary visibility therefore affect coverage;
ghosts do not contribute. Periodic wrapping and physical 3D slice coordinates
are handled by the existing SPH search.

The coverage calculation reuses the current frame's neighbor search. Quantity
and streamline mappings share its mask. Display bounds exclude unsupported
samples, and streamlines stop at masked cells. A view with no covered samples
is transparent and hides its colorbar until covered samples return.

Raw interpolated values remain unchanged, including zeros outside support.
`mapToGrid` still returns four values; its optional `computeCoverage=True`
adds the Boolean mask to the returned extent dictionary. Rendering applies the
mask without changing those physical arrays. A bitwise comparison with masking
on/off passed for all four final frontend states:

| Case | Grid | Covered samples | Unsupported samples |
|---|---|---:|---:|
| Periodic TGV | 192×192 | 36,864 | 0 |
| Sod, displayed domain expanded 7.5× | 192×79 | 488 | 14,680 |
| Marrone wall impact | 192×71 | 9,008 | 4,624 |
| Periodic KHH | 192×192 | 36,864 | 0 |

Raw numerical checks (local evidence: `coverage/physical_values.json`).

Rows below are TGV, Sod, Marrone and KHH; columns are Matplotlib, PyVista and
Vispy. Each image compares the earlier filled background on the left with
coverage transparency on the right. Both panels use the same covered-grid
color range, so the comparison changes opacity rather than normalization.

Actual frontend states across all three backends (local evidence: `coverage/case_gallery.png`)

Full-resolution examples: Sod / PyVista (local evidence: `coverage/sod_pyvista.png`),
Marrone / Matplotlib (local evidence: `coverage/marrone31_matplotlib.png`),
TGV / Vispy (local evidence: `coverage/tgv_vispy.png`), KHH / PyVista (local evidence: `coverage/khh_pyvista.png`).

A separate zero-field replay uses the actual Marrone particle locations and
sets the displayed quantity to zero. The fluid and boundaries remain colored;
only unsupported space disappears. Columns follow the same backend order.

Genuine zeros remain visible (local evidence: `coverage/zero_gallery.png`)

The actual Marrone velocity field also renders with streamlines through all
three backends. An additional regression supplies deliberately nonzero vectors
outside coverage and verifies that the integrator still stops at the mask.

Marrone velocity and streamlines (local evidence: `coverage/streamline_gallery.png`)

This mask describes support of the chosen SPH estimator. It is not an exact
free-surface reconstruction or a change to interpolation normalization. Existing
SPH weight/partition errors remain visible; this work does not establish physical
convergence of the coarse simulations. Very coarse grids can miss a small fluid
patch entirely and require higher resolution.

## Shared sizing work

Prepared panels share the unscaled spacing estimate per particle population.
Different `markerSizeScale` values remain independent. The cache is fresh for
every update, including quantity-only updates; geometry/domain/support changes
cannot retain an earlier frame's estimate. Fixed-size and hidden populations
avoid unnecessary estimation. Camera-only draws use the resolved diameter.

A controlled comparison used the retained synthetic 100k-particle fixture,
two panels, two CPU threads, three warm-up updates and twelve measured updates.
The independent mode bypasses only the new shared estimate; it uses the same
current renderer and data. PNG encoding is excluded.

| Backend | Independent preparation (ms) | Shared preparation (ms) | Independent total (ms) | Shared total (ms) |
|---|---:|---:|---:|---:|
| Matplotlib | 34.79 | 24.13 | 353.66 | 359.89 |
| PyVista | 27.83 | 17.46 | 68.48 | 51.82 |
| Vispy | 32.99 | 22.37 | 64.95 | 54.15 |

Preparation saved approximately 10–11 ms per update. Total render times vary
even with identical pixels; Matplotlib total time did not improve in this sample. The final independent and
shared screenshots are **pixel-identical on every backend**; see the
pixel comparison (local evidence: `coverage/sizing_pixel_comparison.json`). Raw samples are
`coverage/sizing_{backend}_{independent,shared}.json`. These are measurements on
this software-rendering host, not a universal backend ranking.

## Vispy memory investigation

A fixed real Marrone state was updated 2,000 times in two panels, with changing
quantity ranges and stable geometry. Samples include garbage collection and
Vispy object counts. The first 500 updates are warm-up. Separate fresh processes
isolate marker updates, the original grid transform replacement and grid
transform reuse.

| Probe | RSS growth after warm-up | Vispy object-count growth |
|---|---:|---:|
| Markers | 0.113 MiB | None |
| Grid: replace image transform every update | 7.988 MiB | None |
| Grid: reuse image transform and release EGL binding | 0.125 MiB | None |

The grid path created a new `STTransform` and replaced the scene/shader transform
graph for each data update. It now retains the transform, changing scale and
translation only when geometry changes. A pixel regression verifies changed
colors in the same frame, and geometry/resolution updates retain the transform
while mapping image edges to the new domain. Empty PyVista grids also retain
their lookup table instead of rebuilding it when no norm is available.

Matched fixed-state Vispy memory probes (local evidence: `coverage/memory_comparison.png`)

Raw probes: markers (local evidence: `coverage/vispy_markers_before.json`),
grid before (local evidence: `coverage/vispy_grid_before.json`),
grid after transform reuse (local evidence: `coverage/vispy_grid_after.json`),
final grid implementation (local evidence: `coverage/vispy_grid_final.json`).
Stable scene counts alone did not identify this allocation churn. The matched
RSS comparison supports the improvement; finite runs do not prove zero memory
growth for all workloads or drivers.

## Temporal replay and tests

Each backend completed another **1,000 temporal replay updates** across all four
saved frontend simulations, including scatter/grid changes, streamlines,
changing domains and particle counts, and colorbar/visibility changes. These
replays do not integrate additional physical time.

| Backend | Updates | Camera checks | Detected errors | RSS growth after warm-up |
|---|---:|---:|---:|---:|
| Matplotlib | 1,000 | 765 | 0 | 0.00 MiB |
| PyVista | 1,000 | 765 | 0 | 7.90 MiB |
| Vispy | 1,000 | 765 | 0 | 2.22 MiB |

Visible colorbar presence is included in scene-count contexts: legitimately
empty grids hide their bar, and that layout change is excluded from value-only
camera checks. Scene counts remained bounded for equivalent case/mode/bar
contexts. Raw results are `coverage/stress/{backend}/measurements.json`.
These runs overlapped; their timings are not used for a speed comparison.
PyVista and Vispy still retain some additional RSS during varied workloads.

Selected temporal replay frames (local evidence: `coverage/stress_contact.png`)

The complete local suite passes **86 tests with no skips** on both the sandbox
CPU renderer and the native NVIDIA host. Thirteen new tests
cover numerical and rendered coverage, genuine zeros, empty views and recovery,
periodic domains, visible particle kinds, physical 3D slices, masked streamline
termination, shared sizing/invalidation, Vispy transform/pixel updates and live backend
context switching in a protected subprocess.
The test output (local evidence: `pytest.txt`) and wheel build (local evidence: `wheel_build.txt`) are retained.

## Native display, CUDA and trame

The initial sandbox capability check (local evidence: `coverage/host_capabilities.json`) hid the
GPU and X display. A subsequent host check outside that sandbox (local evidence: `coverage/host_capabilities_unsandboxed.json`)
confirmed a working display and an NVIDIA RTX PRO 6000 Blackwell Workstation
Edition (95 GiB). Earlier CPU-only measurements remain valid for their stated
sandbox environment; they do not describe the host's full capabilities.

All three native backends successfully mapped real windows with CUDA-resident
Marrone input. Matplotlib used TkAgg, Vispy used GLFW, and PyVista used its native
VTK/X renderer. Each window processed resize, camera and live quantity/grid/
streamline changes, exported the updated image and closed. The final size was
900×600. PyVista and Vispy reported the NVIDIA OpenGL renderer. These checks
exercise native event processing programmatically; they do not claim manual
mouse/keyboard acceptance. Raw results are `coverage/native_{backend}.json`.

Native windows before and after updates (local evidence: `coverage/native_gallery.png`)

CPU/CUDA interpolation comparisons used the same four saved frontend states,
including expanded Sod bounds. Coverage masks matched exactly, and the maximum
observed absolute scalar difference was **4.77×10⁻⁷**.
Raw parity checks (local evidence: `coverage/cuda_parity.json`).

The actual CUDA frontend then integrated another 100 steps per case using eager
execution, with CUDA Graphs and pipelined outputs disabled. PyVista rendered
during these runs. All completed without reported divergence, and saved
positions, velocities, densities, masses and supports passed independent finite
checks. These are short GPU integration checks, not converged physical results.

| CUDA case | Total particles, including ghosts | Steps | Final time |
|---|---:|---:|---:|
| TGV | 2,304 | 100 | 0.100000 |
| Sod, expanding display | 1,200 | 100 | 0.457030 |
| Marrone 3.1 | 4,400 | 100 | 0.0196854 |
| KHH | 2,304 | 100 | 0.192700 |

Independent final-state checks (local evidence: `coverage/cuda_frontend_check.json`).
Per-case frames, snapshots and measurements are under `coverage/cuda_frontend/`.
Compilation and concurrent verification work are included in the raw wall times;
these runs are not used for a GPU speed comparison.

Final frames of the short CUDA frontend runs (local evidence: `coverage/cuda_frontend_gallery.png`)

Installing trame exposed an optional-dependency gap: current PyVista delegates
its notebook backend to a separate adapter. The `plot-pyvista` and `plot-all`
extras now request **`pyvista[jupyter]`**, letting PyVista supply its compatible
adapter and notebook dependencies. The rendering CI job imports that adapter
explicitly to catch missing-install regressions.

A real notebook-mode trame startup then passed with the plotting backend: its
loopback UI returned HTTP 200, initialized the render callback, and quantity
updates changed exported pixels. Startup result (local evidence: `coverage/trame_startup.json`),
updated view (local evidence: `coverage/trame_after.png`). This verifies startup/HTTP and backend
updates; browser input and client-rendering mode remain unverified.

## NVIDIA GLX/EGL handoff bug found during host checks

The first combined host suite exited inside `GLXMakeCurrent` after Vispy EGL
rendering. An uncaptured diagnostic reproduced **X BadAccess**; isolated slice
tests passed. Original mixed-context diagnostic (local evidence: `coverage/pytest_host_mixed_contexts.txt`).

Vispy left an EGL context current after offscreen readback/close, and PyVista
left GLX current after managed display/export. Releasing only the EGL binding
fixed the closed-window sequence but still prevented alternating live windows
in the reverse direction. Both backends now release their current context at
those managed API boundaries. Shared contexts are not destroyed and the Vispy
application display is not terminated. Raw renderer handles still carry their
underlying toolkit's context-management responsibilities.

The new subprocess regression keeps both backends open, alternates changed
quantities/exports, closes them, and opens another PyVista plot. Pixel changes
are checked; a fatal driver exit becomes a captured test failure. The complete
host suite now passes **86 tests**, without forcing VTK to EGL, including this
live GLX/EGL sequence. Final NVIDIA-host output (local evidence: `coverage/pytest_host_final.txt`).
Headless CI explicitly selects EGL for both renderers for consistent setup.
The CPU/headless output (local evidence: `pytest.txt`) also passes all 86 tests.

Remote CI has not been executed here. Browser interaction, CUDA Graphs and
asynchronous simulation-output pipelines remain separate validation tasks.

## Reproduce

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
EGL_PLATFORM=surfaceless python examples/grid_coverage.py \
  --backend pyvista --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend/sod-pyvista/final_state.npz" \
  --domain-scale 7.5 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/sod_pyvista.png"
# Repeat with other snapshots/backends; --zero checks genuine zero fields,
# and --streamlines plots the saved velocity field.

EGL_PLATFORM=surfaceless python examples/benchmark_snapshots.py \
  --backend pyvista --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/particle_sizes/uniform_100k.npz" \
  --mosaic AB --frames 12 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/sizing_pyvista_shared.json"
# Add --independent-sizing for the comparison.

EGL_PLATFORM=surfaceless python examples/profile_vispy_memory.py \
  --mode grid --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend/marrone31-pyvista/final_state.npz" \
  --frames 2000 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/vispy_grid_after.json"

EGL_PLATFORM=surfaceless python examples/stress_replay.py \
  --backend vispy --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend" --frames 1000 --warmup 500 \
  --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/stress"

VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow EGL_PLATFORM=surfaceless MPLBACKEND=Agg pytest -q tests
```

On a host with the relevant display/device permissions:

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
python examples/verify_cuda.py --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend" \
  --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/cuda_parity.json"
python examples/verify_native.py --backend vispy --device cuda \
  --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend/marrone31-pyvista/final_state.npz" \
  --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/native_vispy.json"
python examples/verify_trame.py \
  --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/long_frontend/marrone31-pyvista/final_state.npz" \
  --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/trame_startup.json"
python examples/verify_frontends.py --case khh --backend pyvista \
  --device cuda --nx 48 --steps 100 --compact-diagnostics --snapshots \
  --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/coverage/cuda_frontend"
```

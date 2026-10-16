# 3D rendering and particle-count scaling

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Verified 2026-10-16. This follow-up addresses native 3D visibility and large
particle plots. The original implementation had partial 3D paths, rather than
an end-to-end working 3D experience.

## Changes and reproduced failures

- `visualize()` now defaults to `backend="auto"`. Small 2D views retain
  Matplotlib. Native 3D or at least 50,000 visible scatter particles across
  panels prefer installed PyVista, then Vispy where supported. Native 3D grids
  require PyVista. Single-axis sharing uses Vispy or Matplotlib; PyVista
  requires matching `sharex`/`sharey`. Explicit selections are honored. Hidden populations and
  grid interpolation inputs do not trigger large-scatter routing. A figure
  keeps its selected backend during updates; no particles are subsampled.
- Native PyVista 3D grids default to ray-cast **volumes**, replacing the
  previous exterior-face-only view. `renderMode="surface"` retains the old
  representation. `volumeOpacity=0.15` specifies opacity per tenth of the
  domain diagonal, independently of grid resolution. Invalid/uncovered
  samples have zero opacity, while covered physical zeros remain visible.
  Raw `quantity` and normalized `display_quantity` arrays remain available;
  `volume_quantity` is a separate byte transfer-function input, with one
  transparent entry and 255 color entries. Volume colors combine along rays;
  they should not be read as individual point measurements.
- A transparent NaN lookup-table entry caused VTK to classify the entire
  particle actor as translucent. Coincident screen projections blended front
  and rear colors into brown, even for fully valid, opaque particles. Active
  particles now use the opaque depth pass; invalid vertices are omitted while
  preserving their raw point arrays. Passive particles remain translucent.
  Lighting is disabled for scalar-colored geometry to match the colorbar.
  Native 3D particles use circular sphere impostors, without shaded colors.
- Vispy's 3D export crashed when refreshing invisible, unlinked 2D axis widgets.
  Those widgets are now skipped. Its default orthographic depth interval was
  also too wide: NVIDIA rendered overlapping particles according to input
  order. The turntable camera now fits depth to the domain, camera center,
  and orbit distance; perspective uses a bounded near/far ratio.
- Switching a PyVista planar slice back to 3D retained the planar camera.
  Dimension transitions now restore the appropriate projection and orientation.
  Both accelerated backends link only compatible coordinate systems, so a 3D
  view and a planar slice can occupy the same mosaic with sharing enabled.
  Ordinary quantity changes retain camera state.
- Automatic support selection compared a slice's 2D sample count with a 3D
  particle population. The resulting query kernels caused stripes and holes
  between particle layers. Slices now default to the particles' physical
  supports. Explicit `gridSupport` choices still take precedence. A periodic
  constant-field test checks planes both on and between particle layers.
- Offscreen PyVista GLX captures on this NVIDIA host waited approximately one
  second per frame for display synchronization. Single-buffer offscreen
  windows preserve readable pixels and eliminate that delay. Native windows
  retain double buffering. Simply disabling buffer swaps produced black
  screenshots and was rejected; those diagnostic results are not performance
  evidence.

## Rendered examples

The self-contained [example](../examples/three_dimensions.py) samples a 3D
Taylor–Green analytical velocity field on a regular lattice, cropped to a sphere
so empty space is visible. It exercises actual 3D SPH interpolation and both
accelerated renderers, including CUDA input, volume updates, camera orbit,
moving physical slices, and mixed mosaics. This is analytical input, **not an
integrated 3D fluid simulation**. The earlier four integrated frontend cases
remain 2D and are documented in [GRID_COVERAGE.md](GRID_COVERAGE.md).

3D particles, volume and physical slice (local evidence: `3d/examples/gallery.png`)

Individual images:

- PyVista particles (local evidence: `3d/examples/pyvista_particles.png`),
  volume (local evidence: `3d/examples/pyvista_volume.png`),
  orbited volume after quantity update (local evidence: `3d/examples/pyvista_volume_orbit.png`),
  exterior-face surface (local evidence: `3d/examples/pyvista_surface.png`).
- PyVista mixed view (local evidence: `3d/examples/pyvista_mixed.png`),
  Vispy mixed view (local evidence: `3d/examples/vispy_mixed.png`).
- Moving XY slices at Z = π (local evidence: `3d/examples/pyvista_slice_mid.png`) and
  Z = π/2 (local evidence: `3d/examples/pyvista_slice_quarter.png`). Scalar magnitude is mapped
  before SPH interpolation: near the analytical velocity node, kernel-averaged
  speed can be positive because neighboring particles have nonzero speed.

## Performance

The [benchmark](../examples/benchmark_scaling.py) renders every particle at
768 × 576 pixels, using automatic marker sizing and two CPU preparation threads.
Each workload has unique seeded positions, three warm-up updates and eight
measured updates. Every update supplies the complete particle buffers and
changes scalar values. Rendering includes synchronized pixel readback; PNG
encoding and simulation stepping are excluded. Creation includes the first
capture. Workloads run serially, on the NVIDIA RTX PRO 6000 Blackwell host.
Tested package versions are recorded in versions.json (local evidence: `3d/versions.json`).
Matplotlib uses Agg; final PyVista uses a single-buffer offscreen GLX context;
Vispy uses EGL. The JSON records renderer details and checks that exported
images contain the colored particle field.

| Backend | Dimension | 10,000 particles | 100,000 particles | 1,000,000 particles |
|---|---:|---:|---:|---:|
| matplotlib | 2D | 48.6 ms | 177.7 ms | 1442.0 ms |
| pyvista | 2D | 7.1 ms | 19.2 ms | 148.3 ms |
| vispy | 2D | 7.0 ms | 21.7 ms | 186.2 ms |
| pyvista | 3D | 11.1 ms | 22.7 ms | 150.4 ms |
| vispy | 3D | 12.6 ms | 26.4 ms | 201.3 ms |

One-million-particle cost breakdown (medians are computed separately):

| Backend | Dimension | Preparation | Render + readback | Total |
|---|---:|---:|---:|---:|
| matplotlib | 2D | 125.7 ms | 1315.7 ms | 1442.0 ms |
| pyvista | 2D | 125.4 ms | 22.6 ms | 148.3 ms |
| vispy | 2D | 173.7 ms | 13.0 ms | 186.2 ms |
| pyvista | 3D | 133.1 ms | 16.7 ms | 150.4 ms |
| vispy | 3D | 183.0 ms | 17.0 ms | 201.3 ms |

Full update scaling (local evidence: `3d/scaling.png`)

The preparation column includes snapshots, filtering, color normalization,
CPU/GPU transfers where needed, and fresh spacing estimates. Rendering is much
cheaper than full updates at a million particles; selecting a GPU renderer
alone does not make every simulation update run at 60 fps. The 50,000-particle
threshold is a conservative heuristic, not a measured crossover for every
machine, mosaic, marker size, or driver. Explicit Matplotlib remains useful for
small figures and publication formats.

Raw final samples and exported images are in 3d/benchmarks (local evidence: `3d/benchmarks`).
3d/auto/auto_2d.json (local evidence: `3d/auto/auto_2d.json`) checks actual default routing with
100,000 particles. Pre-fix GLX measurements in `3d/baseline_glx` document the
one-second pacing problem and are separate from final comparisons. The
`no_swap_probe` was blank and is invalid for performance comparison.

## Verification and remaining scope

Follow-up: [notebook interaction and preparation profiling](NOTEBOOK_INTERACTIVITY.md)
now covers actual browser input and reduces the million-particle preparation
cost. Its newer 117-test host run supersedes the suite count below; the 3D
measurements here remain the original baseline.

**105 tests pass with no skips** on both CPU/Mesa and the NVIDIA host,
including unforced GLX/EGL context switching. Both runs report 1,823
dependency deprecation warnings. Logs: CPU (local evidence: `3d/pytest_cpu.txt`),
NVIDIA (local evidence: `3d/pytest_host.txt`). CUDA examples completed on both accelerated
backends with 11,056 analytical 3D particles and a 40³ volume sampling grid.
The wheel was rebuilt and its packaged sources, README, and MIT license were
checked against the checkout; no remote CI or release is claimed.

Regression coverage includes backend selection and missing dependencies,
covered zero versus void in a real volume, raw-data preservation, scalar/palette
updates, grid mode and resolution changes, empty-view recovery, compatible
camera links, slice/3D round trips, and front/rear occlusion independent of
particle order. Occlusion checks cover perspective and orthographic cameras
at domain scales 0.001, 1 and 1000 on both renderers. The full suite also covers
the existing 2D frontend snapshots, streamlines, automatic sizing and live
GLX/EGL context switching.

Remaining scope:

- Native 3D streamline integration, free-surface mesh reconstruction, arbitrary
  oblique planes, and Vispy 3D volume rendering are not implemented.
- Existing slices are axis-aligned 2D panels. There is no interactive clipping
  plane widget or embedded slice plane in the 3D scene yet.
- This pass does not integrate a 3D frontend simulation or exercise manual
  mouse/keyboard use, trame browser input, or long 3D replay memory behavior.
- PyVista's automatic 3D point diameter is projected at a representative focal
  depth; Vispy scales markers in scene coordinates. A perspective PyVista
  cloud with a very large depth range can still have approximate physical
  marker sizes. No per-particle sphere meshes are allocated.
- CPU preparation remains significant at a million particles. Automatic
  selection checks installed dependencies and capabilities; graphics-runtime
  errors are reported rather than silently changing renderer. A plot that
  grows past the threshold after creation retains its existing backend.

Reproduce with the optional plotting dependencies installed:

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
python examples/three_dimensions.py --backend pyvista --device cuda --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/output/3d"
EGL_PLATFORM=surfaceless python examples/three_dimensions.py --backend vispy --device cuda --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/output/3d"
python examples/benchmark_scaling.py --backend pyvista --dim 3 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/output/scaling"
EGL_PLATFORM=surfaceless python examples/benchmark_scaling.py --backend vispy --dim 3 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/output/scaling"
python examples/benchmark_scaling.py --backend auto --dim 2 --counts 100000 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/output/auto"
pytest -q tests
```

Use `--device cpu` without CUDA. Headless hosts may need
`VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow` and `EGL_PLATFORM=surfaceless`.
CI includes the new numerical/routing tests in the CPU job and actual rendering
in its rendering job; no remote CI run is claimed.

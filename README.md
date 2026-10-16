# warpSPHPlotting

Stateful visualization of SPH particle fields with Matplotlib, PyVista, and
Vispy. The same `visualize()` / `PlotState` API supports particle coloring,
SPH-derived quantities, regular grids, streamlines, updates, and export.

## Install

```bash
pip install -e .                       # CPU/Matplotlib plotting
pip install -e '.[plot-pyvista]'        # VTK renderer; optional trame notebooks
pip install -e '.[plot-vispy]'          # OpenGL renderer; optional inline widget
pip install -e '.[notebooks]'          # Matplotlib widgets
pip install -e '.[dev,plot-all]'        # tests and all rendering backends
```

`warpSPHCore`, PyTorch, Warp, NumPy, and Matplotlib are shared dependencies.
Matplotlib supplies normalization and colormaps for every backend. Initialize
Warp (`import warp as wp; wp.init()`) before requesting SPH operations or grids.
The `rocket`, `mako`, `flare`, `crest`, `vlag`, and `icefire` maps additionally
require seaborn; standard Matplotlib maps work without it.

## Start with a runnable example

```bash
python examples/quickstart.py --backend matplotlib
python examples/quickstart.py --backend pyvista
EGL_PLATFORM=surfaceless python examples/quickstart.py --backend vispy
```

The self-contained example creates a periodic Taylor–Green analytical field,
compares particles with an interpolated grid and streamlines, updates its
amplitude, and writes PNGs and physical VTK arrays under `output/quickstart/`.
`demo.ipynb` runs the same example from a fresh kernel. This analytical example
is separate from the actual frontend simulations described below.

For hands-on interaction, install `pip install -e '.[plot-all]' jupyterlab jupyter-server-proxy`
and launch `jupyter lab interactive.ipynb` from this
directory. [The review notebook](interactive.ipynb) provides PyVista/trame and
Vispy widgets with case, field, grid/streamline, sizing and camera controls.
It replays saved frontend frames when available and labels its analytical
fallback explicitly. Playback slows down for expensive updates. Save view +
diagnostics writes a PNG and camera/update history in the external `warpSPHDevelopment/plotting/output/notebook-review/` directory.
The view stays fixed during playback. A new case or changed domain refits it;
enable **Follow bounds each frame** for automatic fitting on every update.
[Browser verification and preparation benchmarks](reports/NOTEBOOK_INTERACTIVITY.md)
document the tested interactions and remaining limits.

```python
from warpSPHPlotting import visualize, PlottingOptions, Mapping, Backend

plot = visualize(
    particleState, domain,
    quantities=velocity,
    plotOptions=PlottingOptions(mapping=Mapping.L2, plotTitle="Speed"),
    backend=Backend.PyVista,       # strings are also accepted
    backendOptions={"jupyter_backend": "static"},
)
plot.updateQuantities(newVelocity, newParticleState=newParticles,
                      newDomain=newDomain, redraw=False)
plot.export("speed.png")
plot.close()
```

For multiple panels, use a mosaic such as `"AB"` or `"AA;BC"`, and dictionaries
keyed by the panel labels. A tensor updates all panels; a dictionary updates its
specified panels; `key="A"` targets one panel. Domain and particle-state changes
refresh every panel, including panels whose quantities were not replaced.
Per-panel option overrides use `newOptions={"A": {"markerSize": 4}}`.
Unknown keys/options fail explicitly. Input particle state is never modified.

The default `backend="auto"` uses Matplotlib for small 2D plots. Native 3D
views, or at least 50,000 visible scatter particles summed across panels,
prefer installed PyVista, then Vispy. Native 3D grids require PyVista.
Single-axis sharing prefers Vispy, since PyVista links whole cameras and
requires matching `sharex`/`sharey`. Hidden
particles and grid-only interpolation inputs do not count toward this threshold.
This is a routing heuristic, not a particle limit or subsampling: every visible
particle is rendered. `plot.backend` reports the resolved choice. It stays fixed
for the figure's lifetime; explicit backend choices are always honored. If no
accelerated backend is installed, large 2D plots warn and retain Matplotlib;
native 3D reports the required extra. Graphics initialization errors propagate.
Use an explicit backend when passing renderer-specific `backendOptions`.

Particle markers now size automatically when `markerSize=None` (the default).
The diameter follows physical particle volume `(mass / density) ** (1 / dimension)`,
with separate estimates for fluid and boundary particles. Rearrangement alone
does not shrink the markers. Use `PlottingOptions(markerSpacing="neighbors")`
to cap sizes by sampled close-neighbor distances when reducing overlap matters
more than keeping a steady diameter. The default is `markerSpacing="volume"`. Empty
domain area does not inflate markers. Sizes follow zoom, window layout and
export resolution. `PlottingOptions(markerSizeScale=0.5)` makes automatic
markers smaller than the default `0.65` fraction of spacing.

Set `markerSize=9` to retain a fixed Matplotlib scatter size in points squared.
GPU backends convert this using Matplotlib's `sqrt(s)` marker diameter at
96 dpi. An explicit GPU `backendOptions={"point_size": 3}` instead sets a
fixed pixel diameter; per-panel `markerSize` takes precedence.
[Examples, movie and performance measurements](reports/PARTICLE_SIZING.md)
compare fixed and automatic sizing across all three backends. Multi-panel plots
share spacing estimates within each update; every update starts a fresh cache.
The [Marrone sizing follow-up](reports/PHYSICAL_PARTICLE_SIZING.md) compares both
rules over all 51 recorded frames. The notebook exposes them under **Size rule**.

`visualizeParticlesNew(fig, ax, ...)` and `updatePlot(state, ...)` remain
available for existing Matplotlib figures. They use the shared preprocessing
and normalization rules. A vector quantity needs a component or norm mapping;
column scalars `(N, 1)` are accepted. Per-kind tuple entries may be compact or
full-length arrays; `None` denotes zero-valued samples for that kind.

## Rendering capabilities

| Capability | Matplotlib | PyVista | Vispy |
|---|---|---|---|
| 2D particle fields | Yes | Yes | Yes |
| Automatic particle size / zoom tracking | Yes | Yes | Yes |
| 3D point clouds | Use PyVista/Vispy | Yes | Yes |
| 2D SPH grids | Yes | Yes | Yes |
| Native 3D grid surface | No | Yes | No |
| Native 3D volume ray casting | No | Yes | No |
| Physical 3D planar slices | Yes | Yes | Yes |
| 2D vector streamlines | Yes | Yes | Yes |
| Axis ticks / labels | Yes | Yes | Yes, in 2D |
| Notebook display | Matplotlib/widgets | Static/trame | Image/widget |
| Image export | Matplotlib formats | Raster/vector capture | Pillow raster formats |
| Physical VTK data export | Bundled backend-independent writers | Same | Same |

Grid plots use `GridVisualization(maskEmpty=True)` by default: cells without
particle-kernel coverage are transparent, genuine zeros remain colored, and
streamlines stop at uncovered regions. Empty views hide their colorbar until
coverage returns. `maskEmpty=False` restores the earlier filled background.
This display mask preserves raw SPH interpolation values; it does not redefine
the physical free surface. [Rendered comparisons and reliability measurements](reports/GRID_COVERAGE.md)
cover all four saved frontend cases and all three backends.

Streamlines use the shared 2D grid integrator. They require vector data before
scalar mapping, or an appropriate `streamLineOperation`. Grid resolution and
scatter/grid/visibility changes are supported during updates. Three-dimensional streamline integration is not implemented. For 3D inputs,
`GridVisualization(sliceAxis=2, slicePosition=0.5)` interpolates a planar slice
in physical 3D space and projects the result onto its remaining two axes for
any backend. The slice position defaults to the domain midpoint; vector
streamlines use the in-plane components. Native 3D grids remain available in
PyVista without a slice. Slices default to the particles' physical kernel
supports, avoiding gaps when the slice lies between particle layers.

Native PyVista 3D grids now default to volume ray casting, so interior values
are visible instead of only the enclosing grid faces:

```python
options = PlottingOptions(
    gridVisualization=GridVisualization(resolution=48, volumeOpacity=0.15),
)
plot = visualize(particles3d, domain3d, density, options)  # auto -> PyVista
```

`renderMode="surface"` restores the exterior-face view. Volume opacity is
specified per tenth of the domain diagonal, so refining the grid does not
also increase opacity. Uncovered/invalid samples are transparent; genuine zeros
remain visible. The ray caster uses 255 scalar color entries plus a transparent
entry, while the mesh keeps raw physical `quantity` and normalized
`display_quantity` arrays. Translucent volume colors combine along each ray.
This is field volume rendering, not a reconstructed free-surface mesh.

Run `python examples/three_dimensions.py` for an analytical 3D Taylor–Green
field with particle, volume, orbit, and moving-slice exports; `--backend vispy`
checks native particle clouds and slices there. `--device cuda` also exercises
3D interpolation on CUDA. [3D examples and scaling measurements](reports/THREE_DIMENSIONS.md)
include full-population benchmarks through one million particles.

String mosaics support blank `.` cells and rectangular panel spans. Vispy links
2D cameras according to `sharex` / `sharey`. PyVista links full cameras when both
are enabled; independent single-axis camera linking is not implemented there.
Ordinary quantity updates retain the camera; changed domains refit it.
Panels with different dimensions or slice axes have independent cameras even
when sharing is enabled. Compatible cameras can still be linked. Returning
from a planar slice to a native 3D view restores an isometric camera.

## Color semantics

- Fluid and actively colored boundaries share one norm. Passive particles are
  gray and do not affect limits.
- `Symmetric` defaults to center zero; `midPoint="median"` is resolved on each
  frame. `SymmetricLog` retains signed values.
- NaN/Inf values are masked from automatic limits. Logarithmic rendering also
  masks nonpositive samples. All-invalid data requires explicit valid bounds.
- Constant fields receive a finite nonzero display range.
- PyVista meshes preserve physical `quantity` arrays separately from normalized
  `display_quantity` arrays. The colorbar uses inverse-normalized physical ticks.

PyVista uses upright colorbar labels beside the strip by default. It measures
their width to keep them inside each panel after resizing or changing limits.
Tick text actors persist across updates. Its options include:

```python
backendOptions = {
    "jupyter_backend": "static",      # static | trame | none (native window)
    "window_size": (1200, 600),
    "scalar_bar": {
        "orientation": "vertical",    # horizontal is also supported
        "label_rotation": 0,          # upright by default; rotation is optional
        "font_size": 12,
        "n_labels": 3,
        "label_format": ".6g",       # distinguish 0.997 from 1.003
        "title": "density",
        "title_rotation": 0,
        "title_font_size": 12,
    },
}
```

PyVista also accepts normalized viewport `position`, `width`, and `height`
inside `scalar_bar`. Default fonts scale with panel height; explicit font sizes
remain fixed. The layout reserves space around the data for the bar. In 2D,
a complete plotting frame surrounds the data with padding; the blue physical
domain outline stays inside that frame and above interpolated surfaces.
Default vertical bars follow the plot frame with a 12-pixel gap. Figures with
2D streamlines temporarily disable window MSAA to avoid white fringes around
the black lines; the previous setting returns when the overlay is removed.

Matplotlib defaults to constrained layout, which keeps titles and tick labels
inside the figure after resizing; pass a Matplotlib `layout` option through
`backendOptions` to override it. Its colorbar padding is reduced to 2.5%.

Vispy draws 2D domain outlines and black streamlines above grid images, reserves
a separate row for panel titles, and uses upright colorbar endpoint labels.
Its colorbar strip sits 12 pixels from the view frame. Offscreen EGL resizes
also update the widget layout, preventing cropped panels.

Vispy supports `jupyter_backend="native"`, `"notebook"`, or `"image"`, with
`app_backend="egl"` for offscreen OpenGL. On headless Mesa systems, set
`EGL_PLATFORM=surfaceless` before Python starts. Native-window updates were
separately verified with CUDA data on the host display.

PyVista/trame notebooks use server rendering to preserve physical colorbar text
and axes; VTK.js client rendering does not reproduce those overlays and is
rejected explicitly. `backendOptions={"jupyter_backend": "trame",
"jupyter_kwargs": {"height": 600}}` adjusts the viewer; width defaults to
`"100%"`. Ordinary left dragging pans planar views; the wheel zooms. Native 3D
views retain trackball rotation. Browser checks and example captures are in the
[notebook report](reports/NOTEBOOK_INTERACTIVITY.md).
The [camera/frame crash follow-up](reports/NOTEBOOK_CAMERA_FIX.md) covers trame
rendering across kernel threads and its real frontend replay reproducer.
The [Vispy streamline follow-up](reports/VISPY_NOTEBOOK_STREAMLINES.md) covers
nested notebook draws, concurrent frame updates and a recorded Marrone replay.
After updating the library, restart the notebook kernel and run all cells to
replace its existing Python classes and callbacks.

All backends retain the camera on ordinary frame updates. Actual domain bounds
or slice-axis changes refit the view; loading a new case can request a one-off
refit with `plot.updateQuantities(values, resetView=True)`. Opt into continuous
fitting with `PlottingOptions(dynamicBounds=True)` or
`plot.updateQuantities(values, dynamicBounds=True)`. This includes visible
particle extents without changing the physical domain used for interpolation.
See [viewport verification and examples](reports/VIEWPORT_RETENTION.md).

## Export physical data

No PyVista/Vispy runtime is required by the bundled XML writers:

```python
from warpSPHPlotting import (
    export_particles_vtp, export_grid_vti, write_time_series_pvd,
)
export_particles_vtp("particles.vtp", particleState,
                     {"velocity": velocity, "density": density}, domain=domain)
export_grid_vti("grid.vti", gridState, {"field": gridQuantity}, resolution, extent)
write_time_series_pvd("series.pvd", [(0.0, "particles.vtp")])
```

The writers preserve numerical arrays and grid ordering, validate coordinates,
and support optional JSON metadata. `resolution` describes grid points, and VTI
origin/spacing are recovered from the actual sampling positions. The legacy
`vtk-export` extra remains available without adding dependencies.

Vispy image writes can run in background workers:

```python
from warpSPHPlotting import waitForExports
plot.export("frame.png", background=True)
waitForExports()                     # waits for pending writes and reports failures
```

Call the drain before reading frames or encoding a video. `dpi` changes raster
resolution for Matplotlib; Vispy stores it as metadata, and PyVista uses
`window_size` for pixel dimensions.

## Verify actual frontend simulations

`examples/verify_frontends.py` uses the sibling `warpSPH` installation's actual
case construction, integration, diagnostics, and output scheduler. It replaces
only plotting hooks so identical fields can be compared across renderers.
`warpSPH` is an optional verification dependency, not a plotting dependency.

```bash
python examples/verify_frontends.py --case tgv --backend pyvista --steps 40
python examples/verify_frontends.py --case sod --backend pyvista --steps 40 --sparse-view
python examples/verify_frontends.py --case marrone31 --backend pyvista --steps 300 --checkpoint
python examples/verify_frontends.py --case khh --backend pyvista --steps 40
```

`--sparse-view` expands only the plotting domain around the real Sod solution;
it tests an increasingly empty viewport without changing the solver's physical
boundary conditions. The Marrone case uses the section 3.1 tank proportions,
free-slip walls, and Sun-2017 delta-SPH. These bounded runs verify visualization
and usability; they do not establish shock accuracy, convergence, or agreement
with experimental impact-pressure records.

Use `--checkpoint` to save a complete frontend state to `checkpoint.h5`; resume
with `--resume PATH --resume-offset STEP` and identical case settings.

Use `--native-hooks` to exercise the frontend's own plotting hooks; the KHH
check uses `--grid-resolution 128` by default instead of its expensive 1024
grid. Omit `--native-hooks` for identical speed/density panels across backends.

Each run writes PNGs, a final-state NPZ, and JSON timing/diagnostic data under
the external `plotting/reports/frontend/` directory. Reuse those samples without the frontend installation:

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
python examples/render_gallery.py --backend pyvista \
  --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/frontend/tgv-pyvista/final_state.npz" --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/tgv.png"
python examples/benchmark_snapshots.py --backend pyvista \
  --input "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/frontend/tgv-pyvista/final_state.npz" \
  --count 100000 --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/reports/benchmark.json"
```

Benchmarks separate preparation from rendering and exclude PNG compression.
Run comparisons serially, with identical scenes and thread counts. Repeated
samples at larger counts test throughput, not a higher-resolution simulation.

## Tests

```bash
MPLBACKEND=Agg pytest -q tests/test_correctness.py
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow EGL_PLATFORM=surfaceless MPLBACKEND=Agg pytest -q tests
```

Backend tests explicitly skip missing optional libraries. The rendering CI job
installs both backends and software EGL. The optional PyVista extra uses `pyvista[jupyter]` so its notebook adapter is
installed alongside the renderer. CUDA and native-display checks have separate
reproduction scripts; browser interaction remains outside the CPU CI scope.

Released under the [MIT License](LICENSE). The license file and package metadata
use the same license.

Extended frontend trajectories, videos, and 6,000 renderer replay updates are
documented in [the long-run verification report](reports/LONG_RUNS.md).
`examples/verify_frontends.py --snapshots` saves temporal states;
`examples/stress_replay.py` replays them through each backend with resource,
camera, mode-switching, and export checks. `examples/stress_colorbars.py`
isolates repeated Vispy colorbar visibility changes.

The [grid coverage verification](reports/GRID_COVERAGE.md) also covers native windows,
CUDA numerical parity, four short CUDA frontend runs and the NVIDIA GLX/EGL
context-handoff regression.

The [notebook follow-up](reports/NOTEBOOK_INTERACTIVITY.md) adds actual browser
interaction, notebook context/resize regressions and faster CPU preparation.

Generated verification artifacts are kept outside Git under the sibling
`warpSPHDevelopment/plotting` directory. Set `WARPSPH_DEVELOPMENT_ROOT` to
relocate the shared root; tools append `plotting`. The notebook discovers its
recordings there and falls back to an explicitly labeled analytical example
when they are absent. [Storage policy and regeneration commands](reports/README.md)
explain how to recreate evidence without growing the source repository.

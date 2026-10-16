# Correctness, Backend Reliability, and Feature Completion Plan

Status: implementation and local CPU/headless verification completed (2026-10-16).

Evidence, rendered examples, actual frontend trajectories, benchmark samples,
and remaining host checks are recorded in `reports/VERIFICATION.md`.
The current suite passes 132 tests on the NVIDIA host (130 plus two explicit
display-dependent skips under the CPU sandbox); a wheel builds. Native 3D, volume rendering,
automatic backend selection and million-particle timings are documented in
[the 3D follow-up](reports/THREE_DIMENSIONS.md). The copyright holder authorized MIT,
and `LICENSE` now matches the package metadata. Native windows with CUDA data, short CUDA frontend runs and trame startup
have since been verified outside the sandbox. Actual notebook input, replay,
camera retention, widget export and CPU preparation optimization are covered in
[the notebook follow-up](reports/NOTEBOOK_INTERACTIVITY.md). The extended Marrone run reaches wall
impact at nx=33; it does not establish
experimental impact accuracy. See `reports/LONG_RUNS.md` for the extended
trajectories, 6,000 replay updates, resource measurements, and fixes found.
Spatial search reuse is frame-local, with a fresh cache on every update.
The [Vispy notebook follow-up](reports/VISPY_NOTEBOOK_STREAMLINES.md) fixes nested
widget draws and verifies streamlines against recorded Marrone frames.
The [viewport follow-up](reports/VIEWPORT_RETENTION.md) preserves cameras during
frame updates, adds explicit/dynamic fitting and fixes the Sod replay viewport.
The [physical sizing follow-up](reports/PHYSICAL_PARTICLE_SIZING.md) keeps marker
sizes stable under rearrangement and makes close-neighbor limiting optional.
Generated evidence is archived outside Git under `warpSPHDevelopment/plotting`;
[storage and regeneration](reports/README.md) describes the retained scripts.


Baseline: repository survey of version 0.6.0, commit `bc24ad6`.
This document replaces the earlier backend expansion plan, which predates much
of the current implementation.

## Outcomes

1. Plot colors and colorbar labels faithfully represent the selected quantity.
2. Creating and updating a plot follow the same preprocessing rules.
3. Quantity, domain, particle-count, visibility, and rendering-mode changes
   either work or fail with a precise explanation before corrupting plot state.
4. All three backends have documented, tested installation and lifecycle paths.
5. PyVista remains suitable for fast simulation updates, including after fixes
   to normalization and colorbar layout.
6. Advertised capabilities, examples, exports, and package metadata match the code.

Public entry points should remain compatible where their behavior is defined.
Silent failures and incorrect numerical behavior should be corrected directly.
New behavior and intentional limitations should be documented alongside the fix.

## Evidence and Validation Rules

The survey reproduced numerical and Matplotlib update bugs on CPU. Real SPH
interpolation, gradient calculation, streamlines, updates, and PNG export also
passed CPU smoke checks. Vispy scene-state checks ran without a graphics
context. PyVista findings were identified from source and need runtime checks.

- Turn each substantive reproduced failure into a regression test with an
  observable numerical or user-visible outcome.
- Cover numerical helpers and state transitions without requiring a display.
- Exercise actual rendering for backend integration and visual acceptance.
  Mocked actors and changed property values do not establish visual correctness.
- Keep unavailable-dependency skips explicit. A skipped backend rendering job
  must not be reported as a passing backend validation.
- Use small deterministic particle fixtures for correctness; use larger separate
  fixtures for performance measurements.
- Validate a fix before expanding the refactor around it. Keep changes in small,
  reviewable commits with tests and relevant documentation.

## 1. Establish Regression Coverage and a PyVista Rendering Baseline

### Work

- Add a conventional `tests/` layout and shared scalar/vector particle fixtures.
- Add CPU/headless Matplotlib fixtures and optional-backend test markers.
- Capture the survey failures before changing production behavior.
- Provide a reproducible PyVista rendering environment and record Python,
  PyVista, VTK, rendering backend, and notebook frontend versions.
- Add a minimal PyVista colorbar reproducer independent of SPH computation.
- Record initial creation time and steady-state update time for each available
  backend, with PyVista as a priority given its observed speed in user workflows.

### Acceptance

- Pytest discovers meaningful tests instead of collecting zero tests.
- Confirmed bugs fail for their expected reason before their fixes.
- PyVista screenshots and a repeatable performance baseline are available before
  changing its scalar mapping or scalar-bar implementation.

## 2. Repair Normalization and Quantity Validation

Primary files: `math.py`, `state.py`, `options.py`.

### Work

- Preserve negative bounds for symmetric-log scaling; remove the positive-only
  lower-bound repair from that path.
- Resolve the default symmetric center to zero.
- Keep requested options separate from resolved per-frame normalization values.
  In particular, `midPoint="median"` must be recalculated on every frame.
- Define finite-value handling: exclude invalid values from automatic bounds,
  render invalid samples consistently, and give a clear error when no usable
  values or valid explicit bounds exist.
- Define logarithmic behavior for nonpositive samples and invalid limits.
- Handle constant fields, empty populations, and explicitly supplied bounds.
- Validate quantity length, scalar/vector/tensor rank, and mapping components
  before rendering or mutating an existing plot.
- Define a single per-type tuple contract. Support compact per-type arrays and
  `None` entries consistently, validate tuple length, and account for absent
  particle kinds without indexing nonexistent entries. Preserve any established
  full-length-array form explicitly if needed for compatibility.
- Keep original physical values available independently of display transforms.

### Acceptance

- Signed symmetric-log inputs retain distinct negative, zero, and positive
  colors; symmetric defaults work without specifying a midpoint.
- Median centers follow changed data without mutating the requested setting.
- Empty fluid with visible boundary particles works.
- NaN/Inf, constant values, invalid mappings, and malformed tuples have deliberate,
  tested outcomes.
- Equivalent physical values use equivalent colors across populations/backends.

## 3. Unify Frame Preparation and Public Update Semantics

Primary files: `visualize.py`, `update.py`, `state.py`, `domain.py`,
`backends/_render_util.py`, `backends/base.py`.

### Work

- Introduce one internal prepared-frame representation containing the effective
  domain, transformed particle state, scalar quantity, per-kind populations,
  resolved normalization, and optional streamline vector field.
- Route legacy Matplotlib creation/updates and other backend paths through it
  incrementally, retaining public wrappers.
- Apply rotation and periodic wrapping consistently to positions and vector
  quantities. Grid interpolation must use the matching transformed particles.
- Define quantity update dispatch:
  - a shared tensor updates every panel that uses that shared quantity;
  - a dictionary updates its specified panels;
  - an explicit key targets that panel regardless of the original input form;
  - unknown keys and incompatible combinations raise useful errors.
- Validate an update before changing stored quantities/options. Avoid leaving
  bookkeeping claiming success when preparation fails.
- Make domain updates part of the backend contract and update every affected
  panel's effective domain, geometry, and bounds.
- Replace deep copies of particle tensors with an explicit visualization
  snapshot policy that supports non-leaf autograd tensors and does not mutate
  the caller's state.
- Preserve a user's interactive camera during ordinary quantity updates; refit
  only for domain changes, initial creation, or an explicit reset request.
- Add a consistent close method for figure/window resources.

### Acceptance

- Shared-tensor and dictionary multi-panel updates both change displayed data.
- Keyed tensor updates work; invalid requests leave stored state consistent.
- Initial and updated rotated plots use the same coordinate convention.
- Domain changes propagate through both public update methods on every backend.
- Differentiable particle positions can be plotted without a deepcopy error.
- Repeated updates preserve an intentional zoom/pan and do not mutate input state.

## 4. Make Backend Artist Lifecycles Correct

Primary files: `scatter.py`, `grid.py`, `streamLines.py`, `domain.py`,
`backends/matplotlib_backend.py`, `backends/pyvista_backend.py`,
`backends/vispy_backend.py`.

### Work

- Give each panel ownership of its artists, actors, colorbar, domain outline,
  title, and streamline overlay.
- Implement create/update/hide/show/remove transitions, including populations
  that become empty and later reappear.
- Apply passive styling, marker-size, title, domain-outline, and colorbar changes
  to existing objects as well as newly created ones.
- Remove or detach obsolete objects when changing scatter/grid mode.
- Rebuild grid geometry when resolution, dimensionality, origin, or spacing
  changes; keep the value-only fast path when geometry is unchanged.
- Refresh Vispy image transforms when grid geometry changes.
- Rebuild PyVista point-cloud connectivity when particle counts change, while
  retaining efficient same-count point/scalar updates.
- Create streamlines when enabled, replace them when updated, and remove their
  owned lines/arrows when disabled. Preserve unrelated user annotations.
- Update the existing Matplotlib domain rectangle instead of adding one per frame.
- Fix Vispy colorbar updates when there is no active colored population; support
  creating/removing colorbars after initial rendering.
- Correct automatic marker sizing to use particle spacing in the appropriate
  dimension, and validate consistent explicit marker-size semantics.

### Acceptance

- A transition matrix covers hidden/active/passive, empty/nonempty, colorbar
  on/off, outline on/off, and scatter/grid changes.
- Enabling or disabling streamlines after creation works in Matplotlib.
- Domain/resolution changes update geometry, not just scalar buffers.
- Artist/actor counts remain bounded across repeated updates and mode switches.
- Particle-count changes display exactly the current particles.

## 5. Correct PyVista Scalar Mapping and Colorbar Presentation

Primary file: `backends/pyvista_backend.py`.

### Numerical Mapping

- Centralize scalar mapping and scalar-bar creation/update for fluid, boundary,
  and grid paths.
- Ensure logarithmic, symmetric, and symmetric-log norms affect actual colors.
  Select a VTK lookup-table strategy or explicit normalized display array after
  checking supported runtime APIs.
- Preserve physical scalar arrays and label the bar in physical units even if
  rendering uses transformed values. Updating colors alone is insufficient.
- Attach exactly one shared scalar bar to the active colored populations,
  including boundary-only plots.
- Update lookup tables and label ranges only when necessary.

### Label Rotation and Layout Investigation

The user reports that PyVista colorbar labels are incorrectly rotated and
visually poor, and previous attempts have been difficult. Treat this as a
rendering investigation with its own acceptance gate.

1. Reproduce the current failure in a small scene and capture rendered output.
2. Identify which element is wrong: tick text, title text, exponent/offset text,
   or overall bar orientation. Check them separately.
3. Inspect the actual scalar-bar actor and text properties before and after
   rendering. Determine whether VTK's automatic layout or actor replacement
   overwrites the requested orientation, font sizing, or positioning.
4. Check single-panel and mosaic scenes, resize behavior, static export, and
   the relevant native/notebook display modes. Record differences rather than
   assuming the same property behaves identically everywhere.
5. Implement one shared scalar-bar style helper, applied during creation and
   relevant updates. Avoid scattering rotation fixes across `add_mesh` calls.
6. Prefer VTK's native text/layout controls. If they cannot meet the intended
   result, evaluate a narrowly scoped custom-label overlay with explicit
   positioning and resize handling.
7. Expose only useful presentation settings through backend options: bar
   placement/orientation, tick/title text orientation, font sizes, numeric
   format, and label count. Validate supported settings.

Limit the initial investigation to one focused diagnostic pass and one native
implementation pass. If native layout remains unreliable, record the exact
failing configuration and assess the overlay approach separately. Continue
other fixes while this work is isolated; do not mark the issue fixed without
rendered evidence.

### Visual Acceptance

- Agree on the intended appearance from a concrete before/after comparison;
  the exact desired tick/title angles are not yet specified.
- Labels are readable, correctly oriented, and unclipped in single/multi-panel
  plots and after resizing, quantity updates, and export.
- Exercise negative values, logarithmic ranges, scientific notation, and a
  narrow density interval such as `[0.997, 1.003]`.
- Scalar-bar labels and particle colors describe the same physical values.
- Repeated value updates retain scalar-bar actors where possible and avoid
  rebuilding text/layout when nothing relevant changed.
- Compare update timings to the baseline and investigate material regressions.

## 6. Repair Export and Grid Edge Cases

Primary files: `backends/asyncExport.py`, `grid.py`, export implementations.

### Work

- Retrieve completed futures' results before dropping them; retain/report write
  failures through the documented synchronization point.
- Drain all writes even if one fails, then report the failure. Keep queue
  backpressure correct under concurrent submissions.
- Validate grid resolution and nonzero domain extents. Ensure every requested
  axis has a valid point count, including thin domains.
- Reconcile edge-aligned point spacing with returned spacing/support volume.
- Make raster export format behavior explicit, including alpha-channel handling
  for JPEG and unsupported formats/options.

### Acceptance

- Early failed writes remain observable after later successful submissions.
- Draining exports waits for all previously submitted work before returning or
  reporting failure.
- Thin domains and each documented alignment produce finite, consistent grids.
- Supported export formats open successfully and contain the intended frame.

## 7. Packaging, Documentation, and CI

### Work

- Declare the compatible `warpSPHCore` dependency and document its installation.
- Make package import work with the documented base installation. Keep backend
  imports lazy, and declare shared normalization/colormap dependencies wherever
  they are required by backend extras.
- Normalize backend names using enum values; validate backend options and emit
  actionable missing-dependency errors.
- Check capability flags before accepting unsupported requests. Unsupported
  streamlines or dimensions must not silently disappear.
- Add isolated installation checks for base and each plotting extra.
- Add CPU numerical/Matplotlib CI, optional backend rendering jobs, and a clearly
  identified CUDA validation path where available.
- Rewrite README examples around `visualize()` / `PlotState` while retaining
  legacy API examples and a backend feature matrix.
- Repair fresh-kernel notebook execution, including `prepData`, optional
  comparison dependencies, seaborn colormaps, and practical default data sizes.
- Correct package description and keywords.
- Reconciled the license file and package metadata as MIT after confirmation
  from the copyright holder. Do not infer a licensing change from the
  survey. This decision does not block correctness work.

### Acceptance

- Installation instructions work in clean supported environments.
- Enum and string backend selection behave identically.
- CI distinguishes numerical, state, rendering, and environment failures.
- A new user can run one small example per backend with documented dependencies.

## 8. Complete Advertised Feature Gaps

Schedule these as separate increments after the shared correctness/lifecycle
work, so additional features inherit the repaired behavior.

- Implement lightweight particle `.vtp`, regular-grid `.vti`, and time-series
  `.pvd` export; verify arrays, physical units, coordinates, ordering, and times
  with an independent reader and ParaView smoke check. Ensure the chosen writer
  actually supports each promised format.
- Add PyVista grid-vector streamlines with documented seeding and domain behavior.
- Add Vispy streamlines using a tested integration path and line visuals.
- Add Vispy axis labels/ticks and camera linking; define cross-backend sharex /
  sharey behavior and clearly document any 3D limitation.
- Implement consistent mosaic parsing for blank cells and rectangular spans,
  rejecting unsupported/nonrectangular arrangements explicitly.
- Define dimension support: retain working 2D grids, add explicit slice/projection
  controls for higher-dimensional data, and implement native 3D grid rendering
  where the backend supports it. Avoid silently treating 3D arrays as 2D.

### Acceptance

- Each feature has a minimal example, capability declaration, and integration test.
- Exported files round-trip with unchanged values and geometry.
- Unsupported combinations have clear errors until implemented.

## 9. Measure and Optimize the Working Pipeline

### Work

- Benchmark 30k and 100k particles, scalar-only updates, moving particles,
  grid remapping, multiple panels, and image export.
- Measure preprocessing, SPH operations, transfers, rendering, and encoding
  separately, with warm-up and relevant GPU synchronization.
- Compare all backends on the same data, viewport, and visible features.
- Reuse unchanged grids and neighbor-search structures where validity can be
  established; invalidate caches on geometry/domain/support changes.
- Avoid repeated deep copies, duplicate normalization, unnecessary transfers,
  and rebuilding colorbar resources on every frame.
- Preserve correctness tests while optimizing. Record environment and particle
  distributions with results, rather than asserting a universal fastest backend.

### Acceptance

- Published timings distinguish creation cost from steady-state update cost.
- PyVista correctness and label fixes preserve its useful fast update path.
- Each optimization has measured benefit and explicit cache invalidation behavior.

## Delivery Order

1. Regression coverage and PyVista visual/performance baseline.
2. Numerical normalization and input validation.
3. Shared preparation and update/domain dispatch.
4. Backend lifecycle fixes.
5. PyVista mapping and colorbar presentation; investigate label layout early,
   with final acceptance after the lifecycle and mapping changes settle.
6. Export/grid reliability, packaging, documentation, and CI.
7. Feature completion in independently reviewable increments.
8. Performance improvements supported by measurements.

Correctness fixes can proceed without waiting for uncertain visual-layout work.
The PyVista label issue remains an explicit tracked deliverable with rendered
acceptance, rather than an assumed one-line rotation change.

Automatic particle sizing is implemented across all three backends, including
independent fluid/boundary spacing, local compression, zoom/resize/DPI tracking,
and explicit overrides. See [the verification report](reports/PARTICLE_SIZING.md).

The [grid coverage and resource follow-up](reports/GRID_COVERAGE.md)
adds transparent unsupported cells, shared sizing work and measured Vispy
transform reuse, with examples and another 3,000 temporal replay updates.

Subsequent host checks accessed the native display and CUDA outside the sandbox,
verified native updates and short CUDA frontend runs, and fixed a GLX/EGL
handoff failure. Trame startup and dependency installation were also checked.
See the grid coverage follow-up for exact scope and remaining browser checks.

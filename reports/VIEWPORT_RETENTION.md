# Keep the camera during frame updates

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Implemented and verified on 2026-10-16 for Matplotlib, PyVista and Vispy.

Ordinary particle/quantity updates retain the camera. View bounds are fitted
on creation, when the actual domain extents/dimension or displayed slice axes
change, or when explicitly requested. Equal-valued replacement domain objects
do not cause a refit. In-place changes to the caller's domain tensors are
detected against the renderer's previous snapshot.

`PlottingOptions(dynamicBounds=True)` opts into fitting the domain and visible
particle extents on every frame. It defaults to `False`; its display bounds
are separate from the physical domain used for periodic wrapping and SPH/grid
interpolation. `plot.updateQuantities(values, dynamicBounds=True)` enables it
on an existing plot; `False` returns to retaining the camera. A one-off
`plot.updateQuantities(values, resetView=True)` refits without enabling following.
The new dataclass option is appended to preserve existing positional arguments.

The notebook requests a one-off refit when loading a different case, including
cases with identical domain bounds. It has a **Follow bounds each frame**
checkbox, disabled by default, and includes its value in saved diagnostics.
PyVista also retains the camera when the colorbar appears/disappears; that
previously adjusted zoom and window center independently of domain changes.

## Sod's animated display bounds

The recorded Sod stress test deliberately expands its display domain every
frame (41 different bounds across 41 snapshots). The notebook was replaying
those as genuine domain changes, repeatedly triggering camera fitting.
It now caches the initial sparse viewport for that case. Following explicitly
enables the animated stress-test viewport. Other cases continue to honor their
recorded physical domain changes; their domains are not frozen globally.

## Verification and examples

The cross-backend regression zooms the view, updates particle buffers and
quantities with equal-valued domains, and toggles the colorbar while checking
exact camera retention. It then checks genuine domain mutation, one-off refit,
dynamic fitting of particles outside the physical domain, and disabling that
mode. A separate notebook test checks fixed/animated Sod loading and explicit
case-change refits.

[`examples/verify_viewport.py`](../examples/verify_viewport.py) renders real
saved frontend states for Sod (frames 0, 11, 40, gridded) and Marrone 3.1
(frames 0, 25, 50, streamlines) through all three backends. It verifies **18
retained-camera updates and 12 requested refits** and saves the rendered PNGs.
Structured results (local evidence: `viewport/replay/verification.json`),
execution log (local evidence: `viewport/replay_host.txt`).

Sod frame 40 retaining the initial user zoom, Vispy (local evidence: `viewport/replay/vispy-sod-0040-retained.png`)

Marrone frame 50 retaining the initial user zoom, PyVista (local evidence: `viewport/replay/pyvista-marrone31-0050-retained.png`)

The complete suite passes **123 tests on the NVIDIA host**, and **121 with two
explicit GLFW/display skips in the CPU sandbox**, with 1,914 dependency warnings
in both runs. Host log (local evidence: `viewport/pytest_host.txt`),
CPU log (local evidence: `viewport/pytest_cpu.txt`). These checks use real renderer code and saved
simulation states; they do not constitute a new manual browser interaction test.

```bash
python examples/verify_viewport.py
pytest -q tests/test_notebook_interaction.py
```

**Restart the notebook kernel and run all cells** to load both the library and
the revised notebook controls.

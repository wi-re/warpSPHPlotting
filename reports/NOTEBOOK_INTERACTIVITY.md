# Notebook interaction and CPU preparation

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Verified locally on 2026-10-16. This follows the [3D/scaling pass](THREE_DIMENSIONS.md).

Subsequent user feedback reproduced a native camera-to-frame crash. The
[camera interaction fix](NOTEBOOK_CAMERA_FIX.md) adds scoped trame context
management and a 118-test host suite; the results below describe the earlier pass.

## Try the notebook

Open [interactive.ipynb](../interactive.ipynb) from the repository root. It
provides PyVista/trame and optional Vispy/jupyter_rfb widgets. Case selection
replays actual saved Taylor–Green, Sod, Marrone 3.1 and KHH frontend frames from
the external `plotting/reports/long_frontend/` directory; it does not run the solver in the UI. Missing recordings
leave an explicitly labeled analytical Taylor–Green fallback. The original
[demo.ipynb](../demo.ipynb) remains the short analytical quickstart.

```bash
pip install -e '.[plot-all]' jupyterlab jupyter-server-proxy
jupyter lab interactive.ipynb
```

Use a Python kernel with those dependencies. Vispy's widget needs an OpenGL
context provider, such as GLFW (`pip install glfw`) on this local machine.
The notebook starts trame on loopback through Jupyter's proxy. The original review ran Jupyter on `http://127.0.0.1:8890`, with token authentication
with a local browser session. Temporary dependency/kernel configuration lives
under `/tmp/warpsph-*`; the permanent Python environment was not changed.

The controls change case/frame, speed/density, particles/grid/streamlines,
palette, linear/symmetric-log scaling, marker size, boundaries and colorbar.
Streamlines explicitly select speed because their vectors are velocity.
Playback is paced to at least 500 ms and slows further based on the last Python
update time. This reduces queued stale frames; it is not a displayed-FPS or
network-latency guarantee. Stop playback before inspecting a frame closely.

Save view + diagnostics writes a PNG plus camera state and recent update history
to the external `plotting/output/notebook-review/` directory. Names include case, mode, field, frame and a unique
time suffix so repeated feedback captures do not overwrite one another.

## Browser findings and fixes

The actual JupyterLab browser pass used Python 3.13.14, JupyterLab 4.6.4,
ipykernel 7.4.0, PyVista 0.49.1, VTK 9.7.1 and the host NVIDIA RTX PRO 6000
Blackwell. The browser viewport was 1280 × 720, with the Jupyter file sidebar
open and about 831 pixels available for plot output.

- **Planar camera rotation:** ordinary left dragging rotated PyVista 2D data
  out of its plane. Planar scenes now use VTK's 2D pan/zoom style; scenes with a
  native 3D panel retain trackball rotation. Browser captures verify a changed
  focal point with unchanged viewing direction/up vector. Changing palette
  then preserves the entire camera exactly.
- **Clipped viewers and controls:** PyVista's 960-pixel iframe overflowed the
  cell. Trame and Vispy notebook viewers now default to 100% width, and example
  control rows wrap. Trame's actual DOM width was 831 pixels. Vispy receives the
  browser resize and exports at the resulting physical canvas size; the new
  regression also checks a 600 × 400 resize.
- **Missing client-rendered scientific labels:** switching to VTK.js client
  rendering dropped the custom physical colorbar labels and changed the axes.
  The captured failure (local evidence: `notebook/client_mode.jpg`) is retained as evidence.
  Trame now uses server rendering, and explicit client/hybrid mode requests
  fail before allocating a rendering window. Client overlay support remains
  an open feature gap.
- **Notebook export crash:** Vispy's GLFW context remained current after a
  widget draw. A callback on another kernel thread could then report
  `GLX: Failed to make context current`; the actual browser export killed the
  kernel. Notebook frame callbacks and explicit render/readback now release
  the context afterward. An isolated sequential-thread reproducer failed
  before the fix and completes five update/export cycles after it. The browser
  subsequently completed the same KHH grid export with both backends open.
  This does not authorize simultaneous rendering from arbitrary threads.
- **Playback backlog:** a fixed 200 ms timer was faster than the roughly
  315 ms Taylor–Green streamline update in this session. The review helper now
  adjusts its playback interval instead of continually issuing frames faster
  than the previous measured update.

Browser checks included PyVista Taylor–Green streamline playback/pause and
export; Sod density at its last frame and colorbar hide/show; KHH density grid
and export; Marrone wall-impact frame 50, automatic/fixed sizes and boundary
visibility. Vispy checks covered KHH density grid, frame selection, export,
mouse pan and palette changes. Its saved camera rectangle changes after pan
and remains exactly equal after recoloring. Both backends ran from fresh
kernels. This is a focused interaction pass, not exhaustive browser or device
coverage; perceived latency and long hands-on sessions still need user feedback.

Taylor–Green streamlines in the actual browser (local evidence: `notebook/tgv_streamlines.jpg`)

Vispy KHH grid after pan and palette change (local evidence: `notebook/vispy_khh_browser.jpg`)

Marrone wall-impact particle export (local evidence: `notebook/marrone_ready.png`)

Camera evidence: PyVista before pan (local evidence: `notebook/pyvista_pan_start.json`),
after pan (local evidence: `notebook/pyvista_pan_end.json`), after recoloring (local evidence: `notebook/pyvista_palette_end.json`);
Vispy before pan (local evidence: `notebook/vispy_pan_start.json`), after pan (local evidence: `notebook/vispy_pan_end.json`),
after recoloring (local evidence: `notebook/vispy_palette_end.json`).

## Preparation profile and performance

The original million-particle 2D preparation profile spent about 99 of 124 ms
in automatic spacing estimation. Short-row NumPy coordinate reductions and
temporary NxD arrays accounted for much of that work. The estimator now uses
contiguous coordinate columns and builds flattened spatial keys one axis at a
time. It retains the same particles, query/candidate order, sort and quantiles.
There is no new subsampling or cache carried across frames.

The retained [previous estimator](../examples/_reference_sizing.py) matches the new
one **exactly in all 106 cases**: 2D/3D, float32/64, empty/singleton populations,
compression, invalid/coincident coordinates, strided tensors, and saved real
frontend populations. Equality results (local evidence: `notebook/sizing_equality.json`).

Alternating old/new preparation in one process, with three warmups and twenty
measured updates each, reduced the median from **125.98 to 62.77 ms** for one
million 2D particles. Raw samples (local evidence: `notebook/preparation_comparison.json`).
The before (local evidence: `notebook/preparation_before.txt`) and after (local evidence: `notebook/preparation_after.txt`)
profiles show sorting remains a substantial cost.

End-to-end comparisons below render every particle, use two CPU threads,
768 × 576 output, three warmups and eight measured updates with synchronized
pixel readback. PNG encoding is excluded. Old/new workloads ran in separate
serial processes on the same host; these are session measurements rather
than an idle-machine or browser-FPS guarantee.

| Backend / dimension | Particles | Preparation before → after (ms) | Total before → after (ms) |
|---|---:|---:|---:|
| Matplotlib 2D | 100,000 | 18.1 → 11.8 | 181.3 → 174.0 |
| Matplotlib 2D | 1,000,000 | 125.7 → 67.4 | 1461.8 → 1396.8 |
| PyVista 2D | 100,000 | 14.0 → 7.6 | 19.3 → 12.7 |
| PyVista 2D | 1,000,000 | 127.0 → 67.4 | 149.4 → 88.8 |
| PyVista 3D | 100,000 | 18.1 → 11.6 | 22.8 → 16.9 |
| PyVista 3D | 1,000,000 | 139.0 → 74.7 | 156.2 → 91.7 |
| Vispy 2D | 100,000 | 16.7 → 9.9 | 22.2 → 15.3 |
| Vispy 2D | 1,000,000 | 174.6 → 114.1 | 187.6 → 126.2 |
| Vispy 3D | 100,000 | 20.2 → 13.2 | 24.2 → 17.1 |
| Vispy 3D | 1,000,000 | 189.8 → 119.2 | 201.1 → 130.1 |

All ten before/after image pairs are pixel-identical.
Pixel comparisons (local evidence: `notebook/sizing_pixel_equality.json`),
raw timings and images (local evidence: `notebook/benchmarks`).
Matplotlib still has a large draw cost at high counts; preparation optimization
does not remove that limitation. Grid interpolation and the browser transport
are separate costs from this scatter benchmark.

Reproduce the controlled comparisons:

```bash
export WARPSPH_DEVELOPMENT_ROOT="${WARPSPH_DEVELOPMENT_ROOT:-$PWD/../warpSPHDevelopment}"
python examples/verify_sizing_preparation.py --reference examples/_reference_sizing.py
python examples/benchmark_scaling.py --backend pyvista --dim 2 --counts 100000 1000000 --marker-spacing neighbors --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/output/scaling-after"
python examples/benchmark_scaling.py --backend pyvista --dim 2 --counts 100000 1000000 --reference-sizing examples/_reference_sizing.py --output "$WARPSPH_DEVELOPMENT_ROOT/plotting/output/scaling-before"
```

Repeat with `--backend matplotlib` (2D) or `--backend vispy` and `--dim 3` as
appropriate. Headless Vispy may need `EGL_PLATFORM=surfaceless`.

## Verification and remaining work

The final suite passes **117 tests on the NVIDIA host, with no skips**.
The CPU/Mesa sandbox passes **116 tests with one explicit skip** for the new
GLFW notebook test because the host display is inaccessible there. Both runs
report 1,823 dependency deprecation warnings.
CPU log (local evidence: `notebook/pytest_cpu.txt`), host log (local evidence: `notebook/pytest_host.txt`).
New regressions cover finite/strided sizing inputs, 2D pan versus 3D rotation,
unsupported notebook rendering modes, and actual cross-thread notebook
frame/export/resize behavior in an isolated process.

The wheel is rebuilt and its sources, README and MIT license checked against
the checkout; package verification (local evidence: `license_verification.json`) and
source hashes (local evidence: `source_manifest.json`) identify the delivered files. No remote
CI, release, CUDA Graph capture or new async export stress run is claimed.

Open work remains client-rendered overlays, broader browser/manual usability
feedback, further CPU sorting/transfer costs, and the previously documented
3D feature gaps. Proper integrated 3D simulation examples remain deferred
until the frontend cases provide those inputs. This pass does not alter or
extend the physical validity claims of the earlier frontend simulations.

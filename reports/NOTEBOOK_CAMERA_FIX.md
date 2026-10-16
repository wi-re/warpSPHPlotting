# Camera interaction followed by Sod frame changes

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Fixed and verified locally on 2026-10-16, following the
[notebook interaction pass](NOTEBOOK_INTERACTIVITY.md).

The reported sequence was moving the PyVista camera, selecting Sod, and changing
frame. The image became corrupted and the kernel could die. An isolated
reproduction with the real saved frontend states and trame's VTK interaction/
capture API produced matching VAO/buffer errors and **exited with a segmentation
fault (139)**. Native trace (local evidence: `notebook-camera/before_protocol.txt`).

Reported corrupted notebook view (local evidence: `notebook-camera/reported_crash.png`)

## Cause and fix

Notebook execution, widget callbacks and trame rendering can run on different
kernel threads. Our previous cleanup covered explicit backend show/export calls,
but trame's remote interaction and image-capture calls could retain the graphics
context. A subsequent frame update on another thread then used invalid OpenGL
state. Releasing only at VTK's render EndEvent would also be insufficient:
image readback can make the context current again after rendering finishes.

Managed PyVista notebook windows now serialize scene mutations, direct
`Plotter.render()` calls (including toolbar callbacks), and trame interaction/
capture calls with the same lock. They explicitly acquire their context before
graphics work and release it after the complete outer render/readback operation.
Nested image callbacks retain the context until the enclosing export finishes.
The trame application hooks apply only to registered windows and are removed
when those windows close; unrelated windows retain their original behavior.

The mixed-backend check also reproduced an independent Vispy asynchronous
`KeyError: timestamp`. Its synchronous frame push had been replacing real
browser feedback with a synthetic acknowledgment containing only an index.
Direct frame pushes now leave browser acknowledgments intact. The regression
actually runs the pending RFB callback and checks for asynchronous errors.

## Verification

`examples/verify_notebook_camera.py` starts an isolated real trame notebook view,
subscribes to its image-delivery protocol, and alternates camera pan/zoom and
frame updates across threads. It uses the recorded frontend states, including
Sod frame 11. The final run completed **33 case/frame checks** across three
cycles with both PyVista and the Vispy notebook widget open and exporting.
It verified camera retention for unchanged domains and saved a trame JPEG and
PNG export for each frame. JPEG encoding is asynchronous, so comparisons wait
for the current image instead of accidentally accepting an older frame.

All final captures contain valid colored output, with mean absolute JPEG/PNG
RGB differences below 5 levels on the 0–255 scale. There were no VTK buffer
errors, kernel/process deaths, or asynchronous callback errors in the final
mixed replay. Results (local evidence: `notebook-camera/mixed/verification.json`),
execution log (local evidence: `notebook-camera/mixed_protocol.txt`).

Sod frame 11 after camera interaction and frame update (local evidence: `notebook-camera/mixed/cycle-0-sod-0011.png`)

Marrone wall-impact frame in the same replay (local evidence: `notebook-camera/mixed/cycle-0-marrone31-0050.png`)

The full suite passes **118 tests on NVIDIA** and **117 tests with one explicit
GLFW/display skip in the CPU sandbox**, with 1,823 dependency warnings in each
run. Host log (local evidence: `notebook-camera/pytest_host.txt`),
CPU log (local evidence: `notebook-camera/pytest_cpu.txt`).
The final expanded pan/zoom and callback regressions also pass separately:
host (local evidence: `notebook-camera/final_interop_host.txt`),
CPU (local evidence: `notebook-camera/final_interop_cpu.txt`).

The regression checks image content and JPEG/PNG agreement after repeated camera,
domain and particle-count changes, direct plotter renders, release on a failed
capture, and cleanup without changing unregistered windows. The automated replay
exercises real trame view/protocol and graphics code; it does not claim a new
manual browser or network-latency check. Browser control of the available tab
was blocked, so the user's live session was not restarted or retested through UI.

Reproduce with the notebook dependencies and recordings present:

```bash
python examples/verify_notebook_camera.py --with-vispy --cycles 3
pytest -q tests/test_renderer_interop.py
```

**Restart the notebook kernel and run all cells to load this fix.** Refreshing
the webpage alone leaves the old Python classes and callbacks in its kernel.
The loopback Jupyter server remains running independently of the chat turn.

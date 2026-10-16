# Vispy notebook streamline blackout

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Fixed and verified locally on 2026-10-16. The reported Marrone notebook failure
was `RuntimeError: EventEmitter loop detected!` in
`_flush_notebook_frame → get_frame → canvas.events.draw`.

## Reproduction and fix

A nested widget frame request during a draw reproduced the exact traceback
using the saved Marrone frame 50. Before-fix trace (local evidence: `vispy-streamlines/before.txt`),
after-fix check (local evidence: `vispy-streamlines/after.txt`).

The framebuffer path allowed a second request to enter Vispy's active draw
emitter. Its unconditional graphics-context cleanup could also unbind the
context while the outer frame was still drawing. Notebook draws and scene
updates on separate kernel threads could overlap as well.

Each Vispy canvas now shares a lock across notebook capture, scene updates,
resize/input events and exports. A nested frame request returns `None` and
schedules a fresh draw after the current operation finishes. Nested calls keep
the graphics context until the outer operation ends; subsequent calls from
other threads acquire it explicitly. Queued frame callbacks after close return
`None`. Unexpected synchronous rendering failures now propagate rather than
being silently swallowed by the notebook flush.

## Rendered verification

[`examples/verify_vispy_notebook.py`](../examples/verify_vispy_notebook.py) replays
the real Marrone 3.1 recording at frames 0, 25 and 50: 4,400 particles including
boundaries, with the notebook's 96-cell grid resolution. Two cycles switch
particles → grid → streamlines → particles, completing **24 checks**. Capture
runs on another thread during title/frame updates, and scheduled RFB callbacks
run on a real asyncio loop with simulated complete client acknowledgments.

Every widget frame has colored content. Widget/PNG comparisons stay below one
mean RGB level on the 0–255 scale, including the initial layout settling.
Hiding the streamline visual removes 1,170 dark pixels at frame 25 and 1,984
at frame 50, confirming visible trajectories over the colored field. Frame 0
has zero velocity and correctly produces no streamline pixels. There are no
asynchronous callback errors or rendering failures.
Structured results (local evidence: `vispy-streamlines/replay/verification.json`),
execution log (local evidence: `vispy-streamlines/replay_host.txt`).

Marrone frame 50, actual Vispy widget framebuffer (local evidence: `vispy-streamlines/replay/cycle-1-frame-0050-step-2-streamlines-widget.png`)

The new regression also holds an active draw open while another thread tries
to update the scene, verifies that the update waits, requests nested frames,
checks visible streamlines, runs queued widget callbacks and checks close
cleanup. It passes on the host display/GPU. The complete suite passes
**119 tests on NVIDIA**, and **117 with two explicit GLFW/display skips in the
CPU sandbox**. Both runs emit 1,823 existing dependency warnings.
Host log (local evidence: `vispy-streamlines/pytest_host.txt`),
CPU log (local evidence: `vispy-streamlines/pytest_cpu.txt`).

The replay exercises real Vispy/jupyter_rfb widget and graphics code; it does
not claim a new manual browser check or measure browser latency. Browser
control of the available tab remains blocked, so the user's active notebook
kernel has not been restarted through the UI.

Reproduce with the notebook dependencies, recordings and an accessible display:

```bash
python examples/verify_vispy_notebook.py --cycles 2
pytest -q tests/test_renderer_interop.py
```

**Restart the notebook kernel and run all cells to load the fix.** A webpage
refresh alone retains the previous Python classes and callback wrappers.

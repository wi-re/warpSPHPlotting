# Stable physical marker sizes during Marrone playback

Generated media and raw measurements are stored outside Git. See [evidence storage and regeneration](README.md); artifact filenames below are relative to the external `plotting/reports/` tree.

Implemented and verified on 2026-10-16 following the user's observed shrinking
fluid markers and preference for physical sizing by default.

## Cause and behavior

The old automatic rule capped a population's nominal particle spacing by the
tenth percentile of sampled nearest-neighbor distances. In the recorded Marrone
flow, that fluid estimate falls from 0.030108 to 0.008239, even though the nominal
volume spacing stays near 0.0302. Close pairs developing in the moving fluid
therefore shrank **every fluid marker by about 73%**. Wall positions and spacing
remain fixed, explaining their steady appearance. Returning to frame 0 restores
its larger neighbor distances. This is a state-dependent heuristic effect,
rather than a renderer accumulating repeated scale factors.

`PlottingOptions(markerSpacing="volume")` is now the default on every backend.
It derives spacing from the tenth percentile of physical volumes `mass / density`,
raised to `1 / dimension`; the default diameter remains 0.65 times that spacing.
Fluid and boundary populations retain independent estimates. Real mass/density
changes still affect size, and camera zoom, resize and export DPI still determine
its projected diameter. Invalid/absent usable volumes fall back to geometry.

Use `markerSpacing="neighbors"` to retain the old close-pair cap when reducing
overlap takes priority. Physical sizing can show overlapping particles where
positions crowd; it does not conceal that crowding by shrinking the population.
Explicit `markerSize` and backend pixel overrides retain their behavior.

The notebook's **Size rule** dropdown offers **Physical volume** (default) and
**Close neighbors**. It is enabled for automatic sizing and recorded in saved
diagnostics. Spacing caches distinguish the two modes within an update, share
work between compatible panels, and start fresh on the next frame.

## Real frontend replay

[`examples/verify_marrone_sizing.py`](../examples/verify_marrone_sizing.py) replays
all 51 saved Marrone frames through Matplotlib, PyVista and Vispy, with both
rules side by side: **153 plot updates / 306 panel-frame comparisons**.
The input has 4,400 particles: 800 fluid, 1,800 boundaries and 1,800 ghosts;
the fluid and boundary populations are rendered.

In every backend, physical fluid and boundary diameters stay within **0.1%**
of their initial values throughout the recording. At the final frame the fluid
diameter changes by just **−0.016%**, reflecting its small density change.
The neighbor-limited fluid diameter falls to **27.36%** of its initial size.
Replaying frame 0 and frame 50 again reproduces both endpoint sizes exactly.
Structured measurements (local evidence: `physical-sizing/replay/verification.json`),
execution log (local evidence: `physical-sizing/replay_host.txt`).

Diameter history over the recorded Marrone frames (local evidence: `physical-sizing/replay/diameter-history.png`)

PyVista: physical volume versus optional neighbor limiting, Marrone frame 50 (local evidence: `physical-sizing/replay/pyvista-frame-0050.png`)

Matching comparisons are available for
Matplotlib (local evidence: `physical-sizing/replay/matplotlib-frame-0050.png`) and
Vispy (local evidence: `physical-sizing/replay/vispy-frame-0050.png`). No simulation was rerun;
these images use actual saved frontend particle states.

## Regression and package checks

The new checks cover stable size during rearrangement, physical density changes,
switching modes in an existing plot, invalid-option rollback, geometric fallback,
empty populations and separate mode caches. Both rules are checked for shared
multi-panel work and invalidation on the next update. Existing image tests cover
zoom, resize, explicit sizes and export DPI.

The complete suite passes **130 tests on NVIDIA**, and **128 with two explicit
GLFW/display skips in the CPU sandbox**, with 2,276 dependency warnings in both
runs. Host log (local evidence: `physical-sizing/pytest_host.txt`),
CPU log (local evidence: `physical-sizing/pytest_cpu.txt`). The wheel matches package source,
README and MIT license. These are renderer/replay checks; no new manual browser
interaction is claimed.

```bash
python examples/verify_marrone_sizing.py
pytest -q tests/test_particle_sizes.py tests/test_grid_coverage.py
```

**Restart the notebook kernel and run all cells** to load the new default and
the Size rule dropdown.

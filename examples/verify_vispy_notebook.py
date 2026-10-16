"""Replay recorded Marrone frames through the real Vispy notebook framebuffer.

Requires jupyter_rfb, an accessible GLFW display and external frontend recordings.
Exercises nested requests, concurrent capture/update and scheduled widget draws.
This checks renderer/widget code; it does not measure browser input or latency.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from _artifacts import evidence_dir
import time

import numpy as np
from PIL import Image
import torch
import vispy.app
import warp as wp

from verify_notebook_camera import load_frame
from warpSPHPlotting import GridVisualization, Mapping, PlottingOptions, visualize


async def verify(args):
    wp.init()
    torch.set_num_threads(2)
    vispy.app.use_app("jupyter_rfb")
    args.output.mkdir(parents=True, exist_ok=True)
    particles, domain, velocity = load_frame(args.root, "marrone31", 0)
    plot = visualize(
        particles,
        domain,
        velocity,
        PlottingOptions(mapping=Mapping.L2, plotTitle="Speed"),
        backend="vispy",
        backendOptions={"jupyter_backend": "notebook"},
        figsize=(9, 5),
        figTitle="Marrone 3.1 · notebook replay",
    )
    widget = plot.fig._backend
    loop = asyncio.get_running_loop()
    errors, records, nested = [], [], []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, ctx: errors.append(str(ctx)))

    def nested_request(event):
        if not nested:
            nested.append(widget.get_frame())
            plot.show()

    try:
        # A draw callback requesting another frame reproduced the reported
        # EventEmitter traceback before the fix. The outer frame must finish.
        plot.fig.events.draw.connect(nested_request, position="first")
        widget.get_frame()
        plot.fig.events.draw.disconnect(nested_request)
        assert len(nested) == 1 and nested[0] is None
        with ThreadPoolExecutor(max_workers=1) as remote:
            for cycle in range(args.cycles):
                for frame in (0, 25, 50):
                    particles, domain, velocity = load_frame(
                        args.root, "marrone31", frame
                    )
                    for step, mode in enumerate(
                        ("particles", "grid", "streamlines", "particles")
                    ):
                        grid = (
                            None
                            if mode == "particles"
                            else GridVisualization(
                                resolution=96, streamLines=mode == "streamlines"
                            )
                        )
                        # Capture on another kernel thread while the calling
                        # thread updates title, scene data and display mode.
                        capture = loop.run_in_executor(remote, widget.get_frame)
                        started = time.perf_counter()
                        plot.updateTitle(f"Marrone 3.1 · frame {frame} · {mode}")
                        plot.updateQuantities(
                            velocity,
                            newParticleState=particles,
                            newDomain=domain,
                            newOptions={"A": {"gridVisualization": grid}},
                            redraw=True,
                            yieldSeconds=0,
                        )
                        update_ms = (time.perf_counter() - started) * 1000
                        assert await asyncio.wait_for(capture, timeout=15) is not None

                        # Simulate complete client feedback to exercise the
                        # real asynchronous widget callback and its statistics.
                        widget._frame_feedback = {
                            "index": widget._rfb_frame_index,
                            "timestamp": time.time(),
                            "localtime": time.perf_counter(),
                        }
                        widget._has_visible_views = True
                        before = widget._rfb_frame_index
                        widget.request_draw()
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)
                        assert widget._rfb_frame_index > before
                        rgb = widget.get_frame()[:, :, :3]
                        colored = int(np.count_nonzero(np.ptp(rgb, axis=2) > 40))
                        assert colored > 500, (frame, mode, colored)
                        streamline_pixels = None
                        if mode == "streamlines":
                            backend = plot._backend_instance
                            line = plot.plotStates["A"].stream_visual
                            assert line is not None and line.visible
                            with backend._canvas_access.use(plot.fig):
                                line.visible = False
                            without = widget.get_frame()[:, :, :3]
                            with backend._canvas_access.use(plot.fig):
                                line.visible = True
                            streamline_pixels = int(
                                np.count_nonzero(
                                    (rgb.mean(2) < 60) & (without.mean(2) > 100)
                                )
                            )
                            if torch.linalg.vector_norm(velocity, dim=1).max() > 0:
                                assert streamline_pixels > 10
                            else:
                                # The initial dam is at rest: no trajectories
                                # should be invented for a zero velocity field.
                                assert streamline_pixels == 0
                        stem = f"cycle-{cycle}-frame-{frame:04d}-step-{step}-{mode}"
                        Image.fromarray(rgb).save(args.output / f"{stem}-widget.png")
                        plot.export(str(args.output / f"{stem}.png"))
                        with Image.open(args.output / f"{stem}.png") as exported:
                            export_rgb = np.array(exported.convert("RGB"))
                        export_error = float(
                            np.abs(rgb.astype(float) - export_rgb).mean()
                        )
                        assert export_error < 1, (frame, mode, export_error)
                        records.append(
                            dict(
                                cycle=cycle,
                                frame=frame,
                                mode=mode,
                                image=f"{stem}-widget.png",
                                particles=len(particles.positions),
                                update_ms=update_ms,
                                colored_pixels=colored,
                                streamline_pixels=streamline_pixels,
                                widget_export_mean_error=export_error,
                            )
                        )
                        print(stem, f"{update_ms:.1f} ms", flush=True)
        await asyncio.sleep(0)
        assert not errors, errors
    finally:
        plot.close()
        loop.set_exception_handler(previous_handler)
    assert widget.get_frame() is None
    result = dict(
        nested_request_deferred=True,
        async_errors=errors,
        closed_callback_returns_none=True,
        checks=records,
        scope="Real widget/render code; no browser input or latency measurement",
    )
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output", type=Path, default=evidence_dir("vispy-streamlines/replay")
    )
    parser.add_argument("--cycles", type=int, default=2)
    asyncio.run(verify(parser.parse_args()))

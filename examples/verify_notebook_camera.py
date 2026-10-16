"""Exercise camera -> case/frame changes through real trame VTK capture calls.

Requires the saved frontend recordings in the external development archive and PyVista's
notebook dependencies. Starts an isolated loopback trame server, renders every
particle, saves JPEG/PNG comparisons, and closes its own server afterward.
This is a protocol/rendering check, not a browser input or latency measurement.
"""

import argparse
import asyncio
import ctypes
import ctypes.util
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import io
import json
from pathlib import Path
from _artifacts import evidence_dir, recording_paths
import time
import uuid

import numpy as np
from PIL import Image
import torch
from warpSPHCore import DomainDescription, ParticleState
from warpSPHPlotting import Mapping, PlottingOptions, visualize


def load_frame(root, case, frame):
    paths = recording_paths(case, root=root)
    if not paths:
        raise FileNotFoundError(
            f"Missing {case} frontend recording; run examples/regenerate_frontends.py --cases {case}"
        )
    with np.load(paths[frame]) as data:
        particles = ParticleState(
            **{
                name: torch.from_numpy(data[name].copy())
                for name in ("positions", "supports", "masses", "densities", "kinds")
            }
        )
        domain = DomainDescription(
            torch.from_numpy(data["domain_min"].copy()),
            torch.from_numpy(data["domain_max"].copy()),
            torch.from_numpy(data["periodic"].copy()),
            2,
        )
        velocity = torch.from_numpy(data["velocities"].copy())
    if case == "sod":
        center, half = (domain.min + domain.max) / 2, (domain.max - domain.min) * 3.75
        domain = DomainDescription(center - half, center + half, domain.periodic, 2)
    return particles, domain, velocity


def camera(plot):
    return [list(x) for x in plot.fig.camera_position] + [
        plot.fig.camera.parallel_scale
    ]


def context_current(window):
    if window.IsA("vtkEGLRenderWindow"):
        # VTK's EGL IsCurrent flag can remain true after native unbinding.
        egl = ctypes.CDLL(ctypes.util.find_library("EGL"))
        egl.eglGetCurrentContext.restype = ctypes.c_void_p
        return bool(egl.eglGetCurrentContext())
    return bool(window.IsCurrent())


async def verify(args):
    import pyvista as pv
    from trame_pyvista.jupyter import launch_server
    from vtkmodules.vtkWebCore import vtkWebInteractionEvent

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    pv.global_theme.trame.jupyter_server_name = f"camera-check-{uuid.uuid4().hex}"
    pv.global_theme.trame.server_proxy_enabled = False
    # Force the real notebook view even when this verifier runs as a script.
    pv.global_theme.notebook = True
    server = launch_server(host="127.0.0.1")
    await server.ready
    plot = None
    vispy_plot = None
    view_id = None
    records = []
    loop = asyncio.get_running_loop()
    callback_errors = []
    previous_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda loop, context: callback_errors.append(str(context))
    )
    try:
        particles, domain, velocity = load_frame(args.root, "marrone31", 0)
        plot = visualize(
            particles,
            domain,
            velocity,
            PlottingOptions(mapping=Mapping.L2),
            backend="pyvista",
            backendOptions={"jupyter_backend": "trame", "off_screen": True},
            figsize=(8, 5),
        )
        backend = plot._backend_instance
        guard = backend._trame_context_guard
        assert guard is not None
        application, window = guard.application, plot.fig.render_window
        view_id = str(application.GetObjectIdMap().GetGlobalId(window))
        window.GetInteractor().EnableRenderOff()
        subscription = server.protocol_call("viewport.image.push.observer.add", view_id)
        assert subscription.get("success"), subscription
        if args.with_vispy:
            import vispy.app

            vispy.app.use_app("jupyter_rfb")
            vispy_plot = visualize(
                particles,
                domain,
                velocity,
                PlottingOptions(mapping=Mapping.L2),
                backend="vispy",
                backendOptions={"jupyter_backend": "notebook"},
                figsize=(8, 5),
            )

        def capture(move):
            if move:
                for buttons, x, y in [(1, 0.45, 0.5), (1, 0.49, 0.54), (0, 0.49, 0.54)]:
                    event = vtkWebInteractionEvent()
                    event.SetButtons(buttons)
                    event.SetX(x)
                    event.SetY(y)
                    application.HandleInteractionEvent(window, event)
                event = vtkWebInteractionEvent()
                event.SetScroll(0.1)
                application.HandleInteractionEvent(window, event)
            application.InvalidateCache(window)
            encoded = application.StillRenderToBuffer(window, 0, 95)
            deadline = time.monotonic() + 5
            while application.GetHasImagesBeingProcessed(window):
                if time.monotonic() > deadline:
                    raise TimeoutError("trame image encoder did not finish")
                time.sleep(0.01)
                encoded = application.StillRenderToBuffer(
                    window, application.GetLastStillRenderToMTime(), 95
                )
            data = memoryview(encoded).tobytes()
            with Image.open(io.BytesIO(data)) as image:
                rgb = np.array(image.convert("RGB"))
            assert np.count_nonzero(np.ptp(rgb, axis=2) > 40) > 500
            assert not context_current(window)
            return data, rgb

        cases = {
            "sod": [0, 11, 40, 3, 30],
            "marrone31": [0, 50],
            "khh": [0, 40],
            "tgv": [0, 80],
        }
        with ThreadPoolExecutor(max_workers=1) as remote:
            loop = asyncio.get_running_loop()
            for cycle in range(args.cycles):
                for case, frames in cases.items():
                    for frame in frames:
                        await loop.run_in_executor(remote, capture, True)
                        moved = camera(plot)
                        particles, next_domain, velocity = load_frame(
                            args.root, case, frame
                        )
                        same_domain = torch.equal(
                            domain.min, next_domain.min
                        ) and torch.equal(domain.max, next_domain.max)
                        plot.updateQuantities(
                            velocity,
                            newParticleState=particles,
                            newDomain=next_domain,
                            redraw=True,
                            yieldSeconds=0,
                        )
                        if same_domain:
                            assert camera(plot) == moved, (
                                case,
                                frame,
                                "camera changed",
                            )
                        domain = next_domain
                        data, remote_rgb = await loop.run_in_executor(
                            remote, capture, False
                        )
                        stem = f"cycle-{cycle}-{case}-{frame:04d}"
                        (args.output / f"{stem}.jpg").write_bytes(data)
                        plot.export(str(args.output / f"{stem}.png"))
                        with Image.open(args.output / f"{stem}.png") as image:
                            exported_rgb = np.array(image.convert("RGB"))
                        error = float(
                            np.abs(remote_rgb.astype(float) - exported_rgb).mean()
                        )
                        assert error < 5, (case, frame, "remote/export mismatch", error)
                        if vispy_plot is not None:
                            vispy_plot.updateQuantities(
                                velocity,
                                newParticleState=particles,
                                newDomain=domain,
                                redraw=True,
                                yieldSeconds=0,
                            )
                            vispy_plot.export(str(args.output / f"{stem}-vispy.png"))
                        records.append(
                            {
                                "cycle": cycle,
                                "case": case,
                                "frame": frame,
                                "particles": len(particles.positions),
                                "camera_retained": same_domain,
                                "camera": camera(plot),
                                "remote_export_mean_absolute_rgb_error": error,
                                "window": window.GetClassName(),
                                "png": f"{stem}.png",
                                "jpeg": f"{stem}.jpg",
                                "vispy_also_open": args.with_vispy,
                            }
                        )
                        print(
                            f"Verified {stem}, JPEG/PNG error {error:.3f}", flush=True
                        )
    finally:
        if view_id is not None:
            server.protocol_call("viewport.image.push.observer.remove", view_id)
        if vispy_plot is not None:
            vispy_plot.close()
        if plot is not None:
            plot.close()
        await server.stop()
        loop.set_exception_handler(previous_exception_handler)
    assert not callback_errors, callback_errors
    (args.output / "verification.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=evidence_dir("notebook-camera/live")
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--with-vispy", action="store_true")
    arguments = parser.parse_args()
    if arguments.cycles < 1:
        parser.error("--cycles must be positive")
    faulthandler.enable()
    asyncio.run(verify(arguments))

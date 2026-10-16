"""Check trame notebook startup, loopback HTTP and backend render updates on a local renderer.

This is a transport/startup check; it does not automate browser interaction.
Requires the plot-pyvista extra. Native windows and CUDA are not required.
"""

import argparse
import asyncio
import gc
import json
from pathlib import Path
import numpy as np
import torch
from warpSPHPlotting import PlottingOptions, GridVisualization, VisualizeOptions
from warpSPHPlotting.backends.pyvista_backend import PyVistaBackend
from stress_replay import load


async def main(args):
    import aiohttp
    from trame_pyvista.jupyter import launch_server, build_url
    from trame_pyvista.ui import get_viewer
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/warpsph-warp-cache"
    wp.init()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    p, _, d = load(args.input)
    server = launch_server(port=0, host="127.0.0.1")
    await asyncio.wait_for(server.ready, timeout=20)
    backend = PyVistaBackend()
    try:
        fig = backend.create_figure(
            "A",
            (7, 5),
            False,
            False,
            "Trame notebook verification",
            {"jupyter_backend": "trame", "off_screen": True},
        )
        # Explicit notebook display configuration in this headless process.
        fig.notebook = True
        opts = PlottingOptions(
            vMin=0,
            vMax=2,
            gridVisualization=GridVisualization(resolution=64),
            boundaryVisualization=VisualizeOptions.Passive,
        )
        state = backend.render_panel("A", p, d, p.densities, opts)
        backend.show()
        viewer = get_viewer(fig, server=server)
        url = build_url(
            server, ui=fig._id_name, server_proxy_enabled=False, host="127.0.0.1"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                status = response.status
                html = await response.text()
        assert status == 200 and "trame" in html.lower()
        backend.export(str(args.output.with_name("trame_before.png")))
        before = np.array(fig.screenshot(return_img=True)).astype(float)
        backend.update_panel("A", state, p, d, p.densities * 1.5, opts)
        backend.show()
        viewer.update_image()
        backend.export(str(args.output.with_name("trame_after.png")))
        after = np.array(fig.screenshot(return_img=True)).astype(float)
        difference = float(np.mean(np.abs(after - before)))
        assert difference > 1
        result = dict(
            http_status=status,
            ui_initialized=True,
            render_callbacks=len(fig._on_render_callbacks),
            mean_pixel_change=difference,
            window=fig.render_window.GetClassName(),
            browser_interaction_verified=False,
        )
        assert result["render_callbacks"] > 0
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        backend.close()
        await server.stop()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))

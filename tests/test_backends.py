import copy
import numpy as np
import pytest
import torch
from warpSPHPlotting import *
from warpSPHPlotting.backends._mosaic import parse_mosaic


@pytest.mark.parametrize("mosaic", ["AA;BC", "A.;BC", "AB\nCD"])
def test_mosaic(mosaic):
    shape, axes, spans = parse_mosaic(mosaic)
    assert shape == (2, 2) and len(axes) >= 3


@pytest.mark.parametrize("mosaic", ["", "AB\nC", "AA\nAB", ".."])
def test_bad_mosaic(mosaic):
    with pytest.raises(ValueError):
        parse_mosaic(mosaic)


def test_pyvista_colors_lifecycle(particles, domain, tmp_path):
    pv = pytest.importorskip("pyvista")
    s = visualize(
        particles,
        domain,
        torch.logspace(-3, 3, 16),
        PlottingOptions(quantityScaling=PlotScaling.Logarithmic),
        backend="pyvista",
    )
    panel = s.plotStates["A"]
    rgb = [0.0, 0.0, 0.0]
    panel.lookup.GetColor(float(panel.norm(1.0)), rgb)
    from matplotlib import colormaps

    np.testing.assert_allclose(
        rgb, colormaps["viridis"](float(panel.norm(1.0)))[:3], atol=0.01
    )
    assert np.allclose(panel.mesh_fluid["quantity"], torch.logspace(-3, 3, 16).numpy())
    actor = panel.actor_fluid
    count = len(s.fig.renderer.actors)
    for _ in range(3):
        s.updateQuantities(torch.logspace(-3, 3, 16), redraw=False)
    assert panel.actor_fluid is actor and len(s.fig.renderer.actors) == count
    s.updateQuantities(
        torch.ones(16),
        newOptions={"A": {"fluidVisualization": VisualizeOptions.Hide}},
        redraw=False,
    )
    assert "fluid_A" not in panel.actors and panel.colorbar is None
    s.updateQuantities(
        torch.ones(16),
        newOptions={"A": {"fluidVisualization": VisualizeOptions.Visualize}},
        redraw=False,
    )
    assert panel.actor_fluid is not None and panel.colorbar is not None
    smaller = copy.copy(particles)
    for name in ("positions", "supports", "masses", "densities", "kinds"):
        setattr(smaller, name, getattr(smaller, name)[:5])
    s.updateQuantities(torch.ones(5), newParticleState=smaller, redraw=False)
    assert panel.mesh_fluid.n_points == panel.mesh_fluid.n_verts == 5
    path = tmp_path / "render.png"
    s.export(str(path))
    assert path.stat().st_size > 1000
    s.close()


def test_pyvista_grids_streamlines(particles, domain, tmp_path):
    pytest.importorskip("pyvista")
    v = torch.ones(16, 2)
    s = visualize(
        particles, domain, v, PlottingOptions(mapping=Mapping.L2), backend="pyvista"
    )
    for n, lines in [(4, True), (8, False)]:
        s.updateQuantities(
            v,
            newOptions={
                "A": {
                    "gridVisualization": GridVisualization(
                        resolution=n, streamLines=lines
                    )
                }
            },
            redraw=False,
        )
        panel = s.plotStates["A"]
        assert panel.mesh_grid.dimensions == (n, n, 1)
        assert ("stream_A" in panel.actors) == lines
    d = copy.copy(domain)
    d.max = domain.max * 2
    s.updateQuantities(v, newDomain=d, redraw=False)
    assert panel.mesh_grid.spacing[0] == pytest.approx(0.25)
    s.export(str(tmp_path / "grid.png"))
    s.close()


def test_pyvista_grid_scatter_grid_reuses_geometry_safely(particles, domain, tmp_path):
    pytest.importorskip("pyvista")
    q = torch.arange(16.0)
    grid = GridVisualization(resolution=8)
    s = visualize(
        particles, domain, q, PlottingOptions(gridVisualization=grid), backend="pyvista"
    )
    try:
        counts = []
        for _ in range(3):
            s.updateQuantities(
                q, newOptions={"A": {"gridVisualization": None}}, redraw=False
            )
            assert s.plotStates["A"].mesh_grid is None
            s.updateQuantities(
                q, newOptions={"A": {"gridVisualization": grid}}, redraw=False
            )
            assert s.plotStates["A"].mesh_grid.dimensions == (8, 8, 1)
            s.export(str(tmp_path / "grid.png"))
            counts.append(len(s.fig.renderer.actors))
        assert len(set(counts)) == 1
    finally:
        s.close()


def test_vispy_real_render_lifecycle(particles, domain, tmp_path):
    pytest.importorskip("vispy")
    s = visualize(
        particles,
        domain,
        torch.ones(16),
        PlottingOptions(),
        backend="vispy",
        backendOptions={"jupyter_backend": "image", "app_backend": "egl"},
    )
    panel = s.plotStates["A"]
    for variant in (
        VisualizeOptions.Hide,
        VisualizeOptions.Passive,
        VisualizeOptions.Visualize,
    ):
        s.updateQuantities(
            torch.ones(16),
            newOptions={"A": {"fluidVisualization": variant}},
            redraw=False,
        )
    assert panel.markers_fluid.visible and panel.colorbar is not None
    s.updateQuantities(
        torch.ones(16),
        newOptions={
            "A": {"showColorBar": False, "plotDomain": False, "plotTitle": "New title"}
        },
        redraw=False,
    )
    assert (
        panel.colorbar is None
        and not panel.domain_lines.visible
        and panel.title_visual.text == "New title"
    )
    s.export(str(tmp_path / "vispy.png"))
    assert (tmp_path / "vispy.png").stat().st_size > 1000
    s.export(str(tmp_path / "vispy.jpg"))
    s.close()


def test_vispy_colorbar_toggle_reuses_font_resources(particles, domain, tmp_path):
    pytest.importorskip("vispy")
    s = visualize(
        particles,
        domain,
        torch.arange(16.0),
        PlottingOptions(),
        backend="vispy",
        backendOptions={"jupyter_backend": "image", "app_backend": "egl"},
    )
    try:
        bar = s.plotStates["A"].colorbar
        for i in range(5):
            s.updateQuantities(
                torch.arange(16.0),
                newOptions={"A": {"showColorBar": False}},
                redraw=False,
            )
            s.fig.render()
            assert s.plotStates["A"].colorbar is None
            s.updateQuantities(
                torch.linspace(2 + i, 4 + i, 16),
                newOptions={"A": {"showColorBar": True}},
                redraw=False,
            )
            s.export(str(tmp_path / "shown.png"))
            assert s.plotStates["A"].colorbar is bar
            assert bar.clim == (str(2 + i), str(4 + i))
            assert all(t.rotation == 0 for t in bar._colorbar._ticks)
    finally:
        s.close()


def test_vispy_limit_updates_preserve_colorbar_gpu_resources(
    particles, domain, tmp_path
):
    pytest.importorskip("vispy")
    s = visualize(
        particles,
        domain,
        torch.arange(16.0),
        PlottingOptions(),
        backend="vispy",
        backendOptions={"jupyter_backend": "image", "app_backend": "egl"},
    )
    try:
        s.updateQuantities(torch.linspace(2, 4, 16), redraw=False)
        bar = s.plotStates["A"].colorbar
        cmap, texture = bar.cmap, bar.cmap.texture_lut()
        for i in range(5):
            s.updateQuantities(torch.linspace(2 + i, 4 + i, 16), redraw=False)
            s.export(str(tmp_path / "range.png"))
            assert bar.cmap is cmap and bar.cmap.texture_lut() is texture
            assert bar.clim == (str(2 + i), str(4 + i))
        s.updateQuantities(
            torch.arange(16.0),
            newOptions={"A": {"colorMap": ColorMap.RdBu}},
            redraw=False,
        )
        s.export(str(tmp_path / "new_cmap.png"))
        assert bar.cmap is not cmap and bar.cmap.texture_lut() is not texture
        pixels = s.fig.render()
        strip = bar._colorbar
        x, y = bar.node_transform(s.fig.scene).map(
            (strip.pos[0], strip.pos[1] - strip._halfdim[1] + 3)
        )[:2]
        red, green, blue = pixels[round(y), round(x), :3].astype(int)
        assert blue > red + 60  # RdBu's upper end is blue, unlike viridis.
    finally:
        s.close()


def test_vtk_export_roundtrip(particles, domain, tmp_path):
    pv = pytest.importorskip("pyvista")
    path = export_particles_vtp(
        tmp_path / "particles.vtp",
        particles,
        {"q": torch.arange(16.0), "v": torch.ones(16, 2)},
        domain,
        {"time": 0.2},
    )
    mesh = pv.read(path)
    assert mesh.n_points == 16 and mesh.n_verts == 16
    np.testing.assert_allclose(mesh["q"], np.arange(16))
    assert "metadata" in mesh.field_data
    pts, dx, n, vol, ext = generateGrid(domain, 4)
    from types import SimpleNamespace

    values = pts[:, 0] + 2 * pts[:, 1]
    path = export_grid_vti(
        tmp_path / "grid.vti",
        SimpleNamespace(positions=pts),
        {"q": values, "v": pts},
        n,
        ext,
    )
    mesh = pv.read(path)
    np.testing.assert_allclose(mesh["q"], mesh.points[:, 0] + 2 * mesh.points[:, 1])
    np.testing.assert_allclose(mesh["v"], mesh.points[:, :2])
    pvd = write_time_series_pvd(tmp_path / "series.pvd", [(0, path), (0.2, path)])
    assert pv.get_reader(pvd).time_values == [0, 0.2]


def test_vispy_mosaic_axes_match_visible_domain(particles, domain, tmp_path):
    pytest.importorskip("vispy")
    s = visualize(
        particles,
        domain,
        torch.ones(16),
        PlottingOptions(),
        mosaic="AB",
        backend="vispy",
        backendOptions={"jupyter_backend": "image", "app_backend": "egl"},
    )
    s.export(str(tmp_path / "axes.png"))
    b = s._backend_instance
    for key, axes in b._axisWidgets.items():
        rect = b._views[key].camera._real_rect
        np.testing.assert_allclose(
            axes[0].axis.domain, (rect.left, rect.right), atol=1e-5
        )
        np.testing.assert_allclose(
            axes[1].axis.domain, (rect.bottom, rect.top), atol=1e-5
        )
    s.close()


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista", "vispy"])
def test_3d_slice_render_and_update(backend, tmp_path):
    if backend != "matplotlib":
        pytest.importorskip(backend)
    from warpSPHCore import ParticleState, DomainDescription

    x = torch.linspace(0.125, 0.875, 4)
    pos = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    p = ParticleState(
        pos,
        torch.full((64,), 0.8),
        torch.full((64,), 1 / 64),
        torch.zeros(64, dtype=torch.int32),
        torch.ones(64),
    )
    d = DomainDescription(
        torch.zeros(3), torch.ones(3), torch.zeros(3, dtype=torch.bool), 3
    )
    beopts = (
        {"jupyter_backend": "image", "app_backend": "egl"} if backend == "vispy" else {}
    )
    s = visualize(
        p,
        d,
        pos[:, 2],
        PlottingOptions(
            gridVisualization=GridVisualization(
                resolution=4, sliceAxis=2, slicePosition=0.25
            )
        ),
        backend=backend,
        backendOptions=beopts,
    )
    assert s.plotStates["A"].domain.dim == 2
    s.updateQuantities(
        pos[:, 2],
        newOptions={
            "A": {
                "gridVisualization": GridVisualization(
                    resolution=6, sliceAxis=2, slicePosition=0.75
                )
            }
        },
        redraw=False,
    )
    s.export(str(tmp_path / f"slice-{backend}.png"))
    s.close()


def test_pyvista_export_contains_latest_frame(particles, domain, tmp_path):
    pytest.importorskip("pyvista")
    from PIL import Image

    s = visualize(
        particles,
        domain,
        torch.arange(16.0),
        PlottingOptions(markerSize=40),
        backend="pyvista",
    )
    s.export(str(tmp_path / "before.png"))
    s.updateQuantities(torch.arange(16.0).flip(0), redraw=False)
    s.updateTitle("Updated frame")
    renderers = s.fig.render_window.GetRenderers()
    renderers.InitTraversal()
    for _ in range(renderers.GetNumberOfItems()):
        assert (
            renderers.GetNextItem().GetLayer() < s.fig.render_window.GetNumberOfLayers()
        )
    s.export(str(tmp_path / "after.png"))
    a = np.asarray(Image.open(tmp_path / "before.png")).astype(float)
    b = np.asarray(Image.open(tmp_path / "after.png")).astype(float)
    assert np.mean(np.abs(a - b)) > 0.1
    s.close()


def test_3d_slice_planar_streamlines(tmp_path):
    from warpSPHCore import ParticleState, DomainDescription

    x = torch.linspace(0.125, 0.875, 4)
    pos = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    p = ParticleState(
        pos,
        torch.full((64,), 0.8),
        torch.full((64,), 1 / 64),
        torch.zeros(64, dtype=torch.int32),
        torch.ones(64),
    )
    d = DomainDescription(
        torch.zeros(3), torch.ones(3), torch.zeros(3, dtype=torch.bool), 3
    )
    s = visualize(
        p,
        d,
        torch.ones(64, 3),
        PlottingOptions(
            mapping=Mapping.L2,
            gridVisualization=GridVisualization(
                resolution=4, sliceAxis=2, streamLines=True
            ),
        ),
    )
    assert s.plotStates["A"].streamLines is not None
    s.close()


def test_pyvista_horizontal_colorbar(particles, domain, tmp_path):
    pytest.importorskip("pyvista")
    s = visualize(
        particles,
        domain,
        torch.arange(16.0),
        PlottingOptions(),
        backend="pyvista",
        backendOptions={
            "scalar_bar": {"orientation": "horizontal", "title": "physical quantity"}
        },
    )
    s.export(str(tmp_path / "horizontal.png"))
    assert s.plotStates["A"].colorbar.GetOrientation() == 0
    assert all(
        label.GetTextProperty().GetOrientation() == 0
        for label in s.plotStates["A"].labels
    )
    s.close()


@pytest.mark.parametrize("grid", [False, True])
def test_pyvista_frame_and_upright_colorbar(particles, domain, tmp_path, grid):
    pytest.importorskip("pyvista")
    s = visualize(
        particles,
        domain,
        torch.linspace(0.997, 1.003, 16),
        PlottingOptions(
            gridVisualization=GridVisualization(resolution=16) if grid else None,
        ),
        backend="pyvista",
        figsize=(5, 5),
    )
    try:
        panel = s.plotStates["A"]
        bounds = panel.frameBounds
        assert bounds[0] < domain.min[0] < domain.max[0] < bounds[1]
        assert bounds[2] < domain.min[1] < domain.max[1] < bounds[3]
        s.export(str(tmp_path / "frame.png"))
        image = s.fig.screenshot(return_img=True)
        renderer = s.fig.renderer
        renderer.SetWorldPoint(float(bounds[1]), float(bounds[3]), 0.0, 1.0)
        renderer.WorldToDisplay()
        frame_right = renderer.GetDisplayPoint()[0] - renderer.GetOrigin()[0]
        assert panel.colorbar.GetPosition()[0] * renderer.GetSize()[
            0
        ] - frame_right == pytest.approx(12)

        def patch_at(x, y, z=0):
            renderer.SetWorldPoint(float(x), float(y), z, 1)
            renderer.WorldToDisplay()
            px, py, _ = renderer.GetDisplayPoint()
            px, py = round(px), image.shape[0] - 1 - round(py)
            return image[py - 2 : py + 3, px - 2 : px + 3, :3].astype(int)

        # All four frame edges must render, including top/right. The physical
        # domain remains a distinct blue outline inside that gray frame.
        x0, x1, y0, y1 = bounds[:4]
        for x, y in [
            ((x0 + x1) / 2, y0),
            ((x0 + x1) / 2, y1),
            (x0, (y0 + y1) / 2),
            (x1, (y0 + y1) / 2),
        ]:
            pixels = patch_at(x, y)
            gray = (np.ptp(pixels, axis=2) < 8) & (pixels.max(axis=2) < 220)
            assert gray.any(), (x, y)
        for x, y in [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]:
            pixels = patch_at(x, y, 1e-4)
            assert ((pixels[..., 2] - pixels[..., :2].max(axis=2)) > 60).any()
        assert all(t.GetTextProperty().GetOrientation() == 0 for t in panel.labels)

        # A narrower window and longer numbers must keep the labels near the
        # strip and inside the panel, including resize without quantity edits.
        s.fig.window_size = (360, 420)
        s.updateQuantities(torch.linspace(0.000999997, 1.001001, 16), redraw=False)
        s.export(str(tmp_path / "resized.png"))
        s.fig.window_size = (340, 400)
        s.fig.render()  # Also reflow direct/native renders without exporting.
        bar_right = sum([panel.colorbar.GetPosition()[0], panel.colorbar.GetWidth()])
        width = renderer.GetSize()[0]
        for label in panel.labels[:-1]:
            size = [0.0, 0.0]
            label.GetSize(renderer, size)
            x, _ = label.GetPosition()
            assert (x - bar_right) * width == pytest.approx(5)
            assert x * width + size[0] <= width - 4
    finally:
        s.close()


def test_pyvista_streamlines_do_not_add_white_fringes(particles, domain, tmp_path):
    pytest.importorskip("pyvista")
    xy = particles.positions - 0.5
    velocity = torch.stack((-xy[:, 1], xy[:, 0]), dim=1)
    s = visualize(
        particles,
        domain,
        velocity,
        PlottingOptions(
            mapping=Mapping.L2,
            gridVisualization=GridVisualization(resolution=24, streamLines=True),
        ),
        backend="pyvista",
    )
    try:
        samples = s.fig.render_window.GetMultiSamples()
        s.export(str(tmp_path / "streamlines.png"))
        panel = s.plotStates["A"]
        actor = panel.actors["stream_A"]
        assert s.fig.render_window.GetMultiSamples() == 0
        assert not actor.prop.lighting
        with_lines = s.fig.screenshot(return_img=True).astype(int)
        actor.SetVisibility(False)
        s.fig.render()
        without_lines = s.fig.screenshot(return_img=True).astype(int)
        delta = with_lines - without_lines
        # A black overlay can darken its background, never add white highlights.
        assert (delta < -20).any()
        assert delta.max() <= 3
        s.updateQuantities(
            velocity,
            newOptions={
                "A": {
                    "gridVisualization": GridVisualization(
                        resolution=24, streamLines=False
                    )
                }
            },
            redraw=False,
        )
        s.fig.render()
        # The graphics driver may cap the initial requested sample count.
        assert 0 < s.fig.render_window.GetMultiSamples() <= samples
    finally:
        s.close()


@pytest.mark.parametrize("backend", ["matplotlib", "vispy"])
def test_streamline_overlay_is_visible_and_black(backend, particles, domain, tmp_path):
    if backend == "vispy":
        pytest.importorskip("vispy")
    xy = particles.positions - 0.5
    velocity = torch.stack((-xy[:, 1], xy[:, 0]), dim=1)
    s = visualize(
        particles,
        domain,
        velocity,
        PlottingOptions(
            mapping=Mapping.L2,
            gridVisualization=GridVisualization(resolution=24, streamLines=True),
        ),
        backend=backend,
        backendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        ),
    )
    try:
        s.export(str(tmp_path / "with_lines.png"))
        panel = s.plotStates["A"]
        if backend == "vispy":
            with_lines = s.fig.render().astype(int)
            panel.stream_visual.visible = False
            without_lines = s.fig.render().astype(int)
        else:
            # Hold the layout fixed while isolating the overlay's pixel effect.
            s.fig.set_layout_engine(None)
            s.fig.canvas.draw()
            with_lines = np.asarray(s.fig.canvas.buffer_rgba()).copy().astype(int)
            panel.streamLines.lines.set_visible(False)
            for arrow in panel.streamLines._warpsph_arrows:
                arrow.set_visible(False)
            s.fig.canvas.draw()
            without_lines = np.asarray(s.fig.canvas.buffer_rgba()).copy().astype(int)
        delta = with_lines - without_lines
        assert (delta < -20).sum() > 100
        assert delta.max() <= 5
    finally:
        s.close()


@pytest.mark.parametrize("backend", ["matplotlib", "vispy"])
def test_colorbar_resize_layout(backend, particles, domain, tmp_path):
    if backend == "vispy":
        pytest.importorskip("vispy")
    s = visualize(
        particles,
        domain,
        torch.linspace(0.997, 1.003, 16),
        PlottingOptions(plotTitle="Density"),
        mosaic="AB",
        backend=backend,
        backendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        ),
        figsize=(10, 5),
        figTitle="Resizing all panels",
    )
    try:
        s.export(str(tmp_path / "initial.png"))
        if backend == "vispy":
            s.fig.size = (760, 360)
        else:
            s.fig.set_size_inches(7.6, 3.6)
        s.updateQuantities(torch.linspace(0.000999997, 1.001001, 16), redraw=False)
        s.export(str(tmp_path / "resized.png"))
        if backend == "vispy":
            b = s._backend_instance
            assert tuple(s.fig.central_widget.size) == (760, 360)
            for panel in s.plotStates.values():
                widget = panel.colorbar
                strip = widget._colorbar
                transform = widget.node_transform(s.fig.scene)
                bar_left = transform.map(
                    (strip.pos[0] - strip._halfdim[0], strip.pos[1])
                )[0]
                view_right = panel.view.node_transform(s.fig.scene).map(
                    (panel.view.size[0], 0)
                )[0]
                assert bar_left - view_right == pytest.approx(12, abs=1)
                for tick in strip._ticks:
                    assert tick.rotation == 0
                    x, y = transform.map(tick.pos[0])[:2]
                    assert 0 <= x < 760 and 0 <= y < 360
                assert panel.title_visual.pos[1] < panel.view.pos[1]
        else:
            renderer = s.fig.canvas.get_renderer()
            for panel in s.plotStates.values():
                bounds = panel.axis.get_tightbbox(renderer)
                assert bounds.x0 >= 0 and bounds.y0 >= 0
                assert bounds.x1 <= s.fig.bbox.width and bounds.y1 <= s.fig.bbox.height
                cb = panel.fluidScatterResult[1]
                assert 0 < cb.ax.bbox.x0 - panel.axis.bbox.x1 < 25
                assert all(
                    label.get_rotation() == 0 for label in cb.ax.get_yticklabels()
                )
    finally:
        s.close()


@pytest.mark.parametrize("grid", [False, True])
def test_matplotlib_symlog_tick_labels_do_not_overlap(
    particles, domain, tmp_path, grid
):
    s = visualize(
        particles,
        domain,
        torch.linspace(-10, 10, 16),
        PlottingOptions(
            quantityScaling=PlotScaling.SymmetricLog,
            gridVisualization=GridVisualization(resolution=16) if grid else None,
        ),
        backend="matplotlib",
        figsize=(4, 3),
    )
    try:
        s.export(str(tmp_path / "symlog.png"))
        panel = s.plotStates["A"]
        cb = (panel.gridResult if grid else panel.fluidScatterResult)[1]
        renderer = s.fig.canvas.get_renderer()
        boxes = sorted(
            (
                t.get_window_extent(renderer)
                for t in cb.ax.get_yticklabels()
                if t.get_visible()
            ),
            key=lambda b: b.y0,
        )
        assert len(boxes) >= 3
        assert 0 in cb.get_ticks()
        assert all(a.y1 < b.y0 for a, b in zip(boxes, boxes[1:]))
    finally:
        s.close()

"""Physical and pixel regressions for readable, camera-aware particle sizing."""

import copy
import numpy as np
import pytest
import torch
from warpSPHPlotting import visualize, PlottingOptions
from warpSPHPlotting._sizing import particle_spacing


def test_volume_spacing_is_independent_of_close_pairs_and_mode_cache(particles):
    p = copy.copy(particles)
    p.positions = particles.positions * 0.2
    p._spacingCache = {}
    assert particle_spacing(p, "volume") == pytest.approx(0.25)
    assert particle_spacing(p, "neighbors") == pytest.approx(0.05, rel=1e-5)
    assert particle_spacing(p, "volume") == pytest.approx(0.25)
    p._spacingCache = {}
    p.densities = particles.densities * 4
    assert particle_spacing(p, "volume") == pytest.approx(0.125)
    p._spacingCache = {}
    p.masses = torch.full((16,), float("nan"))
    assert particle_spacing(p, "volume") == pytest.approx(0.0375, rel=1e-5)
    p.positions = particles.positions[:0]
    p._spacingCache = {}
    assert particle_spacing(p, "volume") == 0


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista", "vispy"])
def test_automatic_volume_size_stays_stable_during_rearrangement(
    backend, particles, domain
):
    if backend != "matplotlib":
        pytest.importorskip(backend)
    extra = (
        {"jupyter_backend": "image", "app_backend": "egl"} if backend == "vispy" else {}
    )
    plot = visualize(
        particles,
        domain,
        torch.zeros(16),
        PlottingOptions(),
        backend=backend,
        backendOptions=extra,
    )

    def diameter():
        panel = plot.plotStates["A"]
        if backend == "matplotlib":
            return panel.fluidScatterResult[0].world_diameter
        if backend == "pyvista":
            return panel.particleDiameters["fluid"]
        return float(panel.markers_fluid._data["a_size"][0])

    try:
        initial = diameter()
        moved = copy.copy(particles)
        moved.positions = particles.positions * 0.2
        plot.updateQuantities(torch.zeros(16), newParticleState=moved, redraw=False)
        assert diameter() == pytest.approx(initial)
        plot.updateQuantities(torch.zeros(16), markerSpacing="neighbors", redraw=False)
        assert diameter() == pytest.approx(initial * 0.2, rel=1e-5)
        plot.updateQuantities(torch.zeros(16), markerSpacing="volume", redraw=False)
        assert diameter() == pytest.approx(initial)
        # Physical compression still changes size through mass / density.
        moved.densities = particles.densities * 4
        plot.updateQuantities(torch.zeros(16), newParticleState=moved, redraw=False)
        assert diameter() == pytest.approx(initial * 0.5)
        with pytest.raises(ValueError, match="markerSpacing"):
            plot.updateQuantities(
                torch.zeros(16), markerSpacing="invalid", redraw=False
            )
        assert diameter() == pytest.approx(initial * 0.5)
        plot.updateQuantities(torch.zeros(16), newParticleState=particles, redraw=False)
        assert diameter() == pytest.approx(initial)
    finally:
        plot.close()


def test_spacing_uses_local_compression_and_physical_resolution(particles):
    assert particle_spacing(particles) == pytest.approx(0.25)
    p = copy.copy(particles)
    p.positions = particles.positions * 0.2
    # Same nominal particle volume, compressed positions.
    assert particle_spacing(p) == pytest.approx(0.05, rel=1e-5)
    p.positions = particles.positions
    p.masses = particles.masses / 4
    assert particle_spacing(p) == pytest.approx(0.125)
    # Invalid volume data falls back to geometry, including boundary-only lines.
    p.positions = torch.stack((torch.arange(16.0), torch.zeros(16)), -1)
    p.masses = torch.full((16,), float("nan"))
    assert particle_spacing(p) == pytest.approx(0.9375)
    p.positions = particles.positions[:0]
    assert particle_spacing(p) == 0


def test_spacing_handles_3d_and_coincident_particles(particles):
    p = copy.copy(particles)
    x = torch.arange(3.0) / 4
    p.positions = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    p.masses, p.densities = torch.full((27,), 0.25**3), torch.ones(27)
    assert particle_spacing(p) == pytest.approx(0.25)
    p.positions[:] = 0
    assert particle_spacing(p) == pytest.approx(0.25)


def pixels(plot):
    if plot.backend == "matplotlib":
        plot.fig.canvas.draw()
        return np.asarray(plot.fig.canvas.buffer_rgba())[..., :3].copy()
    if plot.backend == "pyvista":
        plot.fig.render()
        return plot.fig.screenshot(return_img=True)[..., :3]
    return plot.fig.render()[..., :3]


def purple_width(rgb):
    # Only the particles are purple: frame/ticks/text are black and bars hidden.
    mask = (
        (rgb[..., 0] > 25)
        & (rgb[..., 0] < 120)
        & (rgb[..., 1] < 30)
        & (rgb[..., 2] > 30)
    )
    padded = np.pad(mask, ((0, 0), (1, 1)))
    edges = np.diff(padded.astype(int), axis=1)
    return int(np.max(np.where(edges == -1)[1] - np.where(edges == 1)[1]))


def zoom(plot, factor):
    if plot.backend == "matplotlib":
        axis = plot.plotStates["A"].axis
        for get, set_ in (
            (axis.get_xlim, axis.set_xlim),
            (axis.get_ylim, axis.set_ylim),
        ):
            lo, hi = get()
            center, half = (lo + hi) / 2, (hi - lo) / 2 / factor
            set_(center - half, center + half)
    elif plot.backend == "pyvista":
        plot.fig.camera.zoom(factor)
    else:
        plot.plotStates["A"].view.camera.zoom(1 / factor)


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista", "vispy"])
def test_automatic_markers_follow_zoom_resize_and_manual_override(
    backend, particles, domain
):
    if backend != "matplotlib":
        pytest.importorskip(backend)
    plot = visualize(
        particles,
        domain,
        torch.zeros(16),
        PlottingOptions(plotDomain=False, showColorBar=False, vMin=0, vMax=1),
        backend=backend,
        figsize=(6, 5),
        backendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        ),
    )
    try:
        initial = purple_width(pixels(plot))
        zoom(plot, 1.5)
        enlarged = purple_width(pixels(plot))
        assert enlarged / initial == pytest.approx(1.5, abs=0.08)
        if backend == "matplotlib":
            plot.fig.set_size_inches(7.5, 6.25)
        elif backend == "pyvista":
            plot.fig.window_size = (720, 600)
        else:
            plot.fig.size = (720, 600)
        resized = purple_width(pixels(plot))
        assert resized / enlarged == pytest.approx(1.25, abs=0.10)
        plot.updateQuantities(
            torch.zeros(16), newOptions={"A": {"markerSize": 36}}, redraw=False
        )
        fixed = purple_width(pixels(plot))
        zoom(plot, 1.5)
        assert abs(purple_width(pixels(plot)) - fixed) <= 1
        plot.updateQuantities(
            torch.zeros(16),
            newOptions={"A": {"markerSize": None, "markerSizeScale": 0.4}},
            redraw=False,
        )
        assert purple_width(pixels(plot)) > fixed * 2
    finally:
        plot.close()


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista", "vispy"])
def test_fluid_and_boundary_have_independent_resolution(backend, particles, domain):
    from warpSPHPlotting import VisualizeOptions

    if backend != "matplotlib":
        pytest.importorskip(backend)
    p = copy.copy(particles)
    p.positions = torch.cat(
        (
            particles.positions,
            torch.stack((torch.arange(8) / 8, torch.full((8,), 0.02)), -1),
        )
    )
    p.masses = torch.cat((particles.masses, torch.full((8,), 1 / 64)))
    p.densities = torch.ones(24)
    p.supports = torch.full((24,), 0.7)
    p.kinds = torch.cat((particles.kinds, torch.ones(8, dtype=particles.kinds.dtype)))
    plot = visualize(
        p,
        domain,
        torch.zeros(24),
        PlottingOptions(boundaryVisualization=VisualizeOptions.Passive),
        backend=backend,
        backendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        ),
    )
    try:
        panel = plot.plotStates["A"]
        if backend == "matplotlib":
            fluid, boundary = (
                panel.fluidScatterResult[0].world_diameter,
                panel.boundaryScatterResult[0].world_diameter,
            )
        elif backend == "pyvista":
            fluid, boundary = (
                panel.particleDiameters["fluid"],
                panel.particleDiameters["boundary"],
            )
        else:
            fluid, boundary = (
                panel.markers_fluid._data["a_size"][0],
                panel.markers_boundary._data["a_size"][0],
            )
        assert boundary / fluid == pytest.approx(0.5)
    finally:
        plot.close()


@pytest.mark.parametrize("backend", ["pyvista", "vispy"])
def test_backend_pixel_size_override_survives_zoom(backend, particles, domain):
    pytest.importorskip(backend)
    opts = {"point_size": 5}
    if backend == "vispy":
        opts.update(jupyter_backend="image", app_backend="egl")
    plot = visualize(
        particles,
        domain,
        torch.zeros(16),
        PlottingOptions(plotDomain=False, showColorBar=False, vMin=0, vMax=1),
        backend=backend,
        backendOptions=opts,
    )
    try:
        before = purple_width(pixels(plot))
        assert before == pytest.approx(5, abs=1)
        zoom(plot, 1.5)
        assert abs(purple_width(pixels(plot)) - before) <= 1
        # Per-panel sizes take precedence over the backend-wide default.
        plot.updateQuantities(
            torch.zeros(16), newOptions={"A": {"markerSize": 144}}, redraw=False
        )
        assert purple_width(pixels(plot)) > before * 2
    finally:
        plot.close()


def test_invalid_size_options_rejected_before_panel_changes(particles, domain):
    plot = visualize(particles, domain, torch.zeros(16), PlottingOptions(), mosaic="AB")
    try:
        initial = plot.plotStates["A"].fluidScatterResult[0].world_diameter
        with pytest.raises(ValueError, match="markerSizeScale"):
            plot.updateQuantities(
                torch.zeros(16),
                newOptions={"A": {"markerSize": 36}, "B": {"markerSizeScale": 0}},
                redraw=False,
            )
        assert plot.plotStates["A"].fluidScatterResult[0].world_diameter == initial
        with pytest.raises(ValueError, match="markerSize"):
            plot.updateQuantities(
                torch.zeros(16),
                newOptions={"A": {"markerSize": float("nan")}},
                redraw=False,
            )
    finally:
        plot.close()


def test_matplotlib_auto_markers_export_at_final_dpi_and_support_legends(
    particles, domain, tmp_path
):
    from PIL import Image

    plot = visualize(
        particles,
        domain,
        torch.zeros(16),
        PlottingOptions(plotDomain=False, showColorBar=False, vMin=0, vMax=1),
    )
    try:
        widths = []
        for dpi in (100, 200):
            path = tmp_path / f"particles_{dpi}.png"
            plot.export(str(path), dpi=dpi)
            widths.append(purple_width(np.asarray(Image.open(path).convert("RGB"))))
        assert widths[1] / widths[0] == pytest.approx(2, abs=0.05)
        panel = plot.plotStates["A"]
        panel.axis.legend([panel.fluidScatterResult[0]], ["Fluid"])
        # The legend constructs its own collection, outside our panel lifecycle.
        plot.export(str(tmp_path / "legend.svg"))
    finally:
        plot.close()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_spacing_preserves_compression_with_strides_and_invalid_positions(
    dim, dtype, strided
):
    from types import SimpleNamespace

    x = torch.arange(8, dtype=dtype) * 0.125
    pos = torch.stack(torch.meshgrid(*([x] * dim), indexing="ij"), -1).reshape(-1, dim)
    pos[:, 0] *= 0.25
    pos = torch.cat(
        [
            pos,
            torch.full((1, dim), float("nan"), dtype=dtype),
            torch.full((1, dim), float("inf"), dtype=dtype),
        ]
    )
    if strided:
        storage = torch.zeros((len(pos), dim * 2), dtype=dtype)
        storage[:, ::2] = pos
        pos = storage[:, ::2]
    particles = SimpleNamespace(
        positions=pos,
        masses=torch.full((len(pos),), 0.125**dim, dtype=dtype),
        densities=torch.ones(len(pos), dtype=dtype),
    )
    assert particle_spacing(particles) == pytest.approx(0.03125, rel=1e-6)

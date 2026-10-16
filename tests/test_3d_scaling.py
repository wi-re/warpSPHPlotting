"""Native 3D visibility, view transitions, and dependency-aware routing."""

from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from warpSPHCore import DomainDescription, ParticleState
from warpSPHPlotting import (
    Backend,
    GridVisualization,
    PlottingOptions,
    VisualizeOptions,
    visualize,
)
from warpSPHPlotting.backends import factory


def selection_frame(n=0, dim=2, grid=None, boundary=0):
    return SimpleNamespace(
        domain=SimpleNamespace(dim=dim),
        options=PlottingOptions(gridVisualization=grid),
        fluid=SimpleNamespace(positions=range(n)),
        boundary=SimpleNamespace(positions=range(boundary)),
    )


def test_auto_selection_counts_visible_work_and_honors_choices(monkeypatch):
    monkeypatch.setattr(factory, "find_spec", lambda name: object())
    assert (
        factory.select_backend(Backend.Auto, [selection_frame(49_999)]) == "matplotlib"
    )
    assert factory.select_backend("auto", [selection_frame(25_000)] * 2) == "pyvista"
    f = selection_frame(60_000, boundary=60_000)
    f.options.fluidVisualization = VisualizeOptions.Hide
    assert factory.select_backend("auto", [f]) == "matplotlib"
    f.options.boundaryVisualization = VisualizeOptions.Passive
    assert factory.select_backend("auto", [f]) == "pyvista"
    assert (
        factory.select_backend(
            "auto", [selection_frame(60_000, grid=GridVisualization())]
        )
        == "matplotlib"
    )
    assert factory.select_backend("auto", [selection_frame(dim=3)]) == "pyvista"
    assert (
        factory.select_backend("auto", [selection_frame(60_000)], sharey=False)
        == "vispy"
    )
    with pytest.raises(ValueError, match="whole cameras"):
        factory.select_backend(
            "auto", [selection_frame(dim=3, grid=GridVisualization())], sharey=False
        )
    assert (
        factory.select_backend(Backend.Matplotlib, [selection_frame(1_000_000, 3)])
        == "matplotlib"
    )


def test_auto_missing_dependencies(monkeypatch):
    monkeypatch.setattr(
        factory, "find_spec", lambda name: object() if name == "vispy" else None
    )
    assert factory.select_backend("auto", [selection_frame(50_000)]) == "vispy"
    assert factory.select_backend("auto", [selection_frame(dim=3)]) == "vispy"
    with pytest.raises(ImportError, match="plot-pyvista"):
        factory.select_backend(
            "auto", [selection_frame(dim=3, grid=GridVisualization())]
        )
    monkeypatch.setattr(factory, "find_spec", lambda name: None)
    with pytest.warns(RuntimeWarning, match="50,000"):
        assert factory.select_backend("auto", [selection_frame(50_000)]) == "matplotlib"
    with pytest.raises(ImportError, match="3D"):
        factory.select_backend("auto", [selection_frame(dim=3)])


def cube():
    x = torch.linspace(0.3, 0.7, 4)
    pos = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    return ParticleState(
        positions=pos,
        supports=torch.full((len(pos),), 0.28),
        masses=torch.full((len(pos),), (0.4 / 3) ** 3),
        densities=torch.ones(len(pos)),
        kinds=torch.zeros(len(pos), dtype=torch.int32),
    ), DomainDescription(
        torch.zeros(3), torch.ones(3), torch.zeros(3, dtype=torch.bool), 3
    )


def test_volume_zero_coverage_updates_and_mode_lifecycle(tmp_path):
    pv = pytest.importorskip("pyvista")
    from PIL import Image

    p, d = cube()
    q = torch.zeros(len(p.positions))
    s = visualize(
        p,
        d,
        q,
        PlottingOptions(
            vMin=0, vMax=1, gridVisualization=GridVisualization(resolution=12)
        ),
    )
    try:
        assert s.backend == "pyvista"
        panel = s.plotStates["A"]
        actor = panel.actor_grid
        assert isinstance(actor, pv.Volume)
        codes = panel.mesh_grid["volume_quantity"]
        assert (codes == 0).any() and (codes == 1).any()
        assert actor.prop.GetScalarOpacity().GetValue(0) == 0
        assert actor.prop.GetScalarOpacity().GetValue(1) > 0
        assert np.all(panel.mesh_grid["quantity"] == 0)
        s.export(str(tmp_path / "zero.png"))
        camera = copy.deepcopy(s.fig.camera_position)
        s.updateQuantities(torch.ones_like(q), redraw=False)
        assert panel.actor_grid is actor and s.fig.camera_position == camera
        s.export(str(tmp_path / "one.png"))
        a = np.asarray(Image.open(tmp_path / "zero.png")).astype(float)
        b = np.asarray(Image.open(tmp_path / "one.png")).astype(float)
        assert np.mean(np.abs(a - b)) > 0.5
        counts = []
        for _ in range(2):
            for mode in ("surface", "volume"):
                s.updateQuantities(
                    q,
                    newOptions={
                        "A": {
                            "gridVisualization": GridVisualization(
                                resolution=10, renderMode=mode
                            )
                        }
                    },
                    redraw=False,
                )
                assert isinstance(panel.actor_grid, pv.Volume) == (mode == "volume")
                s.export(str(tmp_path / f"{mode}.png"))
            counts.append(len(s.fig.renderer.actors))
        assert counts[0] == counts[1]
        empty = copy.copy(p)
        empty.positions = p.positions + 5
        s.updateQuantities(q, newParticleState=empty, redraw=False)
        assert not np.any(panel.mesh_grid["volume_quantity"])
        assert panel.colorbar is None
        s.updateQuantities(q, newParticleState=p, redraw=False)
        assert panel.colorbar is not None
    finally:
        s.close()


@pytest.mark.parametrize("backend", ["pyvista", "vispy"])
def test_mixed_3d_slice_cameras_and_round_trip(backend, tmp_path):
    pytest.importorskip(backend)
    p, d = cube()
    q = p.positions[:, 2]
    beopts = (
        {"jupyter_backend": "image", "app_backend": "egl"} if backend == "vispy" else {}
    )
    options = {
        "A": PlottingOptions(),
        "B": PlottingOptions(
            gridVisualization=GridVisualization(resolution=8, sliceAxis=2)
        ),
    }
    s = visualize(p, d, q, options, mosaic="AB", backend=backend, backendOptions=beopts)
    try:
        a, b = s.plotStates.values()
        if backend == "pyvista":
            assert s.fig.renderers[0].camera is not s.fig.renderers[1].camera
            assert not s.fig.renderers[0].camera.parallel_projection
            assert s.fig.renderers[1].camera.parallel_projection
        else:
            assert type(a.view.camera) != type(b.view.camera)
            assert not a.view.camera._linked_cameras
        s.export(str(tmp_path / "mixed.png"))
        for _ in range(2):
            for grid in (None, GridVisualization(resolution=8, sliceAxis=2)):
                s.updateQuantities(
                    q,
                    key="B",
                    newOptions={"B": {"gridVisualization": grid}},
                    redraw=False,
                )
                s.export(str(tmp_path / "transition.png"))
                if backend == "pyvista":
                    assert s.fig.renderers[1].camera.parallel_projection == (
                        grid is not None
                    )
                    assert (s.fig.renderers[0].camera is s.fig.renderers[1].camera) == (
                        grid is None
                    )
                else:
                    assert bool(a.view.camera._linked_cameras) == (grid is None)
    finally:
        s.close()


@pytest.mark.parametrize("backend", ["pyvista", "vispy"])
@pytest.mark.parametrize("world_scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("perspective", [False, True])
def test_3d_occlusion_is_independent_of_particle_input_order(
    backend, world_scale, perspective, tmp_path
):
    pytest.importorskip(backend)
    from PIL import Image

    p, d = cube()
    p.positions = torch.tensor([[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]]) * world_scale
    d.min = d.min * world_scale
    d.max = d.max * world_scale
    for name in ("supports", "masses", "densities", "kinds"):
        setattr(p, name, getattr(p, name)[:2])
    q = torch.tensor([0.0, 1.0])
    beopts = (
        {"jupyter_backend": "image", "app_backend": "egl"} if backend == "vispy" else {}
    )
    s = visualize(
        p,
        d,
        q,
        PlottingOptions(
            markerSize=900, showColorBar=False, plotDomain=False, vMin=0, vMax=1
        ),
        backend=backend,
        backendOptions=beopts,
    )
    try:
        if backend == "pyvista":
            s.fig.camera_position = [
                tuple(np.array(p) * world_scale)
                for p in [(0.5, 0.5, 3), (0.5, 0.5, 0.5)]
            ] + [(0, 1, 0)]
            s.fig.camera.parallel_projection = not perspective
            s.fig.camera.parallel_scale = world_scale * 0.7
            s.fig.reset_camera_clipping_range()
        else:
            c = s.plotStates["A"].view.camera
            c.elevation, c.azimuth, c.fov = 90, 0, 40 if perspective else 0
            c.center = tuple(np.full(3, 0.5) * world_scale)
        s.export(str(tmp_path / "before.png"))
        for name in ("positions", "supports", "masses", "densities", "kinds"):
            setattr(p, name, getattr(p, name).flip(0))
        s.updateQuantities(q.flip(0), newParticleState=p, redraw=False)
        s.export(str(tmp_path / "after.png"))
        a = np.asarray(Image.open(tmp_path / "before.png"))[:, :, :3].astype(float)
        b = np.asarray(Image.open(tmp_path / "after.png"))[:, :, :3].astype(float)
        assert np.mean(np.abs(a - b)) < 0.05
        # The upper (front) particle is viridis yellow, not the purple rear one.
        assert ((a[:, :, 0] > 200) & (a[:, :, 1] > 180) & (a[:, :, 2] < 100)).sum() > 50
    finally:
        s.close()


def test_slice_constant_field_has_no_particle_layer_checkerboard():
    from warpSPHPlotting.backends._render_util import prepare_frame
    from warpSPHPlotting.grid import mapFrameToGrid

    nx = 16
    x = (torch.arange(nx) + 0.5) / nx
    pos = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    p = ParticleState(
        positions=pos,
        supports=torch.full((len(pos),), 2.2 / nx),
        masses=torch.full((len(pos),), 1 / nx**3),
        densities=torch.ones(len(pos)),
        kinds=torch.zeros(len(pos), dtype=torch.int32),
    )
    d = DomainDescription(
        torch.zeros(3), torch.ones(3), torch.ones(3, dtype=torch.bool), 3
    )
    for position in (0.5, 0.5 + 0.5 / nx):
        frame = prepare_frame(
            p,
            d,
            torch.ones(len(pos)),
            PlottingOptions(
                gridVisualization=GridVisualization(
                    resolution=48, sliceAxis=2, slicePosition=position
                )
            ),
        )
        _, values, _, extent = mapFrameToGrid(frame, frame.particles.quantities)
        assert extent["coverage"].all()
        assert float(values.max() - values.min()) < 0.08
        assert float(values.mean()) == pytest.approx(1, abs=0.08)


def test_pyvista_masked_points_keep_raw_data_without_occluding(tmp_path):
    pytest.importorskip("pyvista")
    p, d = cube()
    q = torch.full((len(p.positions),), float("nan"))
    q[0] = 1
    s = visualize(p, d, q, PlottingOptions(vMin=0, vMax=1), backend="pyvista")
    try:
        panel = s.plotStates["A"]
        assert panel.mesh_fluid.n_points == len(q)
        assert panel.mesh_fluid.n_verts == 1
        assert np.isnan(panel.mesh_fluid["quantity"][1:]).all()
        assert panel.actor_fluid.GetForceOpaque()
        s.export(str(tmp_path / "masked.png"))
        s.updateQuantities(torch.ones_like(q), redraw=False)
        assert panel.mesh_fluid.n_verts == len(q)
    finally:
        s.close()

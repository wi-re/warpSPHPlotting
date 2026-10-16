import copy
import time
import numpy as np
import pytest
import torch
from warpSPHPlotting import (
    visualize,
    PlottingOptions,
    PlotScaling,
    Mapping,
    Backend,
    GridVisualization,
    VisualizeOptions,
    generateGrid,
)
from warpSPHPlotting.math import getBounds, getSharedBounds, mapQuantity
from warpSPHPlotting.state import assembleQuantity


@pytest.mark.parametrize("scale", [PlotScaling.Symmetric, PlotScaling.SymmetricLog])
def test_signed_norm(scale):
    q, norm = getBounds(
        torch.tensor([-10.0, 0.0, 10.0]), PlottingOptions(quantityScaling=scale)
    )
    np.testing.assert_allclose(q, [-10, 0, 10])
    np.testing.assert_allclose(norm(q), [0, 0.5, 1])


def test_median_is_per_frame():
    opts = PlottingOptions(quantityScaling=PlotScaling.Symmetric, midPoint="median")
    _, a = getBounds(torch.tensor([1.0, 2.0, 3.0]), opts)
    _, b = getBounds(torch.tensor([101.0, 102.0, 103.0]), opts)
    assert a.vcenter == 2 and b.vcenter == 102 and opts.midPoint == "median"


@pytest.mark.parametrize("q", [[1.0, float("nan"), 2.0], [1.0, float("inf"), 2.0]])
def test_invalid_samples_masked(q):
    values, norm = getBounds(torch.tensor(q), PlottingOptions())
    assert values.mask.tolist() == [False, True, False]
    assert (norm.vmin, norm.vmax) == (1, 2)


def test_log_masks_nonpositive():
    q, norm = getBounds(
        torch.tensor([-1.0, 0.0, 1.0, 100.0]),
        PlottingOptions(quantityScaling=PlotScaling.Logarithmic),
    )
    assert q.mask.tolist() == [True, True, False, False]
    assert norm(10.0) == pytest.approx(0.5)


@pytest.mark.parametrize("scale", list(PlotScaling))
def test_constant_values(scale):
    q, norm = getBounds(torch.ones(4), PlottingOptions(quantityScaling=scale))
    assert np.isfinite(norm(q)).all() and norm.vmin < norm.vmax


@pytest.mark.parametrize(
    "opts",
    [
        PlottingOptions(vMin=2, vMax=1),
        PlottingOptions(vMin=float("nan")),
        PlottingOptions(quantityScaling=PlotScaling.Logarithmic, vMin=-1),
    ],
)
def test_bad_bounds(opts):
    with pytest.raises(ValueError):
        getBounds(torch.tensor([1.0, 2.0]), opts)


def test_empty_requires_bounds():
    with pytest.raises(ValueError, match="No valid"):
        getBounds(torch.tensor([]), PlottingOptions())
    _, norm = getBounds(torch.tensor([]), PlottingOptions(vMin=0, vMax=1))
    assert norm.vmin == 0


def test_shared_norm():
    a, b, n = getSharedBounds(
        torch.tensor([1.0, 2.0]), torch.tensor([10.0]), PlottingOptions()
    )
    assert n.vmin == 1 and n.vmax == 10 and n(a)[0] == 0 and n(b)[0] == 1


def test_compact_kind_tuples(particles):
    particles.kinds[8:] = 1
    actual = assembleQuantity(particles, (torch.ones(8), torch.full((8,), 2.0)))
    assert actual.tolist() == [1] * 8 + [2] * 8
    assert torch.equal(
        actual, assembleQuantity(particles, (torch.ones(16), torch.full((16,), 2.0)))
    )


def test_mapping_shape():
    with pytest.raises(ValueError):
        mapQuantity(torch.ones(4, 2), Mapping.z)
    assert mapQuantity(torch.ones(4, 2, 2), Mapping.L2).shape == (4,)


def test_multi_panel_updates(particles, domain):
    s = visualize(
        particles,
        domain,
        torch.ones(16),
        PlottingOptions(showColorBar=False),
        mosaic="AB",
        backend=Backend.Matplotlib,
    )
    s.updateQuantities(torch.full((16,), 2.0), redraw=False)
    assert all(
        torch.all(p.fluidParticles.quantities == 2) for p in s.plotStates.values()
    )
    s.updateQuantities(torch.full((16,), 3.0), key="A", redraw=False)
    assert torch.all(s.plotStates["A"].fluidParticles.quantities == 3)
    assert torch.all(s.plotStates["B"].fluidParticles.quantities == 2)
    with pytest.raises(KeyError):
        s.updateQuantities(torch.ones(16), key="bad")
    assert torch.all(s.plotStates["A"].fluidParticles.quantities == 3)


def test_domains_and_camera(particles, domain):
    s = visualize(
        particles, domain, torch.ones(16), PlottingOptions(showColorBar=False)
    )
    s.axes["A"].set_xlim(0.2, 0.4)
    s.updateQuantities(torch.ones(16), redraw=False)
    assert s.axes["A"].get_xlim() == (0.2, 0.4)
    d = copy.copy(domain)
    d.max = domain.max * 2
    s.updateDomain(d)
    assert torch.equal(s.plotStates["A"].domain.max, d.max)
    assert s.axes["A"].get_xlim()[1] > 2
    for _ in range(5):
        s.updateQuantities(torch.ones(16), redraw=False)
    assert len(s.axes["A"].patches) == 1


def test_visibility_transitions(particles, domain):
    s = visualize(particles, domain, torch.arange(16.0), PlottingOptions())

    def update(**kw):
        s.updateQuantities(torch.arange(16.0), newOptions={"A": kw}, redraw=False)

    update(fluidVisualization=VisualizeOptions.Passive, markerSize=77)
    sc, cb, _ = s.plotStates["A"].fluidScatterResult
    assert sc.get_array() is None and cb is None and sc.get_sizes()[0] == 77
    update(fluidVisualization=VisualizeOptions.Hide)
    assert len(s.axes["A"].collections) == 0
    update(fluidVisualization=VisualizeOptions.Visualize)
    assert (
        len(s.axes["A"].collections) == 1
        and s.plotStates["A"].fluidScatterResult[1] is not None
    )


def test_boundary_only(particles, domain):
    particles.kinds[:] = 1
    s = visualize(
        particles,
        domain,
        torch.ones(16),
        PlottingOptions(boundaryVisualization=VisualizeOptions.Visualize),
    )
    assert s.plotStates["A"].boundaryScatterResult[1] is not None


def test_nonleaf_state(particles, domain):
    particles.positions = particles.positions.requires_grad_() * 1.0
    s = visualize(
        particles, domain, torch.ones(16), PlottingOptions(showColorBar=False)
    )
    assert s.particleState.positions.grad_fn is None
    assert particles.positions.grad_fn is not None


def test_grid_and_streamline_transitions(particles, domain):
    v = torch.ones(16, 2)
    s = visualize(particles, domain, v, PlottingOptions(mapping=Mapping.L2))
    for n, stream in [(4, False), (8, True), (8, False)]:
        s.updateQuantities(
            v,
            newOptions={
                "A": {
                    "gridVisualization": GridVisualization(
                        resolution=n, streamLines=stream
                    )
                }
            },
            redraw=False,
        )
        assert s.plotStates["A"].gridResult[0].get_array().size == n * n
        assert (s.plotStates["A"].streamLines is not None) == stream
    s.updateQuantities(v, newOptions={"A": {"gridVisualization": None}}, redraw=False)
    assert len(s.axes["A"].collections) == 1


def test_thin_and_edge_grid(domain):
    domain.max = torch.tensor([1000.0, 1.0])
    pts, dx, n, vol, e = generateGrid(domain, 16, alignment="edge")
    assert n == [16, 2] and torch.isfinite(pts).all()
    assert float(pts[2, 0] - pts[0, 0]) == pytest.approx(float(dx[0]))
    with pytest.raises(ValueError):
        generateGrid(domain, 0)


def test_async_failure_is_retained_and_drain_completes():
    from warpSPHPlotting.backends.asyncExport import submitImageWrite, waitForExports

    finished = []

    class Bad:
        def save(self, *a, **kw):
            raise OSError("failed write")

    class Good:
        def save(self, *a, **kw):
            time.sleep(0.03)
            finished.append(True)

    submitImageWrite(Bad(), "unused", {})
    time.sleep(0.03)
    submitImageWrite(Good(), "unused", {})
    with pytest.raises(OSError, match="failed write"):
        waitForExports()
    assert finished == [True]


def test_failed_multi_panel_validation_preserves_existing_data(particles, domain):
    s = visualize(
        particles,
        domain,
        torch.ones(16),
        PlottingOptions(showColorBar=False),
        mosaic="AB",
    )
    with pytest.raises(ValueError):
        s.updateQuantities(
            {"A": torch.full((16,), 2.0), "B": torch.zeros(3)}, redraw=False
        )
    assert all(
        torch.all(p.fluidParticles.quantities == 1) for p in s.plotStates.values()
    )


def test_no_spurious_pyvista_sleep(monkeypatch):
    from warpSPHPlotting.visualize import _yield_notebook_events

    def forbidden(*args):
        raise AssertionError("non-notebook updates must not sleep")

    monkeypatch.setattr("time.sleep", forbidden)
    _yield_notebook_events(0.02)


def test_rotation_consistent_between_create_and_update(particles, domain):
    from types import SimpleNamespace

    d = SimpleNamespace(
        min=domain.min, max=domain.max, periodic=domain.periodic, dim=2, angles=0.3
    )
    s = visualize(particles, d, torch.arange(16.0), PlottingOptions(showColorBar=False))
    before = s.plotStates["A"].fluidParticles.positions.clone()
    s.updateQuantities(torch.arange(16.0), redraw=False)
    assert torch.allclose(before, s.plotStates["A"].fluidParticles.positions)


def test_real_sph_gradient_creation_and_update(particles, domain):
    from warpSPHCore import (
        warpOperation,
        OperationProperties,
        WarpOperation,
        GradientScheme,
        KernelFunctions,
    )

    op = OperationProperties(
        kernel=KernelFunctions.Wendland4,
        operation=WarpOperation.Gradient,
        gradientMode=GradientScheme.Difference,
    )
    q = particles.positions[:, 0]
    expected = warpOperation(
        queryParticles=particles, queryValues=q, operationProperties=op, domain=domain
    )
    s = visualize(
        particles,
        domain,
        q,
        PlottingOptions(plottingOperation=op, mapping=Mapping.x, showColorBar=False),
    )
    torch.testing.assert_close(
        s.plotStates["A"].fluidParticles.quantities, expected[:, 0]
    )
    s.updateQuantities(q * 2, redraw=False)
    torch.testing.assert_close(
        s.plotStates["A"].fluidParticles.quantities, expected[:, 0] * 2
    )


def test_quantities_do_not_retain_autograd_graphs(particles, domain):
    q = torch.ones(16, requires_grad=True) * 2
    s = visualize(particles, domain, q, PlottingOptions(showColorBar=False))
    assert s.quantities.grad_fn is None and not s.quantities.requires_grad
    q.detach().add_(3)
    assert torch.all(s.quantities == 2)
    s.updateQuantities(q, redraw=False)
    assert s.quantities["A"].grad_fn is None


def test_scalar_grid_and_streamlines_share_frame_search(particles, domain, monkeypatch):
    import warpSPHPlotting.grid as grid

    original = grid.buildCompactHashMap
    builds = []

    def counted(*args, **kwargs):
        builds.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(grid, "buildCompactHashMap", counted)
    v = torch.ones(16, 2)
    s = visualize(
        particles,
        domain,
        v,
        PlottingOptions(
            mapping=Mapping.L2,
            gridVisualization=GridVisualization(resolution=4, streamLines=True),
        ),
    )
    assert len(builds) == 1
    moved = copy.copy(particles)
    moved.positions = particles.positions + 0.01
    s.updateQuantities(v, newParticleState=moved, redraw=False)
    assert len(builds) == 2

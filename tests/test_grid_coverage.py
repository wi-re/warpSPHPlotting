"""Unsupported cells are transparent; supported zero-valued fluid stays visible."""

import copy
import numpy as np
import pytest
import torch
from warpSPHPlotting import (
    visualize,
    PlottingOptions,
    GridVisualization,
    VisualizeOptions,
)
from warpSPHPlotting.backends._render_util import prepare_frame
from warpSPHPlotting.grid import mapFrameToGrid, gridDisplayBounds
from warpSPHPlotting.streamLines import streamlineSegments
from test_particle_sizes import pixels


def small_cloud(particles):
    p = copy.copy(particles)
    p.positions = particles.positions * 0.2 + 0.4
    p.supports = torch.full((16,), 0.1)
    p.masses = torch.full((16,), 0.05**2)
    return p


def test_coverage_is_independent_of_quantity_and_reuses_search(particles, domain):
    p = small_cloud(particles)
    options = PlottingOptions(gridVisualization=GridVisualization(resolution=40))
    f = prepare_frame(p, domain, torch.zeros(16), options)
    gs, values, n, extent = mapFrameToGrid(f, f.particles.quantities)
    assert torch.equal(values, torch.zeros_like(values))
    coverage = extent["coverage"]
    assert coverage.any() and not coverage.all()
    qs, norm = gridDisplayBounds(values, extent, options)
    np.testing.assert_array_equal(np.ma.getmaskarray(qs), ~coverage.numpy())
    assert norm.vmin < 0 < norm.vmax  # Supported zeros define a valid scale.
    search = f.gridCache["_hash"]
    _, ones, _, second = mapFrameToGrid(f, torch.ones(16))
    assert f.gridCache["_hash"] is search and second["coverage"] is coverage
    assert (ones[coverage] > 0).all() and (ones[~coverage] == 0).all()
    positive, positive_norm = gridDisplayBounds(ones, extent, options)
    assert positive_norm.vmin > 0  # Empty zeros cannot lower the color scale.


def test_periodic_grid_has_no_false_holes(particles, domain):
    domain.periodic[:] = True
    f = prepare_frame(
        particles,
        domain,
        torch.ones(16),
        PlottingOptions(gridVisualization=GridVisualization(resolution=32)),
    )
    _, _, _, ext = mapFrameToGrid(f, f.particles.quantities)
    assert ext["coverage"].all()


def test_streamlines_stop_at_coverage_boundary(particles, domain):
    p = small_cloud(particles)
    options = PlottingOptions(gridVisualization=GridVisualization(resolution=40))
    f = prepare_frame(p, domain, torch.zeros(16), options)
    gs, _, n, ext = mapFrameToGrid(f, f.particles.quantities)
    # Deliberately nonzero vectors in the void: coverage, not zero speed,
    # must stop the integrator.
    vectors = torch.zeros(len(gs.positions), 2)
    vectors[:, 0] = 1
    edges = streamlineSegments(gs, vectors, n, ext)
    assert len(edges) > 0
    assert edges.min() > 0.29 and edges.max() < 0.71


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista", "vispy"])
def test_rendered_zeros_survive_empty_space_mask_and_updates(
    backend, particles, domain
):
    if backend != "matplotlib":
        pytest.importorskip(backend)
    p = small_cloud(particles)
    plot = visualize(
        p,
        domain,
        torch.zeros(16),
        PlottingOptions(
            plotDomain=False,
            showColorBar=False,
            vMin=-1,
            vMax=1,
            gridVisualization=GridVisualization(resolution=64, maskEmpty=False),
        ),
        backend=backend,
        backendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        ),
    )

    def green_count():
        rgb = pixels(plot)
        return np.sum(
            (rgb[..., 0] < 80)
            & (rgb[..., 1] > 80)
            & (rgb[..., 1] < 200)
            & (rgb[..., 2] > 80)
        )

    try:
        filled = green_count()
        plot.updateQuantities(
            torch.zeros(16),
            newOptions={
                "A": {
                    "gridVisualization": GridVisualization(resolution=64),
                    "showColorBar": True,
                }
            },
            redraw=False,
        )
        masked = green_count()
        assert masked > 500  # Genuine zeros still render green.
        assert masked < filled * 0.25  # Void no longer has zero-value color.
        # A support update must invalidate coverage, including an empty view.
        tiny = copy.copy(p)
        tiny.supports = torch.full((16,), 1e-6)
        plot.updateQuantities(torch.zeros(16), newParticleState=tiny, redraw=False)
        assert green_count() == 0
        panel = plot.plotStates["A"]
        if backend == "matplotlib":
            assert panel.gridResult[1] is None
        else:
            assert panel.colorbar is None
        plot.updateQuantities(
            torch.zeros(16),
            newParticleState=p,
            newOptions={"A": {"showColorBar": True}},
            redraw=False,
        )
        assert green_count() > 500
        if backend == "matplotlib":
            assert panel.gridResult[1] is not None
        else:
            assert panel.colorbar is not None
    finally:
        plot.close()


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista", "vispy"])
@pytest.mark.parametrize("spacing_mode", ["volume", "neighbors"])
def test_sizing_cache_shared_only_within_current_update(
    backend, spacing_mode, particles, domain, monkeypatch
):
    import warpSPHPlotting._sizing as sizing

    if backend != "matplotlib":
        pytest.importorskip(backend)
    estimator = (
        "_volume_particle_spacing"
        if spacing_mode == "volume"
        else "_estimate_particle_spacing"
    )
    original, calls = getattr(sizing, estimator), []

    def count(p):
        calls.append(len(p.positions))
        return original(p)

    monkeypatch.setattr(sizing, estimator, count)
    plot = visualize(
        particles,
        domain,
        torch.zeros(16),
        {
            "A": PlottingOptions(markerSizeScale=0.5, markerSpacing=spacing_mode),
            "B": PlottingOptions(markerSizeScale=0.8, markerSpacing=spacing_mode),
            "C": PlottingOptions(markerSize=5),
        },
        mosaic="ABC",
        backend=backend,
        backendOptions=(
            {"jupyter_backend": "image", "app_backend": "egl"}
            if backend == "vispy"
            else {}
        ),
    )
    try:
        assert calls == [16]  # Different scales share the unscaled estimate.
        moved = small_cloud(particles)
        plot.updateQuantities(torch.zeros(16), newParticleState=moved, redraw=False)
        assert calls == [16, 16]
        plot.updateQuantities(torch.ones(16), redraw=False)
        assert calls == [16, 16, 16]  # Never reuse a spatial estimate across updates.
    finally:
        plot.close()


def test_vispy_grid_reuses_transform_without_stale_pixels(particles, domain):
    pytest.importorskip("vispy")
    plot = visualize(
        particles,
        domain,
        torch.zeros(16),
        PlottingOptions(
            vMin=0, vMax=2, gridVisualization=GridVisualization(resolution=24)
        ),
        backend="vispy",
        backendOptions={"jupyter_backend": "image", "app_backend": "egl"},
    )
    try:
        panel = plot.plotStates["A"]
        transform = panel.image_grid.transform
        before = pixels(plot).astype(float)
        plot.updateQuantities(torch.ones(16), redraw=False)
        assert panel.image_grid.transform is transform
        after = pixels(plot).astype(float)
        assert np.mean(np.abs(after - before)) > 1
        larger = copy.copy(domain)
        larger.min, larger.max = domain.min - 0.5, domain.max + 1
        plot.updateQuantities(
            torch.ones(16),
            newDomain=larger,
            newOptions={"A": {"gridVisualization": GridVisualization(resolution=32)}},
            redraw=False,
        )
        assert panel.image_grid.transform is transform
        np.testing.assert_allclose(transform.map([0, 0])[:2], larger.min.numpy())
        np.testing.assert_allclose(transform.map([32, 32])[:2], larger.max.numpy())
        pixels(plot)
    finally:
        plot.close()


def test_coverage_respects_visible_particle_kinds(particles, domain):
    p = small_cloud(particles)
    p.kinds = particles.kinds.clone()
    p.kinds[8:] = 1
    p.supports[:] = 0.04
    options = PlottingOptions(
        fluidVisualization=VisualizeOptions.Hide,
        boundaryVisualization=VisualizeOptions.Visualize,
        gridVisualization=GridVisualization(resolution=100),
    )
    f = prepare_frame(p, domain, torch.ones(16), options)
    _, _, n, ext = mapFrameToGrid(f, f.particles.quantities)
    covered = ext["coverage"].reshape(n)
    assert not covered[42, 42] and covered[52, 42]
    options.fluidVisualization, options.boundaryVisualization = (
        VisualizeOptions.Visualize,
        VisualizeOptions.Hide,
    )
    f = prepare_frame(p, domain, torch.ones(16), options)
    _, _, n, ext = mapFrameToGrid(f, f.particles.quantities)
    covered = ext["coverage"].reshape(n)
    assert covered[42, 42] and not covered[52, 42]


def test_empty_3d_slice_uses_physical_coverage(particles):
    from warpSPHCore import DomainDescription

    p = small_cloud(particles)
    p.positions = torch.cat((p.positions, torch.full((16, 1), 0.25)), dim=1)
    p.masses = torch.full((16,), 0.05**3)
    d = DomainDescription(
        torch.zeros(3), torch.ones(3), torch.zeros(3, dtype=torch.bool), 3
    )
    options = PlottingOptions(
        gridVisualization=GridVisualization(
            resolution=32, sliceAxis=2, slicePosition=0.8
        )
    )
    f = prepare_frame(p, d, torch.zeros(16), options)
    gs, values, _, extent = mapFrameToGrid(f, f.particles.quantities)
    assert gs.positions.shape[1] == 2
    assert not extent["coverage"].any()
    assert (values == 0).all()
    qs, norm = gridDisplayBounds(values, extent, options)
    assert qs.mask.all() and norm is None

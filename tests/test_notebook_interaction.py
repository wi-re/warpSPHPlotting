"""Camera behavior at the same VTK interactor boundary used by trame."""

import numpy as np
import pytest
import torch
import copy
from warpSPHCore import DomainDescription
from warpSPHPlotting import PlottingOptions, visualize


def viewport(plot):
    if plot.backend == "matplotlib":
        ax = plot.axes["A"]
        return np.array([*ax.get_xlim(), *ax.get_ylim()])
    if plot.backend == "vispy":
        rect = plot.plotStates["A"].view.camera.rect
        return np.array([rect.left, rect.bottom, rect.width, rect.height])
    return np.r_[
        np.asarray(list(plot.fig.camera_position)).ravel(),
        plot.fig.camera.parallel_scale,
        plot.fig.camera.GetWindowCenter(),
    ]


def zoom(plot):
    if plot.backend == "matplotlib":
        plot.axes["A"].set_xlim(0.2, 0.6)
        plot.axes["A"].set_ylim(0.2, 0.6)
    elif plot.backend == "vispy":
        plot.plotStates["A"].view.camera.rect = (0.2, 0.2, 0.4, 0.4)
    else:
        plot.fig.camera.zoom(2)


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista", "vispy"])
def test_viewport_retention_refit_and_dynamic_bounds(backend, particles, domain):
    if backend != "matplotlib":
        pytest.importorskip(backend)
    extra = (
        {"jupyter_backend": "image", "app_backend": "egl"} if backend == "vispy" else {}
    )
    plot = visualize(
        particles,
        domain,
        torch.arange(16.0),
        PlottingOptions(),
        backend=backend,
        backendOptions=extra,
    )
    try:
        zoom(plot)
        moved = viewport(plot)
        shifted = copy.copy(particles)
        shifted.positions = particles.positions + 0.4
        # New buffers and an equal-valued domain are ordinary frame updates.
        for i in range(3):
            plot.updateQuantities(
                torch.arange(16.0) + i,
                newParticleState=shifted,
                newDomain=copy.deepcopy(domain),
                redraw=False,
            )
            np.testing.assert_array_equal(viewport(plot), moved)
        # Frame-to-frame presentation changes must also retain a panned view.
        for show_bar in (False, True):
            plot.updateQuantities(
                torch.arange(16.0), showColorBar=show_bar, redraw=False
            )
            np.testing.assert_array_equal(viewport(plot), moved)
        # Also detect an in-place mutation of the caller's domain tensors.
        domain.max *= 2
        plot.updateQuantities(torch.ones(16), newDomain=domain, redraw=False)
        assert not np.allclose(viewport(plot), moved)
        zoom(plot)
        moved = viewport(plot)
        plot.updateQuantities(torch.ones(16), resetView=True, redraw=False)
        assert not np.allclose(viewport(plot), moved)
        zoom(plot)
        moved = viewport(plot)
        shifted.positions = particles.positions + 4
        plot.updateQuantities(
            torch.ones(16), newParticleState=shifted, dynamicBounds=True, redraw=False
        )
        assert not np.allclose(viewport(plot), moved)
        # Following includes moving particles without changing physics bounds.
        assert torch.equal(plot.plotStates["A"].domain.max, domain.max)
        if backend == "matplotlib":
            assert plot.axes["A"].get_xlim()[1] > 4.875
        elif backend == "vispy":
            assert plot.plotStates["A"].view.camera.rect.right > 4.875
        else:
            assert plot.plotStates["A"].frameBounds[1] > 4.875
        plot.updateQuantities(torch.ones(16), dynamicBounds=False, redraw=False)
        zoom(plot)
        moved = viewport(plot)
        shifted.positions = particles.positions + 8
        plot.updateQuantities(torch.ones(16), newParticleState=shifted, redraw=False)
        np.testing.assert_array_equal(viewport(plot), moved)
    finally:
        plot.close()


def test_notebook_sod_viewport_and_same_domain_case_reset(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    pytest.importorskip("ipywidgets")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from notebook_review import NotebookReview

    paths = []
    for frame, extent in enumerate((1.0, 2.0)):
        path = tmp_path / f"state_{frame}.npz"
        np.savez(
            path,
            positions=np.array([[0.5, 0.5]], dtype=np.float32),
            supports=np.ones(1, dtype=np.float32),
            masses=np.ones(1, dtype=np.float32),
            densities=np.ones(1, dtype=np.float32),
            kinds=np.zeros(1, dtype=np.int32),
            velocities=np.ones((1, 2), dtype=np.float32),
            domain_min=np.zeros(2, dtype=np.float32),
            domain_max=np.full(2, extent, dtype=np.float32),
            periodic=np.zeros(2, dtype=bool),
        )
        paths.append(path)
    review = NotebookReview(root=tmp_path)
    review.case.unobserve(review._case_changed, names="value")
    review._changing = True
    review.sources = {"Sod · sparse viewport": paths, "Other case": paths[:1]}
    review.case.options = list(review.sources)
    review.case.value = "Sod · sparse viewport"
    review.frame.max = 1
    first = review._load()[1]
    review.frame.value = 1
    second = review._load()[1]
    assert torch.equal(first.min, second.min) and torch.equal(first.max, second.max)
    review.dynamic.value = True
    followed = review._load()[1]
    assert not torch.equal(first.max, followed.max)
    assert review._options()[1].dynamicBounds
    calls = []
    review.plot = SimpleNamespace(
        plotStates={"A": None},
        updateTitle=lambda title: None,
        updateQuantities=lambda *args, **kwargs: calls.append(kwargs),
    )
    review.camera = lambda: {"camera": "new case fit"}
    review._changing = False
    review.case.observe(review._case_changed, names="value")
    review.case.value = "Other case"
    assert calls[-1]["resetView"] is True
    assert not review._reset_after_update


@pytest.mark.parametrize("dimension", [2, 3])
def test_pyvista_drag_pans_planar_views_and_rotates_native_3d(
    dimension, particles, domain, tmp_path
):
    pytest.importorskip("pyvista")
    if dimension == 3:
        particles.positions = torch.column_stack(
            (particles.positions, torch.linspace(0.2, 0.8, len(particles.positions)))
        )
        domain = DomainDescription(
            torch.zeros(3), torch.ones(3), torch.zeros(3, dtype=torch.bool), 3
        )
    plot = visualize(
        particles,
        domain,
        torch.arange(len(particles.positions), dtype=torch.float32),
        PlottingOptions(markerSize=9),
        backend="pyvista",
    )
    try:
        plot.export(str(tmp_path / "before.png"))
        camera = plot.fig.camera
        direction = np.array(camera.direction)
        focal = np.array(camera.focal_point)
        up = np.array(camera.up)
        iren = plot.fig.iren.interactor
        iren.SetEventInformation(250, 230, 0, 0)
        iren.InvokeEvent("LeftButtonPressEvent")
        iren.SetEventInformation(300, 260, 0, 0)
        iren.InvokeEvent("MouseMoveEvent")
        iren.InvokeEvent("LeftButtonReleaseEvent")
        if dimension == 2:
            np.testing.assert_allclose(camera.direction, direction, atol=1e-10)
            np.testing.assert_allclose(camera.up, up, atol=1e-10)
            assert not np.allclose(camera.focal_point, focal)
        else:
            assert not np.allclose(camera.direction, direction)
        plot.export(str(tmp_path / "after.png"))
    finally:
        plot.close()


def test_pyvista_client_mode_rejected_before_allocating_a_window():
    pytest.importorskip("pyvista")
    from warpSPHPlotting.backends.pyvista_backend import PyVistaBackend

    backend = PyVistaBackend()
    with pytest.raises(ValueError, match="physical colorbar"):
        backend.create_figure(
            "A",
            (7, 5),
            True,
            True,
            None,
            {"jupyter_backend": "trame", "jupyter_kwargs": {"mode": "client"}},
        )
    assert backend._plotter is None

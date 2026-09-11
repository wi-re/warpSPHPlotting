"""PyVista rendering backend (Phase 3).

This backend represents particles as ``pv.PolyData`` point clouds, uses
matplotlib colormaps for scalar colouring (pyvista accepts the same names),
and supports efficient in-place scalar/position updates without rebuilding
the full scene.

Notebook display modes (pass via ``backendOptions={'jupyter_backend': ...}``):
  ``"static"``  — default; renders off-screen and shows a PNG snapshot inline.
  ``"trame"``   — interactive widget (requires ``trame``; install with
                  ``pip install 'warpSPHPlotting[plot-pyvista]'``).
  ``"none"``    — pop-out native window (good for desktop use).

Update pattern
--------------
After the initial :func:`~warpSPHPlotting.visualize.visualize` call, changes are
pushed via ``plotState.updateQuantities(...)`` which calls
:meth:`PyVistaBackend.update_panel` → ``mesh.points`` / ``mesh.point_data``
mutation + ``plotter.render()``.

For *static* notebook mode, call ``plotState.show()`` after each update to
capture a new screenshot.  For *trame*, the live widget updates automatically.

Current limitations (Phase 3)
------------------------------
- Grid-mapped visualisation (``GridVisualization``) is not yet supported.
- Streamlines are not yet supported.
- Log / symmetric-log scalar bar scales show clamped linear ranges.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .base import AbstractBackend
from ..enumTypes import VisualizeOptions
from ..options import PlottingOptions


# ─────────────────────────────────────────────────────────────────────────────
# Panel state dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PyVistaVisualizationState:
    """Per-panel rendering state kept by the PyVista backend."""

    plotter: Any                   # shared pv.Plotter reference
    row: int
    col: int

    # point-cloud meshes & actor handles; None if that particle type is hidden
    mesh_fluid: Optional[Any]
    actor_fluid: Optional[Any]
    mesh_boundary: Optional[Any]
    actor_boundary: Optional[Any]

    # grid (ImageData) mesh & actor; None when gridVisualization is off
    mesh_grid: Optional[Any]
    actor_grid: Optional[Any]

    domain: Any                    # DomainDescription (detached copy)
    options: PlottingOptions

    # per-type particle states and assembled quantity (for update bookkeeping)
    fluidParticles: Any
    boundaryParticles: Any
    ghostParticles: Any
    assembledQuantity: Any
    rotatedQuantities: Any         # full per-particle quantity after ops/mapping

    fluid_scalar_range: Optional[Tuple[float, float]] = None
    boundary_scalar_range: Optional[Tuple[float, float]] = None
    grid_scalar_range: Optional[Tuple[float, float]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_3d(positions) -> np.ndarray:
    """Convert (N, 2-or-3) float tensor → (N, 3) float64 numpy array."""
    pts = positions.detach().cpu().numpy().astype(np.float64)
    if pts.ndim == 1:
        pts = pts[:, None]
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    return pts


def _parse_mosaic(mosaic: str) -> Tuple[Tuple[int, int], Dict[str, Tuple[int, int]]]:
    """Parse a matplotlib mosaic string → (nrows, ncols), {label: (row, col)}.

    Handles simple rectangular mosaics like ``'AB'`` or ``'AB\\nCD'``.
    """
    rows = [r for r in mosaic.strip().split("\n") if r.strip()]
    nrows = len(rows)
    ncols = max(len(r) for r in rows)
    label_to_pos: Dict[str, Tuple[int, int]] = {}
    for r, row in enumerate(rows):
        for c, label in enumerate(row):
            if label.strip() and label not in label_to_pos:
                label_to_pos[label] = (r, c)
    return (nrows, ncols), label_to_pos


def _sharedMarkerBounds(fluid, boundary, opts):
    """`math.getSharedBounds` for the two marker sets, honouring Hide/Passive.

    A set that is hidden, drawn passive (flat grey), or empty contributes
    nothing to the norm -- it is not coloured by the quantity, so letting its
    values set the range would stretch the map for no visible gain.
    """
    from ..math import getSharedBounds  # noqa: PLC0415

    coloured = (VisualizeOptions.Hide, VisualizeOptions.Passive)
    fluidQuantities = (fluid.quantities
                       if opts.fluidVisualization not in coloured
                       and fluid.positions.shape[0] > 0 else None)
    boundaryQuantities = (boundary.quantities
                          if opts.boundaryVisualization not in coloured
                          and boundary.positions.shape[0] > 0 else None)
    return getSharedBounds(fluidQuantities, boundaryQuantities, opts)


def _scalar_range(q_clipped: np.ndarray, norm) -> Tuple[float, float]:
    """Extract a (vmin, vmax) pair from a matplotlib norm, with fallbacks."""
    import matplotlib.colors as mcolors  # noqa: PLC0415

    try:
        if isinstance(norm, mcolors.CenteredNorm) and norm.halfrange is not None:
            return (
                float(norm.vcenter - norm.halfrange),
                float(norm.vcenter + norm.halfrange),
            )
        vmin = norm.vmin
        vmax = norm.vmax
        if vmin is not None and vmax is not None:
            return float(vmin), float(vmax)
    except AttributeError:
        pass

    return float(np.nanmin(q_clipped)), float(np.nanmax(q_clipped))


def _apply_norm(q_clipped: np.ndarray, norm) -> np.ndarray:
    """Apply a matplotlib norm to produce values in [0, 1] for PyVista.

    Storing norm(q) in the PyVista mesh and using ``clim=(0, 1)`` ensures
    all norm types (Normalize, CenteredNorm, LogNorm, SymLogNorm) render
    with the correct colormap distribution regardless of data offset/scale.
    Without this, CenteredNorm with ``vcenter != 0`` causes the colormap to
    be shifted because PyVista applies a plain linear mapping between
    ``clim`` and the colormap range.
    """
    result = norm(q_clipped)
    if hasattr(result, 'data'):  # masked array returned by some norms
        result = result.data
    return np.clip(result.astype(np.float64), 0.0, 1.0)


def _grid_to_image_data(gridState, gridQuantity, nxs, gridExtent, options) -> Any:
    """Convert mapToGrid output to a ``pv.ImageData`` with scalar point data."""
    import pyvista as pv  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from ..math import getBounds  # noqa: PLC0415

    qs, norm = getBounds(gridQuantity, options)
    scalar_range = _scalar_range(qs, norm)

    # Infer origin and spacing from the grid positions.
    # generateGrid places points at cell centres; we recover spacing from the
    # difference between adjacent points along each axis.
    nx, ny = nxs[0], nxs[1]
    pts = gridState.positions.detach().cpu()
    dx = float(pts[ny, 0] - pts[0, 0]) if nx > 1 else float(gridExtent['max'][0] - gridExtent['min'][0])
    dy = float(pts[1, 1]  - pts[0, 1]) if ny > 1 else float(gridExtent['max'][1] - gridExtent['min'][1])
    ox = float(pts[0, 0])
    oy = float(pts[0, 1])

    image = pv.ImageData()
    image.dimensions = (nx, ny, 1)
    image.spacing   = (dx, dy, 1.0)
    image.origin    = (ox, oy, 0.0)
    # PyVista/VTK ImageData uses X-fastest (Fortran-like) flat order:
    #   flat index k = ix + nx * iy
    # But generateGrid uses indexing='ij' (C-order): k = ny * ix + iy
    # Transpose from (nx, ny) → (ny, nx) before C-flatten so X varies fastest.
    image.point_data["quantity"] = qs.reshape(nx, ny).T.reshape(-1).astype(np.float64)

    return image, scalar_range


def _make_domain_wireframe(domain_) -> Any:
    """Return a pyvista line mesh tracing the domain bounding box."""
    import pyvista as pv  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    dim = domain_.dim

    if dim == 2:
        # Explicit line segments forming the 2D domain rectangle.
        # pv.PolyData lines format: for each segment [2, i0, i1].
        pts = np.array([
            [mn[0], mn[1], 0.0],
            [mx[0], mn[1], 0.0],
            [mx[0], mx[1], 0.0],
            [mn[0], mx[1], 0.0],
        ], dtype=float)
        lines = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0], dtype=int)
        mesh = pv.PolyData()
        mesh.points = pts
        mesh.lines = lines
        return mesh
    else:
        bounds = [mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]]
        return pv.Box(bounds=bounds).extract_feature_edges()


def _cmap_name(options: PlottingOptions) -> str:
    return options.colorMap.value + ("_r" if options.flipColorMap else "")


def _point_size(options: PlottingOptions, default_size: float) -> float:
    """Resolve per-panel point size with backend default fallback."""
    if options.markerSize is None:
        return float(default_size)
    return float(options.markerSize)


def _in_ipykernel() -> bool:
    """Best-effort detection for notebook kernels (Jupyter/VS Code)."""
    try:
        from IPython import get_ipython  # noqa: PLC0415

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend implementation
# ─────────────────────────────────────────────────────────────────────────────

class PyVistaBackend(AbstractBackend):
    """PyVista / VTK-backed rendering backend (Phase 3)."""

    def __init__(self) -> None:
        self._plotter: Any = None
        self._axes: Dict[str, Tuple[int, int]] = {}
        self._jupyter_backend: str = "static"
        self._point_size: float = 8.0
        self._shown: bool = False
        self._display_handle: Any = None  # IPython display handle for in-place updates
        self._trame_view: Any = None      # keep viewer alive across incremental updates

    # -------------------------------------------------------------------------
    # Figure management
    # -------------------------------------------------------------------------

    def create_figure(
        self,
        mosaic: str,
        figsize: Tuple[float, float],
        sharex: bool,
        sharey: bool,
        figTitle: Optional[str],
        backendOptions: Optional[dict],
    ) -> Any:
        import pyvista as pv  # noqa: PLC0415

        opts = backendOptions or {}
        requested_backend = opts.get("jupyter_backend", "static")
        notebook_fallback = opts.get("notebook_fallback_backend", "trame")
        # Native pop-out windows from notebook kernels are often unreliable
        # (especially in VS Code). Fall back to a notebook-safe backend.
        if requested_backend == "none" and _in_ipykernel():
            self._jupyter_backend = notebook_fallback
        else:
            self._jupyter_backend = requested_backend
        self._point_size = float(opts.get("point_size", 8.0))
        self._shown = False
        self._display_handle = None  # reset so new figure gets its own output cell anchor
        self._trame_view = None
        # Note: do NOT call pv.set_jupyter_backend() here — it’s a global
        # setting and interferes with other plotters.  We pass jupyter_backend
        # directly to plotter.show() instead.

        shape, label_to_pos = _parse_mosaic(mosaic)
        self._axes = label_to_pos

        # Offscreen rendering is required for static image capture;
        # interactive modes need on-screen (off_screen=False).
        off_screen = self._jupyter_backend == "static"

        self._plotter = pv.Plotter(
            shape=shape,
            window_size=[int(figsize[0] * 96), int(figsize[1] * 96)],
            off_screen=off_screen,
            title=figTitle or "warpSPHPlotting",
        )
        return self._plotter

    def get_axes(self) -> Dict[str, Tuple[int, int]]:
        return self._axes

    # -------------------------------------------------------------------------
    # Per-panel rendering
    # -------------------------------------------------------------------------

    def render_panel(
        self,
        panel_key: str,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: PlottingOptions,
    ) -> PyVistaVisualizationState:
        import pyvista as pv  # noqa: PLC0415

        from ._render_util import prepare_particle_states  # noqa: PLC0415
        from ..math import getBounds  # noqa: PLC0415

        row, col = self._axes[panel_key]
        self._plotter.subplot(row, col)

        # ── preprocessing ────────────────────────────────────────────────────
        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = prepare_particle_states(
            particleState, domain, quantity, options
        )

        # ── camera & title ────────────────────────────────────────────────────
        # NOTE: view_xy() is deferred to after all meshes are added so that
        # reset_camera() (called internally) has actual scene bounds to fit.
        if opts.plotTitle is not None:
            self._plotter.add_text(
                opts.plotTitle,
                position='upper_edge',
                font_size=9,
                color='black',
                name=f'title_{panel_key}',
            )

        # ── domain bounding box ───────────────────────────────────────────────
        if opts.plotDomain:
            self._plotter.add_mesh(
                _make_domain_wireframe(domain_),
                style="wireframe",
                color="blue",
                line_width=1,
                name=f"domain_{panel_key}",
            )

        # ── grid or scatter path ───────────────────────────────────────────────
        mesh_fluid, actor_fluid, fluid_range = None, None, None
        mesh_boundary, actor_boundary, boundary_range = None, None, None
        # One norm across fluid AND boundary -- see `math.getSharedBounds`.
        # Normalising them separately and drawing both through the same colormap
        # makes wall colours incomparable to fluid colours, and the single
        # scalar bar can only describe one of the two.
        fluidQs, boundaryQs, sharedNorm = _sharedMarkerBounds(fluid, boundary, opts)
        mesh_grid, actor_grid, grid_range = None, None, None

        if opts.gridVisualization is not None:
            # ── grid (ImageData) ──────────────────────────────────────────────
            from ..grid import mapToGrid  # noqa: PLC0415

            gridState, gridQuantity, nxs, gridExtent = mapToGrid(
                particleState=particleState,
                quantity=rotated_q,
                domain=domain_,
                nx=opts.gridVisualization.resolution,
                targetNeighbors=50,
                kernel=opts.plottingKernel,
                alignment="center",
                includeFluid=opts.fluidVisualization != VisualizeOptions.Hide,
                includeBoundary=opts.boundaryVisualization != VisualizeOptions.Hide,
                gridMode=opts.gridVisualization.gridSupport,
            )

            mesh_grid, grid_range = _grid_to_image_data(gridState, gridQuantity, nxs, gridExtent, opts)
            actor_grid = self._plotter.add_mesh(
                mesh_grid,
                scalars="quantity",
                cmap=_cmap_name(opts),
                clim=grid_range,
                show_scalar_bar=opts.showColorBar,
                scalar_bar_args={"title": panel_key},
                name=f"grid_{panel_key}",
            )

        else:
            point_size = _point_size(opts, self._point_size)
            # ── fluid point cloud ─────────────────────────────────────────────
            if opts.fluidVisualization != VisualizeOptions.Hide and fluid.positions.shape[0] > 0:
                pts = _to_3d(fluid.positions)
                mesh_fluid = pv.PolyData(pts)

                if opts.fluidVisualization == VisualizeOptions.Passive:
                    actor_fluid = self._plotter.add_mesh(
                        mesh_fluid,
                        style="points",
                        point_size=point_size,
                        color="gray",
                        opacity=0.5,
                        name=f"fluid_{panel_key}",
                    )
                else:
                    fluid_range = _scalar_range(fluidQs, sharedNorm)
                    mesh_fluid["quantity"] = fluidQs
                    actor_fluid = self._plotter.add_mesh(
                        mesh_fluid,
                        style="points",
                        point_size=point_size,
                        render_points_as_spheres=False,
                        scalars="quantity",
                        cmap=_cmap_name(opts),
                        clim=fluid_range,
                        show_scalar_bar=opts.showColorBar,
                        scalar_bar_args={"title": panel_key},
                        name=f"fluid_{panel_key}",
                    )

            # ── boundary point cloud ──────────────────────────────────────────
            if opts.boundaryVisualization != VisualizeOptions.Hide and boundary.positions.shape[0] > 0:
                pts = _to_3d(boundary.positions)
                mesh_boundary = pv.PolyData(pts)

                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    actor_boundary = self._plotter.add_mesh(
                        mesh_boundary,
                        style="points",
                        point_size=point_size,
                        color="gray",
                        opacity=0.5,
                        name=f"boundary_{panel_key}",
                    )
                else:
                    boundary_range = _scalar_range(boundaryQs, sharedNorm)
                    mesh_boundary["quantity"] = boundaryQs
                    actor_boundary = self._plotter.add_mesh(
                        mesh_boundary,
                        style="points",
                        point_size=point_size,
                        render_points_as_spheres=False,
                        scalars="quantity",
                        cmap=_cmap_name(opts),
                        clim=boundary_range,
                        show_scalar_bar=False,   # avoid duplicate scalar bars
                        name=f"boundary_{panel_key}",
                    )

        # ── axis bounds / labels ──────────────────────────────────────────────
        self._plotter.show_bounds(
            show_zaxis=(domain_.dim > 2),
            show_zlabels=(domain_.dim > 2),
            xlabel='X',
            ylabel='Y',
            font_size=8,
            location='outer',
            ticks='both',
            grid=False,
        )

        # ── camera fit ────────────────────────────────────────────────────────
        # Called after all actors are in the scene so reset_camera() has
        # valid bounds to zoom to.
        if domain_.dim == 2:
            self._plotter.view_xy()
        self._plotter.reset_camera()

        return PyVistaVisualizationState(
            plotter=self._plotter,
            row=row,
            col=col,
            mesh_fluid=mesh_fluid,
            actor_fluid=actor_fluid,
            mesh_boundary=mesh_boundary,
            actor_boundary=actor_boundary,
            mesh_grid=mesh_grid,
            actor_grid=actor_grid,
            domain=domain_,
            options=opts,
            fluidParticles=fluid,
            boundaryParticles=boundary,
            ghostParticles=ghost,
            assembledQuantity=assembled,
            rotatedQuantities=rotated_q,
            fluid_scalar_range=fluid_range,
            boundary_scalar_range=boundary_range,
            grid_scalar_range=grid_range,
        )

    def update_panel(
        self,
        panel_key: str,
        panel_state: PyVistaVisualizationState,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: PlottingOptions,
        **kwargs: Any,
    ) -> PyVistaVisualizationState:
        import copy as _copy  # noqa: PLC0415
        from ._render_util import prepare_particle_states  # noqa: PLC0415
        from ..math import getBounds  # noqa: PLC0415
        import pyvista as pv  # noqa: PLC0415

        # Apply any option overrides supplied via newOptions kwargs so that
        # colormap changes etc. are reflected in this update.
        if kwargs:
            options = _copy.deepcopy(options)
            for k, v in kwargs.items():
                if hasattr(options, k):
                    setattr(options, k, v)

        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = prepare_particle_states(
            particleState, domain, quantity, options
        )

        colormap_changed = (
            panel_state.options.colorMap != opts.colorMap
            or panel_state.options.flipColorMap != opts.flipColorMap
            or panel_state.options.showColorBar != opts.showColorBar
        )

        self._plotter.subplot(panel_state.row, panel_state.col)

        mesh_fluid = panel_state.mesh_fluid
        actor_fluid = panel_state.actor_fluid
        # One norm across fluid AND boundary -- see `math.getSharedBounds`.
        fluidQs, boundaryQs, sharedNorm = _sharedMarkerBounds(fluid, boundary, opts)

        fluid_range = panel_state.fluid_scalar_range
        mesh_boundary = panel_state.mesh_boundary
        actor_boundary = panel_state.actor_boundary
        boundary_range = panel_state.boundary_scalar_range
        mesh_grid = panel_state.mesh_grid
        actor_grid = panel_state.actor_grid
        grid_range = panel_state.grid_scalar_range

        if opts.gridVisualization is not None:
            # ── grid update: replace actor via same name ──────────────────────
            from ..grid import mapToGrid  # noqa: PLC0415

            gridState, gridQuantity, nxs, gridExtent = mapToGrid(
                particleState=particleState,
                quantity=rotated_q,
                domain=domain_,
                nx=opts.gridVisualization.resolution,
                targetNeighbors=50,
                kernel=opts.plottingKernel,
                alignment="center",
                includeFluid=opts.fluidVisualization != VisualizeOptions.Hide,
                includeBoundary=opts.boundaryVisualization != VisualizeOptions.Hide,
                gridMode=opts.gridVisualization.gridSupport,
            )

            mesh_grid, grid_range = _grid_to_image_data(
                gridState, gridQuantity, nxs, gridExtent, opts
            )
            if panel_state.mesh_grid is not None and panel_state.actor_grid is not None:
                # Fast path: keep actor alive and only update scalar field.
                panel_state.mesh_grid.point_data["quantity"] = mesh_grid.point_data["quantity"]
                mesh_grid = panel_state.mesh_grid
                actor_grid = panel_state.actor_grid
                try:
                    actor_grid.mapper.scalar_range = grid_range
                except Exception:
                    pass
                if colormap_changed:
                    actor_grid = self._plotter.add_mesh(
                        mesh_grid,
                        scalars="quantity",
                        cmap=_cmap_name(opts),
                        clim=grid_range,
                        show_scalar_bar=opts.showColorBar,
                        scalar_bar_args={"title": panel_key},
                        name=f"grid_{panel_key}",
                    )
            else:
                # Slow path: initial create or after mode switch.
                actor_grid = self._plotter.add_mesh(
                    mesh_grid,
                    scalars="quantity",
                    cmap=_cmap_name(opts),
                    clim=grid_range,
                    show_scalar_bar=opts.showColorBar,
                    scalar_bar_args={"title": panel_key},
                    name=f"grid_{panel_key}",
                )

        else:
            point_size = _point_size(opts, self._point_size)
            # ── scatter update: mutate existing meshes when possible ──────────
            if fluid.positions.shape[0] > 0:
                pts = _to_3d(fluid.positions)
                if opts.fluidVisualization == VisualizeOptions.Passive:
                    if panel_state.mesh_fluid is not None and panel_state.actor_fluid is not None:
                        mesh_fluid = panel_state.mesh_fluid
                        actor_fluid = panel_state.actor_fluid
                        mesh_fluid.points = pts
                        try:
                            actor_fluid.prop.point_size = point_size
                        except Exception:
                            pass
                    else:
                        mesh_fluid = pv.PolyData(pts)
                        actor_fluid = self._plotter.add_mesh(
                            mesh_fluid,
                            style="points",
                            point_size=point_size,
                            color="gray",
                            opacity=0.5,
                            name=f"fluid_{panel_key}",
                        )
                elif opts.fluidVisualization != VisualizeOptions.Hide:
                    q = fluidQs
                    fluid_range = _scalar_range(fluidQs, sharedNorm)
                    if panel_state.mesh_fluid is not None and panel_state.actor_fluid is not None:
                        mesh_fluid = panel_state.mesh_fluid
                        actor_fluid = panel_state.actor_fluid
                        mesh_fluid.points = pts
                        mesh_fluid.point_data["quantity"] = q
                        try:
                            actor_fluid.mapper.scalar_range = fluid_range
                        except Exception:
                            pass
                        try:
                            actor_fluid.prop.point_size = point_size
                        except Exception:
                            pass
                        if colormap_changed:
                            actor_fluid = self._plotter.add_mesh(
                                mesh_fluid,
                                style="points",
                                point_size=point_size,
                                render_points_as_spheres=False,
                                scalars="quantity",
                                cmap=_cmap_name(opts),
                                clim=fluid_range,
                                show_scalar_bar=opts.showColorBar,
                                scalar_bar_args={"title": panel_key},
                                name=f"fluid_{panel_key}",
                            )
                    else:
                        mesh_fluid = pv.PolyData(pts)
                        mesh_fluid["quantity"] = q
                        actor_fluid = self._plotter.add_mesh(
                            mesh_fluid,
                            style="points",
                            point_size=point_size,
                            render_points_as_spheres=False,
                            scalars="quantity",
                            cmap=_cmap_name(opts),
                            clim=fluid_range,
                            show_scalar_bar=opts.showColorBar,
                            scalar_bar_args={"title": panel_key},
                            name=f"fluid_{panel_key}",
                        )
                else:
                    mesh_fluid = None
                    actor_fluid = None

            if boundary.positions.shape[0] > 0:
                pts = _to_3d(boundary.positions)
                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    if panel_state.mesh_boundary is not None and panel_state.actor_boundary is not None:
                        mesh_boundary = panel_state.mesh_boundary
                        actor_boundary = panel_state.actor_boundary
                        mesh_boundary.points = pts
                        try:
                            actor_boundary.prop.point_size = point_size
                        except Exception:
                            pass
                    else:
                        mesh_boundary = pv.PolyData(pts)
                        actor_boundary = self._plotter.add_mesh(
                            mesh_boundary,
                            style="points",
                            point_size=point_size,
                            color="gray",
                            opacity=0.5,
                            name=f"boundary_{panel_key}",
                        )
                elif opts.boundaryVisualization != VisualizeOptions.Hide:
                    q = boundaryQs
                    boundary_range = _scalar_range(boundaryQs, sharedNorm)
                    if panel_state.mesh_boundary is not None and panel_state.actor_boundary is not None:
                        mesh_boundary = panel_state.mesh_boundary
                        actor_boundary = panel_state.actor_boundary
                        mesh_boundary.points = pts
                        mesh_boundary.point_data["quantity"] = q
                        try:
                            actor_boundary.mapper.scalar_range = boundary_range
                        except Exception:
                            pass
                        try:
                            actor_boundary.prop.point_size = point_size
                        except Exception:
                            pass
                        if colormap_changed:
                            actor_boundary = self._plotter.add_mesh(
                                mesh_boundary,
                                style="points",
                                point_size=point_size,
                                render_points_as_spheres=False,
                                scalars="quantity",
                                cmap=_cmap_name(opts),
                                clim=boundary_range,
                                show_scalar_bar=False,
                                name=f"boundary_{panel_key}",
                            )
                    else:
                        mesh_boundary = pv.PolyData(pts)
                        mesh_boundary["quantity"] = q
                        actor_boundary = self._plotter.add_mesh(
                            mesh_boundary,
                            style="points",
                            point_size=point_size,
                            render_points_as_spheres=False,
                            scalars="quantity",
                            cmap=_cmap_name(opts),
                            clim=boundary_range,
                            show_scalar_bar=False,
                            name=f"boundary_{panel_key}",
                        )
                else:
                    mesh_boundary = None
                    actor_boundary = None


        panel_state.fluid_scalar_range = fluid_range
        panel_state.mesh_fluid = mesh_fluid
        panel_state.actor_fluid = actor_fluid
        panel_state.mesh_boundary = mesh_boundary
        panel_state.actor_boundary = actor_boundary
        panel_state.boundary_scalar_range = boundary_range
        panel_state.mesh_grid = mesh_grid
        panel_state.actor_grid = actor_grid
        panel_state.grid_scalar_range = grid_range
        panel_state.fluidParticles = fluid
        panel_state.boundaryParticles = boundary
        panel_state.ghostParticles = ghost
        panel_state.assembledQuantity = assembled
        panel_state.rotatedQuantities = rotated_q
        panel_state.options = opts

        return panel_state

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export(self, filepath: str, **kwargs: Any) -> None:
        """Export the scene to *filepath*.

        The format is inferred from the file extension:

        * **Raster** (``png``, ``jpg``, ``jpeg``, ``bmp``, ``tif``, ``tiff``):
          uses :meth:`pyvista.Plotter.screenshot`.  Pass
          ``transparent_background=True`` to get an alpha channel (PNG only).
        * **Vector** (``svg``, ``eps``, ``ps``, ``pdf``):
          uses :meth:`pyvista.Plotter.save_graphic`.

        Note
        ----
        PyVista is a pixel-based renderer; ``dpi`` has no meaning here and is
        silently ignored.  Use the ``window_size`` option when creating the
        figure to control the output resolution instead.
        """
        if self._plotter is None:
            raise RuntimeError("No plotter to export; call create_figure first.")
        import os
        ext = os.path.splitext(filepath)[1].lower().lstrip(".")
        vector_exts = {"svg", "eps", "ps", "pdf"}
        kwargs.pop("dpi", None)  # not meaningful for a pixel-based renderer
        if ext in vector_exts:
            allowed = {"title", "raster", "painter"}
            self._plotter.save_graphic(filepath, **{k: v for k, v in kwargs.items() if k in allowed})
        else:
            allowed = {"window_size", "transparent_background", "return_img"}
            self._plotter.screenshot(filepath, **{k: v for k, v in kwargs.items() if k in allowed})

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def show(self) -> None:
        """Flush / display the scene.

        Behaviour depends on *jupyter_backend*:

        ``"static"`` (default)
            Renders offscreen, captures a screenshot, and displays it as an
            inline PNG in the current Jupyter cell.  Safe to call repeatedly;
            each call produces a fresh image reflecting the latest data.
        ``"trame"``
            The first call attaches an interactive widget to the notebook cell.
            Subsequent calls call ``plotter.render()`` to push data changes to
            the live widget.
        ``"none"`` or anything else
            Opens a native pop-out window on the first call; subsequent calls
            are no-ops.
        """
        if self._plotter is None:
            return

        if self._jupyter_backend == "static":
            # Render to the offscreen buffer and capture a screenshot.
            # On first call, create an IPython display handle tied to this
            # output cell.  On subsequent calls, call handle.update() so the
            # image is replaced in-place rather than appended as a new output.
            try:
                self._plotter.render()
                img = np.ascontiguousarray(self._plotter.screenshot(return_img=True))
            except KeyboardInterrupt:
                return
            try:
                from IPython.display import display, Image as IPImage  # noqa: PLC0415
                import io, PIL.Image  # noqa: PLC0415

                buf = io.BytesIO()
                PIL.Image.fromarray(img).save(buf, format="PNG")
                ipy_img = IPImage(data=buf.getvalue())

                if self._display_handle is None:
                    # display_id=True returns a DisplayHandle whose .update()
                    # method replaces this output in-place in the notebook.
                    self._display_handle = display(ipy_img, display_id=True)
                else:
                    self._display_handle.update(ipy_img)
            except ImportError:
                pass  # Non-notebook environment — screenshot captured but not shown.
        elif self._jupyter_backend == "trame":
            import pyvista as pv  # noqa: PLC0415

            if not self._shown:
                try:
                    pv.set_jupyter_backend("trame")
                    # Keep the trame viewer open so subsequent updateQuantities
                    # calls can push incremental frames into the same widget.
                    self._trame_view = self._plotter.show(
                        jupyter_backend="trame",
                        auto_close=False,
                    )
                    self._shown = True
                except Exception:
                    # trame not installed; fall back to static screenshot.
                    self._jupyter_backend = "static"
                    self.show()
            else:
                # render() pushes scene changes into the active trame view.
                try:
                    self._plotter.render()
                except KeyboardInterrupt:
                    return

                # If available, nudge the trame-side view to push a fresh
                # frame immediately rather than waiting for notebook idle.
                try:
                    if self._trame_view is not None:
                        updater = getattr(self._trame_view, "update", None)
                        if callable(updater):
                            updater()
                except Exception:
                    pass
        else:
            # Pop-out native window.
            # Pass jupyter_backend='none' directly to show() — this overrides
            # any global PyVista Jupyter setting without side-effects on other
            # plotters.  interactive_update=True returns immediately after
            # opening the window.  plotter.update() is then called to flush
            # the render queue and make the window visible on WSLg.
            if not self._shown:
                self._plotter.show(
                    jupyter_backend="none",
                    interactive_update=True,
                    auto_close=False,
                )
                self._plotter.update()  # flush so WSLg actually paints the window
                self._shown = True
            else:
                self._plotter.update()

    # -------------------------------------------------------------------------
    # Capability flags
    # -------------------------------------------------------------------------

    @property
    def supports_grid(self) -> bool:
        return True

    @property
    def supports_streamlines(self) -> bool:
        # Phase 3: VTK streamline filter deferred to a follow-up
        return False

    @property
    def supports_notebook_inline(self) -> bool:
        return True  # via static screenshot or trame widget

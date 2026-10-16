"""PyVista rendering with physical scalar arrays and persistent scene objects.

`backendOptions` accepts `jupyter_backend` (static/trame/none), `off_screen`,
`window_size`, `point_size`, and `scalar_bar` styling. Tick text is positioned
independently so nonlinear physical labels stay beside the color strip.
"""

from dataclasses import dataclass, field
from typing import Any
import copy
import threading
import numpy as np
import torch
from .base import AbstractBackend
from ._render_util import prepare_frame, view_needs_refit, view_domain
from ..enumTypes import VisualizeOptions
from .._sizing import particle_diameter, explicit_pixel_size
from ..math import getSharedBounds
from ._mosaic import parse_mosaic
from ._trame_context import register_trame_window, serialized_notebook_render


@dataclass
class PyVistaVisualizationState:
    plotter: Any
    row: int
    col: int
    domain: Any
    options: Any
    fluidParticles: Any
    boundaryParticles: Any
    ghostParticles: Any
    assembledQuantity: Any
    rotatedQuantities: Any
    mesh_fluid: Any = None
    actor_fluid: Any = None
    mesh_boundary: Any = None
    actor_boundary: Any = None
    mesh_grid: Any = None
    actor_grid: Any = None
    actors: dict = field(default_factory=dict)
    colorbar: Any = None
    labels: list = field(default_factory=list)
    normSignature: Any = None
    lookup: Any = None
    gridGeometry: Any = None
    title: Any = None
    particleDiameters: dict = field(default_factory=dict)


def _to_3d(positions):
    pts = positions.detach().cpu().numpy()
    return np.column_stack((pts, np.zeros(len(pts)))) if pts.shape[1] == 2 else pts


def _lookup(name):
    """A normalized LUT keeps per-point GPU mapping constant-time.

    Meshes retain raw physical quantities in a separate array. Display arrays
    contain norm(q), and inverse tick labels describe the same transform.
    """
    import pyvista as pv

    return pv.LookupTable(
        cmap=name, scalar_range=(0.0, 1.0), nan_color=(0.5, 0.5, 0.5, 0.0)
    )


class PyVistaBackend(AbstractBackend):
    def __init__(self):
        self._render_lock = threading.RLock()
        self._trame_context_guard = None
        self._plotter = None
        self._axes = {}
        self._shown = False
        self._display_handle = None
        self._states = {}
        self._title_renderer = None
        self._title_actor = None
        self._title_lines = -1
        self._title_layout = None
        self._title_text = ""
        self._streamline_multisamples = None

    def create_figure(self, mosaic, figsize, sharex, sharey, figTitle, backendOptions):
        import pyvista as pv

        if bool(sharex) != bool(sharey):
            raise ValueError(
                "PyVista links full cameras; set sharex and sharey both True or both False."
            )
        opts = backendOptions or {}
        allowed = {
            "jupyter_backend",
            "off_screen",
            "window_size",
            "point_size",
            "scalar_bar",
            "notebook_fallback_backend",
            "jupyter_kwargs",
        }
        unknown = opts.keys() - allowed
        if unknown:
            raise TypeError(f"Unknown PyVista backend options: {sorted(unknown)}")
        self._opts = opts
        self._jupyter_backend = opts.get("jupyter_backend", "static")
        if self._jupyter_backend not in ("static", "trame", "none"):
            raise ValueError("jupyter_backend must be static, trame, or none.")
        if (
            self._jupyter_backend == "trame"
            and opts.get("jupyter_kwargs", {}).get("mode", "server") != "server"
        ):
            raise ValueError(
                "PyVista notebook plots require server rendering to preserve "
                "physical colorbar labels and axes. Set jupyter_kwargs mode='server'."
            )
        shape, axes, spans = parse_mosaic(mosaic)
        # vtk renderer viewports allow rectangular spans without duplicate panels.
        self._axes = axes
        self._spans, self._shape = spans, shape
        self._plotter = pv.Plotter(
            shape=shape,
            off_screen=opts.get("off_screen", self._jupyter_backend == "static"),
            window_size=opts.get(
                "window_size", (int(figsize[0] * 96), int(figsize[1] * 96))
            ),
            title=figTitle or "warpSPHPlotting",
        )
        if self._plotter.off_screen:
            # Static/offscreen captures need one readable front buffer. A
            # double-buffered GLX context can wait for display synchronization
            # on every capture (one second on the tested NVIDIA host).
            # Single buffering avoids that wait while retaining valid pixels;
            # disabling SwapBuffers alone instead leaves screenshots black.
            self._plotter.render_window.DoubleBufferOff()
        for key, (r, c) in axes.items():
            self._plotter.subplot(r, c)
            r0, c0, r1, c1 = spans[key]
            self._plotter.renderer.SetViewport(
                c0 / shape[1],
                1 - (r1 + 1) / shape[0],
                (c1 + 1) / shape[1],
                1 - r0 / shape[0],
            )
        occupied = set(axes.values())
        covered = {
            (r, c)
            for r0, c0, r1, c1 in spans.values()
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
        }
        for r in range(shape[0]):
            for c in range(shape[1]):
                if (r, c) in covered and (r, c) not in occupied:
                    self._plotter.subplot(r, c)
                    self._plotter.renderer.SetDraw(False)
        self._plotter.set_background("white", all_renderers=True)
        # VTK has one camera object for x/y/z; link only when both axes are shared.
        self._share_camera = bool(sharex and sharey)
        self._camera_groups = None
        self._interaction_dimension = None
        self.update_figure_title(figTitle)
        self._layout_observer = self._plotter.render_window.AddObserver(
            "StartEvent", self._refresh_layout
        )
        return self._plotter

    def get_axes(self):
        return self._axes

    @serialized_notebook_render
    def render_panel(
        self, key, particleState, domain, quantity, options, _preparedFrame=None
    ):
        f = _preparedFrame or prepare_frame(particleState, domain, quantity, options)
        r, c = self._axes[key]
        s = PyVistaVisualizationState(
            self._plotter,
            r,
            c,
            f.domain,
            f.options,
            f.fluid,
            f.boundary,
            f.ghost,
            f.assembled,
            f.particles.quantities,
        )
        self._states[key] = s
        return self._render(key, s, f, initial=True)

    @serialized_notebook_render
    def update_panel(
        self, key, panel_state, particleState, domain, quantity, options, **kwargs
    ):
        f = kwargs.pop("_preparedFrame", None)
        opts = copy.deepcopy(options)
        for k, v in kwargs.items():
            if not hasattr(opts, k):
                raise TypeError(f"Unknown plotting option: {k}")
            setattr(opts, k, v)
        return self._render(
            key, panel_state, f or prepare_frame(particleState, domain, quantity, opts)
        )

    def _remove(self, s, name):
        actor = s.actors.pop(name, None)
        if actor is not None:
            self._plotter.remove_actor(actor, reset_camera=False, render=False)

    def _group_cameras(self, key, s, f):
        # A volume and a planar slice need different camera orientations.
        # Rebuild links only when panel coordinate systems change, preserving
        # each camera before detaching it from a formerly compatible group.
        import pyvista as pv

        s.cameraGroup = (f.domain.dim, tuple(f.displayAxes or range(f.domain.dim)))
        signature = tuple((k, state.cameraGroup) for k, state in self._states.items())
        if not self._share_camera or signature == self._camera_groups:
            return
        self._camera_groups = signature
        groups = {}
        for panel_key, state in self._states.items():
            renderer = self._plotter.renderers[state.row * self._shape[1] + state.col]
            camera = pv.Camera()
            camera.DeepCopy(renderer.GetActiveCamera())
            renderer.camera = camera
            group = groups.setdefault(state.cameraGroup, [])
            group.append(state.row * self._shape[1] + state.col)
        for indices in groups.values():
            if len(indices) > 1:
                self._plotter.link_views(indices)

    def _mesh(self, s, name, mesh, passive, opts, lookup):
        previous = s.actors.get(name)
        if previous is not None:
            # Reuse actors; replace input data when topology changes.
            previous.mapper.dataset = mesh
            actor = previous
        else:
            kw = dict(
                name=name,
                show_scalar_bar=False,
                reset_camera=False,
                render=False,
                lighting=False,
                point_size=self._point_size(opts),
                style="points" if "grid" not in name else "surface",
            )
            actor = self._plotter.add_mesh(
                mesh,
                color="gray" if passive else None,
                scalars=None if passive else "display_quantity",
                **kw,
            )
            s.actors[name] = actor
        actor.prop.point_size = self._point_size(opts)
        actor.prop.lighting = False
        actor.prop.render_points_as_spheres = s.cameraGroup[0] == 3
        actor.prop.opacity = 0.5 if passive else 1.0
        # A transparent NaN LUT entry makes VTK classify even fully valid
        # point clouds as translucent. Its OIT pass blends front/back particle
        # colors instead of occluding them. Active scatter has only finite
        # vertices (filtered below), so it belongs in the opaque depth pass.
        actor.SetForceOpaque(not passive and "grid" not in name)
        actor.mapper.scalar_visibility = not passive
        if passive:
            actor.prop.color = "gray"
        else:
            actor.mapper.array_name = "display_quantity"
            actor.mapper.SetLookupTable(lookup)
            actor.mapper.scalar_range = lookup.GetRange()
        return actor

    def _point_size(self, opts):
        if opts.markerSize is not None:
            return explicit_pixel_size(opts.markerSize)
        return float(self._opts.get("point_size", 1.0))

    def _volume(self, key, s, mesh, opts, cmap, domain):
        """Ray-cast the interior, reserving scalar zero for uncovered samples.

        Physical quantities and normalized display values stay on the mesh.
        The additional byte array is only a volume transfer-function input;
        covered physical zeros map to a nontransparent entry, just like all
        other valid samples. Opacity distance uses world units so changing the
        interpolation resolution does not also change the apparent density.
        """
        import pyvista as pv
        from matplotlib import colormaps

        values = np.asarray(mesh.point_data["display_quantity"])
        valid = np.isfinite(values)
        encoded = np.zeros(values.shape, dtype=np.uint8)
        encoded[valid] = 1 + np.rint(np.clip(values[valid], 0, 1) * 254).astype(
            np.uint8
        )
        mesh.point_data["volume_quantity"] = encoded
        mesh.set_active_scalars("volume_quantity")
        signature = (cmap, opts.gridVisualization.volumeOpacity)
        if getattr(s, "volumeSignature", None) != signature:
            colors = np.zeros((256, 4), dtype=np.uint8)
            colors[1:] = colormaps[cmap](np.linspace(0, 1, 255), bytes=True)
            colors[1:, 3] = round(255 * opts.gridVisualization.volumeOpacity)
            s.volumeLookup = pv.LookupTable(values=colors, scalar_range=(0, 255))
            s.volumeSignature = signature
        distance = float(torch.linalg.vector_norm(domain.max - domain.min)) * 0.1
        name = f"grid_{key}"
        actor = s.actors.get(name)
        if actor is None:
            actor = self._plotter.add_volume(
                mesh,
                scalars="volume_quantity",
                cmap=s.volumeLookup,
                clim=(0, 255),
                opacity_unit_distance=distance,
                shade=False,
                show_scalar_bar=False,
                reset_camera=False,
                render=False,
                name=name,
            )
            s.actors[name] = actor
        else:
            actor.mapper.dataset = mesh
            actor.mapper.lookup_table = s.volumeLookup
            actor.prop.apply_lookup_table(s.volumeLookup)
            actor.prop.opacity_unit_distance = distance
        actor.prop.interpolation_type = "linear"
        return actor

    def _resize_particles(self, key, s):
        if s.options.markerSize is not None or "point_size" in self._opts:
            return
        renderer = self._plotter.renderer
        # Project at the focal plane. This is exact for the 2D parallel camera
        # and a representative depth for native 3D point clouds.
        camera = renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint())
        direction = np.asarray(camera.GetDirectionOfProjection())
        right = np.cross(direction, camera.GetViewUp())
        right /= np.linalg.norm(right)
        renderer.SetWorldPoint(*focal, 1)
        renderer.WorldToDisplay()
        origin = np.asarray(renderer.GetDisplayPoint())
        renderer.SetWorldPoint(*(focal + right), 1)
        renderer.WorldToDisplay()
        pixels_per_unit = np.linalg.norm(
            np.asarray(renderer.GetDisplayPoint()) - origin
        )
        for kind, diameter in s.particleDiameters.items():
            actor = s.actors.get(f"{kind}_{key}")
            if actor is not None:
                actor.prop.point_size = max(0.1, diameter * pixels_per_unit)

    def _render(self, key, s, f, initial=False):
        import pyvista as pv
        from ..grid import mapFrameToGrid, gridDisplayBounds

        self._plotter.subplot(s.row, s.col)
        opts = f.options
        dimension_changed = initial or s.domain.dim != f.domain.dim
        self._group_cameras(key, s, f)
        changed = initial or view_needs_refit(
            s.domain, f, getattr(s, "displayAxes", None)
        )
        cmap = opts.colorMap.value + ("_r" if opts.flipColorMap else "")
        fq, bq, norm = f.sharedBounds or getSharedBounds(
            (
                f.fluid.quantities
                if opts.fluidVisualization == VisualizeOptions.Visualize
                else None
            ),
            (
                f.boundary.quantities
                if opts.boundaryVisualization == VisualizeOptions.Visualize
                else None
            ),
            opts,
        )
        if opts.gridVisualization:
            gs, gq, n, ext = mapFrameToGrid(f, f.particles.quantities)
            qs, norm = gridDisplayBounds(gq, ext, opts)
        signature = (
            None
            if norm is None
            else (
                cmap,
                type(norm).__name__,
                float(norm.vmin),
                float(norm.vmax),
                opts.quantityLogThreshold,
            )
        )
        if (norm is not None or opts.gridVisualization is not None) and (
            s.lookup is None or getattr(s, "_lookupCmap", None) != cmap
        ):
            s.lookup = _lookup(cmap)
            s._lookupCmap = cmap
        s.normSignature = signature
        s.norm = norm
        if opts.gridVisualization:
            self._remove(s, f"fluid_{key}")
            self._remove(s, f"boundary_{key}")
            s.mesh_fluid = s.actor_fluid = s.mesh_boundary = s.actor_boundary = None
            dims = tuple(n) + ((1,) if len(n) == 2 else ())
            pts = gs.positions.detach().cpu().numpy()
            spacing = tuple(
                float((ext["max"][i] - ext["min"][i]) / n[i]) for i in range(len(n))
            ) + ((1.0,) if len(n) == 2 else ())
            origin = tuple(pts[0]) + ((0.0,) if len(n) == 2 else ())
            geom = (dims, spacing, origin)
            mesh = (
                s.mesh_grid
                if s.mesh_grid is not None and geom == s.gridGeometry
                else pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
            )
            mesh.point_data["quantity"] = (
                gq.detach().cpu().numpy().reshape(n).ravel(order="F")
            )
            mesh.point_data["display_quantity"] = (
                np.ma.filled(norm(qs) if norm is not None else qs, np.nan)
                .reshape(n)
                .ravel(order="F")
            )
            s.mesh_grid = mesh
            s.gridGeometry = geom
            mode = opts.gridVisualization.renderMode if f.domain.dim == 3 else "surface"
            if mode != getattr(s, "gridMode", None):
                self._remove(s, f"grid_{key}")
            s.gridMode = mode
            s.actor_grid = (
                self._volume(key, s, mesh, opts, cmap, f.domain)
                if mode == "volume"
                else self._mesh(s, f"grid_{key}", mesh, False, opts, s.lookup)
            )
        else:
            self._remove(s, f"grid_{key}")
            s.mesh_grid = s.actor_grid = None
            for kind, pop, variant, values in [
                ("fluid", f.fluid, opts.fluidVisualization, fq),
                ("boundary", f.boundary, opts.boundaryVisualization, bq),
            ]:
                name = f"{kind}_{key}"
                mesh = getattr(s, f"mesh_{kind}")
                if variant == VisualizeOptions.Hide or not len(pop.positions):
                    self._remove(s, name)
                    mesh = actor = None
                else:
                    pts = _to_3d(pop.positions)
                    if mesh is None or mesh.n_points != len(pts):
                        mesh = pv.PolyData(pts)
                    else:
                        mesh.points = pts
                    mesh.point_data["quantity"] = pop.quantities.detach().cpu().numpy()
                    if values is not None:
                        mesh.point_data["display_quantity"] = np.ma.filled(
                            norm(values), np.nan
                        )
                    passive = variant == VisualizeOptions.Passive
                    valid = (
                        np.ones(len(pts), dtype=bool)
                        if passive
                        else np.isfinite(mesh.point_data["display_quantity"])
                    )
                    if not valid.all() or mesh.n_verts != len(pts):
                        ids = np.flatnonzero(valid)
                        mesh.verts = np.column_stack(
                            (np.ones(len(ids), dtype=int), ids)
                        ).ravel()
                    actor = self._mesh(
                        s,
                        name,
                        mesh,
                        variant == VisualizeOptions.Passive,
                        opts,
                        s.lookup,
                    )
                setattr(s, f"mesh_{kind}", mesh)
                setattr(s, f"actor_{kind}", actor)
        if opts.plotDomain:
            mn = f.domain.min.cpu().numpy()
            mx = f.domain.max.cpu().numpy()
            if f.domain.dim == 2:
                z = float((mx - mn).max()) * 1e-4
                pts = np.array(
                    [
                        [mn[0], mn[1], z],
                        [mx[0], mn[1], z],
                        [mx[0], mx[1], z],
                        [mn[0], mx[1], z],
                        [mn[0], mn[1], z],
                    ]
                )
                box = pv.lines_from_points(pts)
            else:
                box = pv.Box(
                    bounds=np.column_stack((mn, mx)).ravel()
                ).extract_feature_edges()
            if changed or f"domain_{key}" not in s.actors:
                self._remove(s, f"domain_{key}")
                s.actors[f"domain_{key}"] = self._plotter.add_mesh(
                    box,
                    color="blue",
                    line_width=1,
                    name=f"domain_{key}",
                    reset_camera=False,
                    render=False,
                    lighting=False,
                )
        else:
            self._remove(s, f"domain_{key}")
        title = opts.plotTitle or ""
        if s.title is None:
            from vtkmodules.vtkRenderingCore import vtkTextActor

            s.title = vtkTextActor()
            s.title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            s.title.SetPosition(0.5, 0.96)
            prop = s.title.GetTextProperty()
            prop.SetFontFamilyToArial()
            prop.SetFontSize(
                max(
                    10,
                    round(13 * self._plotter.window_size[1] / (480 * self._shape[0])),
                )
            )
            prop.SetColor(0, 0, 0)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToTop()
            self._plotter.add_actor(
                s.title,
                name=f"title_{key}",
                reset_camera=False,
                pickable=False,
                render=False,
            )
            s.actors[f"title_{key}"] = s.title
        s.title.GetTextProperty().SetFontSize(
            max(10, round(13 * self._plotter.window_size[1] / (480 * self._shape[0])))
        )
        s.title.SetInput(title)
        if opts.gridVisualization and opts.gridVisualization.streamLines:
            if f.domain.dim != 2:
                raise ValueError(
                    "PyVista streamlines currently require a 2D vector grid."
                )
            from ..streamLines import streamlineSegments

            vg, vq, vn, ve = mapFrameToGrid(f, f.streamlineQuantity)
            segments = streamlineSegments(vg, vq, vn, ve)
            self._remove(s, f"stream_{key}")
            if len(segments):
                pts = np.column_stack(
                    (
                        segments.reshape(-1, 2),
                        np.full(
                            segments.size // 2,
                            float((f.domain.max - f.domain.min).max()) * 1e-4,
                        ),
                    )
                )
                lines = np.column_stack(
                    (
                        np.full(len(segments), 2),
                        np.arange(0, len(pts), 2),
                        np.arange(1, len(pts), 2),
                    )
                ).ravel()
                s.actors[f"stream_{key}"] = self._plotter.add_mesh(
                    pv.PolyData(pts, lines=lines),
                    color="black",
                    line_width=1,
                    lighting=False,
                    name=f"stream_{key}",
                    reset_camera=False,
                    render=False,
                )
        else:
            self._remove(s, f"stream_{key}")
        self._colorbar(key, s, norm, opts)
        display_axes = tuple(f.displayAxes or range(f.domain.dim))
        if changed or display_axes != getattr(s, "displayAxes", None):
            self._plotter.remove_bounds_axes()
            fit = view_domain(f)
            bounds = np.column_stack((fit.min.cpu(), fit.max.cpu())).ravel()
            if f.domain.dim == 2:
                # Keep the plotting frame separate from the physical domain
                # and the interpolated grid, including external wall particles.
                bounds = np.r_[bounds, 0.0, 0.0].astype(float)
                for actor in (s.actor_fluid, s.actor_boundary):
                    if actor is not None:
                        b = np.asarray(actor.bounds)
                        bounds[::2] = np.minimum(bounds[::2], b[::2])
                        bounds[1::2] = np.maximum(bounds[1::2], b[1::2])
                pad = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.035
                bounds[:4] += np.array([-pad, pad, -pad, pad])
                x0, x1, y0, y1 = bounds[:4]
                frame = pv.lines_from_points(
                    np.array(
                        [
                            [x0, y0, 0],
                            [x1, y0, 0],
                            [x1, y1, 0],
                            [x0, y1, 0],
                            [x0, y0, 0],
                        ]
                    )
                )
                self._remove(s, f"frame_{key}")
                s.actors[f"frame_{key}"] = self._plotter.add_mesh(
                    frame,
                    color=(0.35, 0.35, 0.35),
                    line_width=1,
                    lighting=False,
                    name=f"frame_{key}",
                    reset_camera=False,
                    render=False,
                )
            else:
                self._remove(s, f"frame_{key}")
            s.frameBounds = bounds
            s.boundsAxes = self._plotter.show_bounds(
                bounds=bounds,
                show_zaxis=f.domain.dim == 3,
                show_zlabels=f.domain.dim == 3,
                xtitle="XYZ"[(f.displayAxes or [0, 1])[0]],
                ytitle="XYZ"[(f.displayAxes or [0, 1])[1]],
                ztitle="Z",
                font_size=10,
                bold=False,
                fmt="%.3g",
                location="outer",
                ticks="both",
                grid=False,
                render=False,
            )
        s.displayAxes = display_axes
        has_bar = opts.showColorBar and norm is not None
        offset = (
            (0.0, -0.22)
            if getattr(s, "barOrientation", "vertical") == "horizontal"
            else (0.14, 0.0)
        )
        if changed:
            if f.domain.dim == 2:
                self._plotter.view_xy(render=False)
                self._plotter.enable_parallel_projection()
            elif dimension_changed:
                self._plotter.disable_parallel_projection()
                self._plotter.view_isometric(render=False)
            self._plotter.camera.SetWindowCenter(0.0, 0.0)
            self._plotter.reset_camera(
                bounds=s.frameBounds if opts.dynamicBounds else None, render=False
            )
            if has_bar:
                self._plotter.camera.zoom(0.85)
                self._plotter.camera.SetWindowCenter(*offset)
        s.particleDiameters = {
            kind: particle_diameter(particles, opts)
            for kind, particles, variant in (
                ("fluid", f.fluid, opts.fluidVisualization),
                ("boundary", f.boundary, opts.boundaryVisualization),
            )
            if opts.markerSize is None
            and "point_size" not in self._opts
            and opts.gridVisualization is None
            and variant != VisualizeOptions.Hide
        }
        s.domain = f.domain
        s.options = opts
        s.assembledQuantity = f.assembled
        s.rotatedQuantities = f.particles.quantities
        s.fluidParticles, s.boundaryParticles, s.ghostParticles = (
            f.fluid,
            f.boundary,
            f.ghost,
        )
        interaction_dimension = (
            3 if any(state.domain.dim == 3 for state in self._states.values()) else 2
        )
        if interaction_dimension != self._interaction_dimension:
            if interaction_dimension == 2:
                self._plotter.enable_2d_style()
            else:
                self._plotter.enable_trackball_style()
            self._interaction_dimension = interaction_dimension
        return s

    def _colorbar(self, key, s, norm, opts):
        from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
        from vtkmodules.vtkRenderingCore import vtkTextActor

        custom = self._opts.get("scalar_bar", {})
        horizontal = custom.get("orientation", "vertical") == "horizontal"
        style = dict(
            orientation="vertical",
            position=None,
            width=None,
            height=None,
            label_rotation=0.0,
            title_rotation=0.0,
            font_size=max(
                10, round(12 * self._plotter.window_size[1] / (480 * self._shape[0]))
            ),
            title_font_size=max(
                10, round(12 * self._plotter.window_size[1] / (480 * self._shape[0]))
            ),
            n_labels=3,
            label_format=".6g",
            title="",
        )
        if custom.keys() - style.keys():
            raise TypeError(
                f"Unknown scalar_bar options: {sorted(custom.keys()-style.keys())}"
            )
        style.update(custom)
        if int(style["n_labels"]) < 2:
            raise ValueError("scalar_bar.n_labels must be >= 2")
        if style["orientation"] not in ("horizontal", "vertical"):
            raise ValueError("scalar_bar.orientation must be horizontal or vertical")
        x, y = style["position"] or ((0.15, 0.10) if horizontal else (0.84, 0.15))
        width = (
            style["width"]
            if style["width"] is not None
            else (0.68 if horizontal else 0.035)
        )
        height = (
            style["height"]
            if style["height"] is not None
            else (0.035 if horizontal else 0.68)
        )
        if (
            not np.isfinite([x, y, width, height]).all()
            or min(x, y) < 0
            or min(width, height) <= 0
            or x + width > 1
            or y + height > 1
        ):
            raise ValueError(
                "scalar_bar position/size must fit inside normalized viewport bounds"
            )
        s.barOrientation = style["orientation"]
        if not opts.showColorBar or norm is None:
            for name in [n for n in s.actors if n.startswith(f"cb_{key}_")]:
                self._remove(s, name)
            s.colorbar = None
            s.labels = []
            s._barSignature = None
            return
        viewport_size = self._plotter.renderer.GetSize()
        frame_right = None
        if (
            not horizontal
            and style["position"] is None
            and s.domain.dim == 2
            and hasattr(s, "frameBounds")
        ):
            renderer = self._plotter.renderer
            x0, x1, y0, y1 = s.frameBounds[:4]
            edges = []
            for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                renderer.SetWorldPoint(float(px), float(py), 0.0, 1.0)
                renderer.WorldToDisplay()
                edges.append(renderer.GetDisplayPoint()[0] - renderer.GetOrigin()[0])
            frame_right = max(edges)
        signature = (
            s.normSignature,
            tuple(sorted(style.items())),
            viewport_size,
            frame_right,
        )
        if getattr(s, "_barSignature", None) == signature:
            return
        if s.colorbar is None:
            bar = vtkScalarBarActor()
            # VTK otherwise reserves internal text space even with labels off,
            # shrinking the strip away from the separate physical tick labels.
            bar.SetBarRatio(1.0)
            bar.SetTextPad(0)
            bar.SetTitleRatio(0.0)
            bar.SetMaximumNumberOfColors(256)
            bar.SetTitle("")
            bar.DrawTickLabelsOff()
            bar.DrawAnnotationsOff()
            self._plotter.add_actor(
                bar,
                name=f"cb_{key}_bar",
                reset_camera=False,
                pickable=False,
                render=False,
            )
            s.actors[f"cb_{key}_bar"] = bar
            s.colorbar = bar
        s.colorbar.SetLookupTable(s.lookup)
        (
            s.colorbar.SetOrientationToHorizontal()
            if horizontal
            else s.colorbar.SetOrientationToVertical()
        )
        count = int(style["n_labels"])
        if len(s.labels) != count + 1:
            for name in [n for n in s.actors if n.startswith(f"cb_{key}_text")]:
                self._remove(s, name)
            s.labels = []
            for i in range(count + 1):
                label = vtkTextActor()
                label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                prop = label.GetTextProperty()
                prop.SetFontFamilyToArial()
                prop.BoldOff()
                prop.ItalicOff()
                prop.ShadowOff()
                prop.SetColor(0, 0, 0)
                prop.SetJustificationToCentered()
                prop.SetVerticalJustificationToCentered()
                self._plotter.add_actor(
                    label,
                    name=f"cb_{key}_text{i}",
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
                s.actors[f"cb_{key}_text{i}"] = label
                s.labels.append(label)
        # The bar uses normalized coordinates; inverse ticks retain physical
        # units while giving nonlinear scales their full color distribution.
        values = norm.inverse(np.linspace(0, 1, count))
        for label, value in zip(s.labels, values):
            label.SetInput(format(float(value), style["label_format"]))
            label.GetTextProperty().SetOrientation(float(style["label_rotation"]))
            label.GetTextProperty().SetFontSize(int(style["font_size"]))
        # Reserve the measured text width, including when a window is resized
        # or changed limits produce longer numbers. Keep explicit positions.
        if not horizontal and style["position"] is None:
            sizes = []
            for label in s.labels[:-1]:
                size = [0.0, 0.0]
                label.GetSize(self._plotter.renderer, size)
                sizes.append(size[0])
            x = min(x, 1.0 - width - (max(sizes) + 12) / max(1, viewport_size[0]))
            if frame_right is not None:
                x = min(x, max(0.0, (frame_right + 12) / max(1, viewport_size[0])))
        s.colorbar.SetPosition(x, y)
        s.colorbar.SetWidth(width)
        s.colorbar.SetHeight(height)
        gap_x = 5.0 / max(1, viewport_size[0])
        gap_y = (style["font_size"] / 2 + 5) / max(1, viewport_size[1])
        for i, label in enumerate(s.labels[:-1]):
            (
                label.SetPosition(x + width * i / (count - 1), y - gap_y)
                if horizontal
                else label.SetPosition(x + width + gap_x, y + height * i / (count - 1))
            )
            if horizontal or style["label_rotation"]:
                label.GetTextProperty().SetJustificationToCentered()
            else:
                label.GetTextProperty().SetJustificationToLeft()
        title = s.labels[-1]
        title.SetInput(str(style["title"]))
        title.SetPosition(x + width / 2, y + height + 0.06)
        title.GetTextProperty().SetOrientation(float(style["title_rotation"]))
        title.GetTextProperty().SetFontSize(int(style["title_font_size"]))
        s._barSignature = signature

    def _refresh_layout(self, *_):
        """Keep text inside each viewport on export and interactive resize."""
        if self._plotter is None:
            return
        # VTK's multisample resolve leaves white fringes around thin lines
        # over an image surface. Single-sample 2D overlays retain the actual
        # background colors instead. Restore the window setting when removed.
        has_streamlines = any(
            s.domain.dim == 2
            and s.options.gridVisualization
            and s.options.gridVisualization.streamLines
            for s in self._states.values()
        )
        window = self._plotter.render_window
        if has_streamlines and self._streamline_multisamples is None:
            self._streamline_multisamples = window.GetMultiSamples()
            window.SetMultiSamples(0)
        elif not has_streamlines and self._streamline_multisamples is not None:
            window.SetMultiSamples(self._streamline_multisamples)
            self._streamline_multisamples = None
        active = self._plotter.renderers.active_index
        try:
            if self._title_actor is not None and (
                self._title_layout[1] != self._plotter.window_size[1]
            ):
                self.update_figure_title(self._title_text)
            for key, s in self._states.items():
                self._plotter.subplot(s.row, s.col)
                self._colorbar(key, s, s.norm, s.options)
                self._resize_particles(key, s)
        finally:
            self._plotter.subplot(*divmod(active, self._shape[1]))

    @serialized_notebook_render
    def show(self):
        if self._plotter is None:
            return
        if self._jupyter_backend == "static":
            self._plotter.render()
            try:
                from IPython import get_ipython

                ip = get_ipython()
            except ImportError:
                ip = None
            if ip is not None and "IPKernelApp" in getattr(ip, "config", {}):
                import io
                from PIL import Image
                from IPython.display import display, Image as IPImage

                buffer = io.BytesIO()
                Image.fromarray(self._plotter.screenshot(return_img=True)).save(
                    buffer, format="PNG"
                )
                output = IPImage(data=buffer.getvalue())
                if self._display_handle is None:
                    self._display_handle = display(output, display_id=True)
                else:
                    self._display_handle.update(output)
        elif self._jupyter_backend == "trame":
            if not self._shown:
                notebook_options = {
                    "mode": "server",
                    "width": "100%",
                    **self._opts.get("jupyter_kwargs", {}),
                }
                if self._trame_context_guard is None:
                    self._trame_context_guard = register_trame_window(
                        self._plotter, self._render_lock, notebook_options.get("name")
                    )
                self._plotter.show(
                    jupyter_backend="trame",
                    auto_close=False,
                    jupyter_kwargs=notebook_options,
                )
            else:
                self._plotter.render()
        elif not self._shown:
            self._plotter.show(interactive_update=True, auto_close=False)
        else:
            self._plotter.update()
        self._shown = True
        # Do not leave GLX/EGL bound when another plotting backend draws next.
        self._plotter.render_window.ReleaseCurrent()

    @serialized_notebook_render
    def update_figure_title(self, title):
        if self._plotter is None:
            return
        from vtkmodules.vtkRenderingCore import vtkRenderer, vtkTextActor

        text = title or ""
        self._title_text = text
        self._plotter.title = text or "warpSPHPlotting"
        if self._title_renderer is None and not text:
            return
        if self._title_renderer is None:
            renderer = vtkRenderer()
            layer = self._plotter.render_window.GetNumberOfLayers()
            renderer.SetLayer(layer)
            renderer.SetInteractive(False)
            renderer.SetPreserveColorBuffer(False)
            renderer.SetBackground(1, 1, 1)
            renderer.SetBackgroundAlpha(1)
            self._plotter.render_window.SetNumberOfLayers(layer + 1)
            self._plotter.render_window.AddRenderer(renderer)
            actor = vtkTextActor()
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            actor.SetPosition(0.5, 0.99)
            prop = actor.GetTextProperty()
            prop.SetFontFamilyToArial()
            prop.SetFontSize(12)
            prop.SetColor(0, 0, 0)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToTop()
            renderer.AddViewProp(actor)
            self._title_actor, self._title_renderer = actor, renderer
        self._title_actor.SetInput(text)
        lines = text.count("\n") + 1 if text else 0
        height = self._plotter.window_size[1]
        if (lines, height) != self._title_layout:
            font_size = max(10, round(12 * height / 480))
            self._title_actor.GetTextProperty().SetFontSize(font_size)
            band = min(0.4, (1.5 * font_size * lines + 8) / height) if lines else 0.0
            rows, cols = self._shape
            for key, (r, c) in self._axes.items():
                self._plotter.subplot(r, c)
                r0, c0, r1, c1 = self._spans[key]
                self._plotter.renderer.SetViewport(
                    c0 / cols,
                    (1 - (r1 + 1) / rows) * (1 - band),
                    (c1 + 1) / cols,
                    (1 - r0 / rows) * (1 - band),
                )
            self._title_renderer.SetDraw(bool(text))
            if text:
                self._title_renderer.SetViewport(0.0, 1.0 - band, 1.0, 1.0)
                self._title_actor.SetPosition(0.5, 0.92)
            self._title_lines = lines
            self._title_layout = (lines, height)

    @serialized_notebook_render
    def export(self, filepath, **kwargs):
        if self._plotter is None:
            raise RuntimeError("No open PyVista figure to export.")
        from pathlib import Path

        try:
            self._refresh_layout()
            # screenshot() can reuse the last rendered buffer after in-place
            # mesh/title edits. Render explicitly so every exported frame is current.
            self._plotter.render()

            kwargs.pop("dpi", None)
            if Path(filepath).suffix.lower() in (".svg", ".pdf", ".eps", ".ps"):
                self._plotter.save_graphic(filepath, **kwargs)
            else:
                self._plotter.screenshot(filepath, **kwargs)
        finally:
            self._plotter.render_window.ReleaseCurrent()

    @serialized_notebook_render
    def close(self):
        if self._plotter is not None:
            if self._trame_context_guard is not None:
                self._trame_context_guard.remove(self._plotter.render_window)
                self._trame_context_guard = None
            self._plotter.render_window.RemoveObserver(self._layout_observer)
            self._plotter.close()
            self._states.clear()
            self._title_actor = self._title_renderer = None
            self._streamline_multisamples = None
            self._plotter = None

    supports_grid = property(lambda self: True)
    supports_streamlines = property(lambda self: True)
    supports_notebook_inline = property(lambda self: True)

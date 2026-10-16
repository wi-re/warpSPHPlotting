"""Vispy rendering backend (Phase 4).

Uses a vispy scene canvas with per-panel ``ViewBox`` widgets arranged in a
grid to replicate matplotlib's subplot-mosaic layout.  Particles are rendered
via ``visuals.Markers`` point clouds.  Grid data is rendered as a coloured
``visuals.Image`` visual.  Scalar colours are computed with matplotlib norms
so that all norm types (Linear, Centered, Log, SymLog) produce the correct
colour distribution.

Display modes (``backendOptions={'jupyter_backend': ...}``)
-----------------------------------------------------------
``"native"`` (default outside Jupyter)
    Opens a pop-out window via the native OS widget.
``"notebook"`` (default inside Jupyter kernels when ``jupyter_rfb`` is
    installed)
    Renders inline as a Jupyter widget.  Falls back to ``"native"``
    silently when ``jupyter_rfb`` is not installed.
``"image"``
    Renders fully offscreen (typically combined with
    ``backendOptions={'app_backend': 'egl'}`` or ``'osmesa'`` for a headless
    GPU/software GL context -- no X server, no native window) and pushes each
    frame inline as a plain Jupyter image-display update. Unlike
    ``"notebook"``, this does not depend on an ``ipywidgets``/``jupyter_rfb``
    comm channel staying alive between kernel and browser, which does not
    reliably come up through a VS Code remote-SSH Jupyter connection even
    though the same kernel process renders fine headless. Use this when
    ``"notebook"`` mode shows nothing (or hangs) over such a connection.

Update pattern
--------------
After the initial :func:`~warpSPHPlotting.visualize.visualize` call, call
``plotState.updateQuantities(...)`` which dispatches to
:meth:`VispyBackend.update_panel`.  Existing ``Markers`` visuals are updated
in-place via ``set_data`` — no scene teardown — so position and colour
changes are fast.

Limits and dependencies
-----------------------
- Grid rendering and streamline integration use 2D fields or planar 3D slices.
- Axis labels are shown for 2D views; 3D scenes use a fitted turntable camera.
- Matplotlib supplies normalization, colormaps, and streamline integration.
- Image/notebook display requires IPython; EGL requires an available GL runtime.
"""

from __future__ import annotations

import contextlib
import io as _io
import os as _os
import sys as _sys
import threading
from functools import wraps
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .base import AbstractBackend
from .asyncExport import submitImageWrite
from ..enumTypes import VisualizeOptions
from ..options import PlottingOptions
from .._sizing import particle_diameter, explicit_pixel_size

# ─────────────────────────────────────────────────────────────────────────────
# Panel state dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VispyVisualizationState:
    """Per-panel rendering state kept by the Vispy backend."""

    canvas: Any  # shared vispy.scene.SceneCanvas reference
    view: Any  # ViewBox for this panel

    # Marker visuals; None if the particle type is hidden or absent
    markers_fluid: Optional[Any]
    markers_boundary: Optional[Any]

    # Line visual for the domain bounding box; None when plotDomain=False
    domain_lines: Optional[Any]

    # Image visual for grid mode; None when gridVisualization is off
    image_grid: Optional[Any]

    # ColorBarWidget placed to the right of the view; None if showColorBar=False
    colorbar: Optional[Any]

    domain: Any  # DomainDescription (detached copy)
    options: PlottingOptions

    # Per-type particle states (for update bookkeeping)
    fluidParticles: Any
    boundaryParticles: Any
    ghostParticles: Any
    assembledQuantity: Any
    rotatedQuantities: Any
    title_visual: Any = None
    stream_visual: Any = None
    cached_colorbar: Any = None


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_3d(positions) -> np.ndarray:
    """Convert (N, 2-or-3) float tensor → (N, 3) float32 numpy array."""
    pts = positions.detach().cpu().numpy().astype(np.float32)
    if pts.ndim == 1:
        pts = pts[:, None]
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
    return pts


def _parse_mosaic(mosaic: str) -> Tuple[Tuple[int, int], Dict[str, Tuple[int, int]]]:
    """Parse a matplotlib mosaic string → (nrows, ncols), {label: (row, col)}.

    Handles simple rectangular mosaics like ``'AB'`` or ``'AB\\nCD'``.
    """
    rows = [r for r in mosaic.strip().split("\n") if r.strip()]
    nrows = len(rows)
    ncols = max(len(r) for r in rows)
    label_to_pos: Dict[str, Tuple[int, int]] = {}
    for r, row_str in enumerate(rows):
        for c, label in enumerate(row_str):
            if label.strip() and label not in label_to_pos:
                label_to_pos[label] = (r, c)
    return (nrows, ncols), label_to_pos


@contextlib.contextmanager
def _suppress_vispy_init_warnings():
    """Suppress libEGL/DRI3 and DPI warnings emitted during vispy canvas creation.

    Two sources of noise are silenced:

    1. **libEGL / DRI3** – written directly to OS file-descriptor 2 (bypassing
       Python's ``sys.stderr``), so we redirect fd 2 at the OS level.
    2. **"could not determine DPI"** – emitted via vispy's internal Python
       logger (``vispy.util.logs.logger``), not to stderr.  On WSL and
       headless Linux there is no X server, so ``xdpyinfo`` / ``xrandr``
       both fail and vispy falls back to 96 DPI after logging the warning.
       We silence it by temporarily raising the vispy logger's level to
       ``ERROR`` for the duration of canvas creation.
    """
    import logging as _logging  # noqa: PLC0415

    # ── fd-2 redirect (libEGL) ───────────────────────────────────────────────
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    saved_fd2 = _os.dup(2)
    _os.dup2(devnull_fd, 2)
    saved_stderr = _sys.stderr
    _sys.stderr = _io.StringIO()

    # ── vispy logger (DPI warning) ───────────────────────────────────────────
    try:
        import vispy.util.logs as _vispy_logs  # noqa: PLC0415

        _vispy_logger = _vispy_logs.logger
        _saved_level = _vispy_logger.level
        _vispy_logger.setLevel(_logging.ERROR)
    except Exception:
        _vispy_logger = None
        _saved_level = None

    try:
        yield
    finally:
        _os.dup2(saved_fd2, 2)
        _os.close(devnull_fd)
        _os.close(saved_fd2)
        _sys.stderr = saved_stderr
        if _vispy_logger is not None:
            _vispy_logger.setLevel(_saved_level)


def _get_cmap(name: str):
    """Return a matplotlib colormap by name; tries seaborn registration if needed."""
    import matplotlib as mpl  # noqa: PLC0415

    try:
        return mpl.colormaps[name]
    except KeyError:
        # Some colormaps (e.g. 'rocket', 'mako') require seaborn to be imported
        # first so they are registered with matplotlib.
        try:
            import seaborn  # noqa: F401, PLC0415
        except ImportError:
            pass
        try:
            return mpl.colormaps[name]
        except KeyError:
            pass
    # Final fallback for very old matplotlib
    import matplotlib.cm as mcm  # noqa: PLC0415

    return mcm.get_cmap(name)


def _mpl_to_vispy_cmap(cmap_name: str):
    """Build a vispy ``Colormap`` from a 256-sample matplotlib LUT.

    This bridges matplotlib colormaps (including seaborn ones) into the vispy
    ``Colormap`` object expected by ``ColorBarWidget``.
    """
    from vispy.color import Colormap as VispyColormap  # noqa: PLC0415

    mpl_cmap = _get_cmap(cmap_name)
    lut = mpl_cmap(np.linspace(0.0, 1.0, 256)).astype(np.float32)

    class PersistentLUTColormap(VispyColormap):
        def texture_lut(self):
            # Vispy otherwise allocates a new texture even when only a
            # colorbar's limits or geometry changed. Our LUT is immutable.
            if not hasattr(self, "_plotting_texture"):
                self._plotting_texture = super().texture_lut()
            return self._plotting_texture

    return PersistentLUTColormap(lut)


def _scalars_to_rgba(q_clipped: np.ndarray, norm, cmap_name: str) -> np.ndarray:
    """Convert a scalar array → (N, 4) RGBA float32 via matplotlib norm + cmap.

    Applying the matplotlib norm directly (rather than a plain linear rescale
    to ``clim``) ensures all norm types — ``Normalize``, ``CenteredNorm``,
    ``LogNorm``, ``SymLogNorm`` — produce the correct colour distribution.

    The input is ravelled to 1-D before processing so that quantities stored
    as column tensors of shape ``(N, 1)`` still produce an ``(N, 4)`` output
    rather than ``(N, 1, 4)`` which vispy ``Markers`` cannot interpret
    correctly.
    """
    q_flat = np.ma.asarray(q_clipped).ravel()  # guarantee 1-D → output is (N, 4)
    cmap = _get_cmap(cmap_name)
    normed = np.ma.filled(norm(q_flat), np.nan)
    normed = np.clip(normed, 0.0, 1.0)
    return cmap(normed).astype(np.float32)


def _cmap_name(options: PlottingOptions) -> str:
    return options.colorMap.value + ("_r" if options.flipColorMap else "")


def _marker_sizes(options, fluid, boundary, default_pixel_size):
    if options.markerSize is not None:
        size = explicit_pixel_size(options.markerSize)
        return "fixed", size, size
    if default_pixel_size is not None:
        return "fixed", default_pixel_size, default_pixel_size

    def diameter(particles, variant):
        if options.gridVisualization is not None or variant == VisualizeOptions.Hide:
            return 0.0
        return particle_diameter(particles, options)

    return (
        "scene",
        diameter(fluid, options.fluidVisualization),
        diameter(boundary, options.boundaryVisualization),
    )


def _domain_line_positions_2d(domain_) -> np.ndarray:
    """Return (5, 3) float32 positions for a closed 2-D bounding-box rectangle."""
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    return np.array(
        [
            [mn[0], mn[1], 0.0],
            [mx[0], mn[1], 0.0],
            [mx[0], mx[1], 0.0],
            [mn[0], mx[1], 0.0],
            [mn[0], mn[1], 0.0],  # close the loop
        ],
        dtype=np.float32,
    )


def _domain_line_positions_3d(domain_) -> np.ndarray:
    """Return (24, 3) float32 positions for all 12 edges of a 3-D bbox.

    Arranged as 12 disjoint segments (use ``connect='segments'``).
    """
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])
    return np.array(
        [
            # bottom face
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y1, z0],
            [x0, y0, z0],
            # top face
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x1, y1, z1],
            [x0, y1, z1],
            [x0, y1, z1],
            [x0, y0, z1],
            # verticals
            [x0, y0, z0],
            [x0, y0, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z0],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def _turntable_camera(domain):
    from vispy.scene.cameras import TurntableCamera

    class DomainTurntableCamera(TurntableCamera):
        _plot_bounds = None

        def _update_projection_transform(self, fx, fy):
            bounds = self._plot_bounds
            if bounds is not None:
                if self.fov == 0:
                    mn, mx = bounds
                    # Vispy's default ortho range is +/- 1e6 world units.
                    # Small domains then lose depth precision on NVIDIA and
                    # overlapping particles render according to input order.
                    # Follow the domain, camera translation, and orbit distance.
                    self._depth_value = max(
                        float(
                            2 * np.linalg.norm(mx - mn)
                            + np.linalg.norm(np.asarray(self.center) - (mn + mx) / 2)
                            + abs(self.distance or 0)
                        ),
                        np.finfo(np.float32).tiny,
                    )
                else:
                    # For perspective, this property is a near/far ratio.
                    self._depth_value = 1_000
            return super()._update_projection_transform(fx, fy)

    camera = DomainTurntableCamera()
    # Camera interactions must not synchronize a running CUDA simulation.
    camera._plot_bounds = (domain.min.cpu().numpy(), domain.max.cpu().numpy())
    return camera


def _fit_camera_to_domain(view: Any, domain_, margin: float = 0.05) -> None:
    """Set PanZoomCamera range to the domain extent with a proportional margin."""
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    dx = float(mx[0] - mn[0])
    dy = float(mx[1] - mn[1])
    if domain_.dim == 2:
        # PanZoomCamera uses a 2-D transform matrix that becomes singular when
        # either axis span is exactly zero. Guard collapsed/near-collapsed
        # domains by enforcing a small minimum span around the axis center.
        min_span = 1e-9
        span_x = max(abs(dx), min_span)
        span_y = max(abs(dy), min_span)

        cx = 0.5 * (float(mn[0]) + float(mx[0]))
        cy = 0.5 * (float(mn[1]) + float(mx[1]))

        half_x = 0.5 * span_x * (1.0 + 2.0 * margin)
        half_y = 0.5 * span_y * (1.0 + 2.0 * margin)

        x0, x1 = cx - half_x, cx + half_x
        y0, y1 = cy - half_y, cy + half_y

        try:
            view.camera.set_range(x=(x0, x1), y=(y0, y1))
        except Exception:
            # Final fallback keeps the camera valid even if upstream vispy
            # rejects a particular range on first layout/resize pass.
            eps = 0.5
            view.camera.set_range(x=(cx - eps, cx + eps), y=(cy - eps, cy + eps))
    else:
        view.camera._plot_bounds = (mn, mx)
        view.camera.set_range(
            x=(float(mn[0]), float(mx[0])),
            y=(float(mn[1]), float(mx[1])),
            z=(float(mn[2]), float(mx[2])),
            margin=margin,
        )


def _view_rect_is_degenerate(view: Any, eps: float = 1e-9) -> bool:
    """Return True when a view has an invalid/near-zero layout rect."""
    try:
        rect = view.rect
        if rect is None:
            return True
        return float(rect.width) <= eps or float(rect.height) <= eps
    except Exception:
        return True


def _camera_rect_is_degenerate(view: Any, eps: float = 1e-12) -> bool:
    """Return True when the attached camera rect has near-zero span."""
    try:
        cam = getattr(view, "camera", None)
        rect = getattr(cam, "rect", None)
        if rect is None:
            return False
        return abs(float(rect.width)) <= eps or abs(float(rect.height)) <= eps
    except Exception:
        return False


def _grid_to_rgba_image(
    gridState: Any,
    gridQuantity: Any,
    nxs: Tuple[int, int],
    gridExtent: Any,
    opts: PlottingOptions,
    cmap_name: str,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Build a (ny, nx, 4) RGBA float32 image from ``mapToGrid`` output.

    Returns
    -------
    rgba_img : np.ndarray  shape ``(ny, nx, 4)``
    (ox, oy, dx, dy) : world-coordinate origin and per-cell spacing

    Notes
    -----
    ``generateGrid`` uses ``indexing='ij'`` (C-order), so the flat index is
    ``k = ny * ix + iy``.  Reshaping to ``(nx, ny)`` and transposing gives
    ``(ny, nx)`` = ``(H, W)`` where row corresponds to y-index and column to
    x-index — the correct layout for a vispy ``Image`` visual whose column
    axis is x and row axis is y.
    """
    from ..grid import gridDisplayBounds

    qs, norm = gridDisplayBounds(gridQuantity, gridExtent, opts)
    nx, ny = nxs[0], nxs[1]

    # Reshape: ij-order (nx, ny) → transpose → (ny, nx) = (H, W)
    q_img = qs.reshape(nx, ny).T
    normed = np.ma.filled(norm(q_img) if norm is not None else q_img, np.nan)
    normed = np.clip(normed, 0.0, 1.0)
    rgba_img = _get_cmap(cmap_name)(normed).astype(np.float32)  # (H, W, 4)

    # Use the domain extent (cell *edges*) as image origin and cell size.
    # Grid points are cell *centers* (alignment='center'), so using pts[0] as
    # the origin would shift the image by half a cell.  gridExtent always
    # contains domain.min / domain.max which are the true edge coordinates.
    mn = gridExtent["min"]
    mx = gridExtent["max"]
    if hasattr(mn, "cpu"):
        mn = mn.cpu().numpy()
        mx = mx.cpu().numpy()
    ox = float(mn[0])
    oy = float(mn[1])
    dx = float((mx[0] - mn[0]) / nx)
    dy = float((mx[1] - mn[1]) / ny)

    return rgba_img, (ox, oy, dx, dy)


def _image_transform(ox: float, oy: float, dx: float, dy: float):
    """Return a vispy ``STTransform`` placing image pixels in world coordinates.

    A vispy ``Image`` of shape ``(H, W)`` has pixel ``(col, row)`` at canvas
    position ``(col, row)``.  The returned transform maps this to world
    ``(ox + col * dx,  oy + row * dy)``, matching the grid layout where
    column = x-index and row = y-index.
    """
    from vispy.visuals.transforms import STTransform  # noqa: PLC0415

    return STTransform(scale=(dx, dy), translate=(ox, oy))


def _release_current_gl_context(canvas):
    """Release offscreen contexts after readback or notebook frame delivery.

    Vispy's EGL backend destroys a surface on close without unbinding its
    context. NVIDIA can then reject VTK's GLXMakeCurrent with BadAccess.
    Notebook GLFW contexts also need releasing before a different kernel
    thread can render a widget callback or export.
    Do not destroy shared contexts or terminate the application display.
    """
    name = canvas.app.backend_name.lower()
    if name == "jupyter_rfb":
        # Notebook execution and widget callbacks can run on different kernel
        # threads. GLFW cannot bind a context still current on another thread.
        context = canvas._backend._context
        if context.glfw:
            context.glfw.make_context_current(None)
        return
    if name != "egl":
        return
    from vispy.app.backends import _egl

    egl = _egl.egl
    egl.eglMakeCurrent(
        _egl._EGL_DISPLAY, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT
    )


class _CanvasAccess:
    """Keep widget draws, scene mutations and framebuffer access on one path."""

    def __init__(self):
        self.lock = threading.RLock()
        self.local = threading.local()
        self.drawing = False
        self.closed = False
        self.redraw_pending = False
        self.schedule_redraw = None

    @contextlib.contextmanager
    def use(self, canvas):
        with self.lock:
            depth = getattr(self.local, "depth", 0)
            self.local.depth = depth + 1
            try:
                if depth == 0 and canvas is not None and not self.closed:
                    canvas.set_current()
                yield
            finally:
                self.local.depth = depth
                if depth == 0 and canvas is not None:
                    _release_current_gl_context(canvas)
                    if self.redraw_pending and not self.closed:
                        self.redraw_pending = False
                        if self.schedule_redraw is not None:
                            self.schedule_redraw()


def _serialized_canvas(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._canvas_access.use(self._canvas):
            return method(self, *args, **kwargs)

    return call


def _position_image(image, ox, oy, dx, dy):
    """Keep the scene/shader transform graph when only pixels change."""
    from vispy.visuals.transforms import STTransform

    if isinstance(image.transform, STTransform):
        # STTransform suppresses events when these values are unchanged.
        image.transform.scale = (dx, dy)
        image.transform.translate = (ox, oy)
    else:
        image.transform = _image_transform(ox, oy, dx, dy)


def _style_colorbar_tick_labels(widget: Any) -> None:
    """Keep physical endpoint labels upright and beside the vertical strip."""
    cb = widget._colorbar
    x, y = cb.pos
    half_width, half_height = cb._halfdim
    for tick, ty in zip(cb._ticks, (y + half_height, y - half_height)):
        tick.rotation = 0
        tick.font_size = 9
        tick.anchors = ("left", "center")
        tick.pos = (x + half_width + 5, ty)
    # Include the longest number in the widget's allocation, rather than
    # rotating numbers to squeeze them into a fixed narrow column.
    width = max(80, 34 + 7 * max(len(str(v)) for v in cb.clim))
    if widget.width_min != width:
        widget.width_min = widget.width_max = width


def _make_colorbar(cmap, limits):
    from vispy.scene.widgets.colorbar import ColorBarWidget

    class UprightColorBarWidget(ColorBarWidget):
        def _update_colorbar(self):
            height = max(1, self.rect.height)
            # The widget follows the view directly: 12 px gap, 14 px strip.
            self._colorbar.pos = (19, height / 2)
            self._colorbar.size = (max(14, height * 0.8), 14)
            _style_colorbar_tick_labels(self)

    return UprightColorBarWidget(
        cmap=_mpl_to_vispy_cmap(cmap),
        orientation="right",
        label="",
        label_color="black",
        clim=tuple(f"{v:.6g}" for v in limits),
    )


def _in_ipykernel() -> bool:
    """Best-effort detection of a running Jupyter kernel."""
    try:
        from IPython import get_ipython  # noqa: PLC0415

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def _sharedMarkerBounds(fluid, boundary, opts):
    """`math.getSharedBounds` for the two marker sets, honouring Hide/Passive.

    A set that is hidden, drawn passive (flat grey), or empty contributes
    nothing to the norm -- it is not coloured by the quantity, so letting its
    values set the range would stretch the map for no visible gain.
    """
    from ..math import getSharedBounds  # noqa: PLC0415

    coloured = (VisualizeOptions.Hide, VisualizeOptions.Passive)
    fluidQuantities = (
        fluid.quantities
        if opts.fluidVisualization not in coloured and fluid.positions.shape[0] > 0
        else None
    )
    boundaryQuantities = (
        boundary.quantities
        if opts.boundaryVisualization not in coloured
        and boundary.positions.shape[0] > 0
        else None
    )
    return getSharedBounds(fluidQuantities, boundaryQuantities, opts)


def _no_display_available() -> bool:
    """True on a Linux session with neither X11 nor Wayland reachable.

    The signal vispy's own GUI-toolkit probe (glfw/Qt/...) has no way to
    check for cheaply up front -- it just tries to open a window and raises.
    Non-Linux platforms are left alone: macOS/Windows native windowing
    doesn't depend on `DISPLAY`, and EGL is a Linux-specific escape hatch.
    """
    return (
        _sys.platform.startswith("linux")
        and not _os.environ.get("DISPLAY")
        and not _os.environ.get("WAYLAND_DISPLAY")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backend implementation
# ─────────────────────────────────────────────────────────────────────────────


class VispyBackend(AbstractBackend):
    """Vispy / OpenGL rendering backend (Phase 4).

    Renders particles as ``visuals.Markers`` point clouds on a
    ``SceneCanvas`` divided into one ``ViewBox`` per mosaic panel.  Scalar
    colours are computed with matplotlib norms + colormaps so all norm types
    (Linear, Centered, Log, SymLog) render correctly.

    Pass ``backendOptions={'jupyter_backend': 'notebook'}`` to enable inline
    Jupyter display (requires ``jupyter_rfb``).
    """

    def __init__(self) -> None:
        self._canvas_access = _CanvasAccess()
        self._canvas: Any = None
        self._grid_widget: Any = None  # top-level vispy Grid
        self._subgrids: Dict[str, Any] = {}  # label → per-panel sub-Grid
        self._axes: Dict[str, Tuple[int, int]] = {}
        self._views: Dict[str, Any] = {}  # label → ViewBox
        self._figure_title_widget: Any = None
        self._figure_title_height_px: int = 36
        self._figure_title: str = ""
        self._jupyter_mode: str = "native"
        self._app_backend: Optional[str] = None
        self._image_display_handle: Any = None
        self._window_pos: Optional[Tuple[int, int]] = None
        self._force_primary_window: bool = False
        self._point_size: Optional[float] = None
        self._shown: bool = False
        self._canvas_size: Tuple[int, int] = (960, 480)  # (px_w, px_h)
        self._states = {}
        self._sharex = self._sharey = False
        self._shape: Tuple[int, int] = (1, 1)  # (nrows, ncols)

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
        import vispy.app as vapp  # noqa: PLC0415
        import vispy.scene as vs  # noqa: PLC0415
        from vispy.scene.widgets import Label  # noqa: PLC0415

        opts = backendOptions or {}

        # Resolve display mode, with graceful fallback when jupyter_rfb is absent
        default_mode = "notebook" if _in_ipykernel() else "native"
        self._jupyter_mode = opts.get("jupyter_backend", default_mode)
        if self._jupyter_mode == "notebook":
            try:
                import jupyter_rfb  # noqa: F401
            except ImportError:
                self._jupyter_mode = "native"

        # In script/native mode we allow pinning the vispy app backend (e.g.
        # PyQt5) to match standalone vispy examples and avoid backend-specific
        # window mapping quirks. "image" mode needs this too -- that's how a
        # headless EGL/OSMesa context gets selected instead of whatever GUI
        # toolkit vispy would otherwise probe for.
        self._app_backend = opts.get("app_backend", None)
        if (
            self._app_backend is None
            and self._jupyter_mode in ("native", "image")
            and _no_display_available()
        ):
            # vispy's backend choice is a one-shot, process-global decision:
            # the *first* canvas it ever creates locks it in, even if that
            # backend then fails to initialise (`RuntimeError: Can only
            # select a backend once, already using [...]` on any later
            # `use_app()` call in the same process). So a headless Linux
            # session -- no DISPLAY, no WAYLAND_DISPLAY -- has to be routed to
            # EGL's offscreen GL context *before* that first attempt, not
            # retried afterwards once the default GUI-toolkit probe
            # (glfw/Qt/...) has already failed for lack of a display.
            self._app_backend = "egl"
        if self._jupyter_mode in ("native", "image") and self._app_backend:
            try:
                vapp.use_app(self._app_backend)
            except Exception:
                # Keep vispy's default backend selection as a fallback.
                pass

        pos_opt = opts.get("window_pos", None)
        if isinstance(pos_opt, (list, tuple)) and len(pos_opt) == 2:
            self._window_pos = (int(pos_opt[0]), int(pos_opt[1]))
        else:
            self._window_pos = None
        self._force_primary_window = bool(opts.get("force_primary_window", False))

        self._point_size = float(opts["point_size"]) if "point_size" in opts else None
        self._shown = False
        self._figure_title = figTitle or ""

        from ._mosaic import parse_mosaic

        shape, label_to_pos, spans = parse_mosaic(mosaic)
        self._sharex, self._sharey = sharex, sharey
        self._axes = label_to_pos
        self._shape = shape

        px_w = int(figsize[0] * 96)
        px_h = int(figsize[1] * 96)
        self._canvas_size = (px_w, px_h)

        access = self._canvas_access

        class PlottingCanvas(vs.SceneCanvas):
            def render(self, *args, **kwargs):
                # EGL's programmatic resize changes the surface without a
                # resize event. Update widgets/transforms before readback.
                with access.use(self):
                    if access.closed:
                        raise RuntimeError("Cannot render a closed Vispy canvas.")
                    if access.drawing:
                        raise RuntimeError(
                            "Export the Vispy canvas after its draw callback returns."
                        )
                    access.drawing = True
                    try:
                        if tuple(self.central_widget.size) != tuple(self.size):
                            self.events.resize(
                                size=self.size, physical_size=self.physical_size
                            )
                        return super().render(*args, **kwargs)
                    finally:
                        access.drawing = False

        with _suppress_vispy_init_warnings():
            self._canvas = PlottingCanvas(
                title=figTitle or "warpSPHPlotting",
                size=(px_w, px_h),
                keys="interactive",
                show=False,
                bgcolor="white",
            )

        if self._canvas.app.backend_name.lower() == "jupyter_rfb":
            # Include asynchronous browser-triggered draws, which bypass
            # PlottingCanvas.render() and our synchronous show() path.
            canvas = self._canvas
            backend = canvas._backend
            get_frame = backend.get_frame

            def released_frame():
                with access.use(canvas):
                    if access.closed:
                        return None
                    if access.drawing:
                        # A draw callback may request show() synchronously.
                        # Queue another frame instead of re-entering events.draw.
                        access.redraw_pending = True
                        return None
                    access.drawing = True
                    try:
                        return get_frame()
                    finally:
                        access.drawing = False

            backend.get_frame = released_frame

            def schedule_redraw():
                if not backend._loop.is_closed():
                    backend._loop.call_soon_threadsafe(backend.request_draw)

            access.schedule_redraw = schedule_redraw

            def protected_event(method):
                @wraps(method)
                def call(*args, **kwargs):
                    with access.use(canvas):
                        if not access.closed:
                            return method(*args, **kwargs)

                return call

            backend.handle_event = protected_event(backend.handle_event)
            backend._emit_resize_event = protected_event(backend._emit_resize_event)
            _release_current_gl_context(self._canvas)

        self._grid_widget = self._canvas.central_widget.add_grid(spacing=2)

        # Keep an explicit title row so figure titles are visible in native
        # windows and notebook widgets (window-title text alone is not).
        self._figure_title_widget = Label(self._figure_title, color="black")
        self._figure_title_widget.height_min = 0
        self._figure_title_widget.height_max = 0
        self._grid_widget.add_widget(
            self._figure_title_widget,
            row=0,
            col=0,
            col_span=max(shape[1], 1),
        )

        # Apply title text + visibility/height and sync native window title.
        self.update_figure_title(figTitle)

        self._views = {}
        self._subgrids = {}
        self._panel_titles = {}
        for label, (r, c) in label_to_pos.items():
            # Each mosaic panel gets a sub-grid so the colorbar column is
            # contained within the panel's allocated space. Columns hold the
            # y axis, view, and optional colorbar; a title row stays above them.
            r0, c0, r1, c1 = spans[label]
            sg = self._grid_widget.add_grid(
                row=r + 1, col=c, row_span=r1 - r0 + 1, col_span=c1 - c0 + 1, spacing=0
            )
            title = Label("", color="black", font_size=10)
            title.height_min = title.height_max = 0
            sg.add_widget(title, row=0, col=1)
            self._panel_titles[label] = title
            view = sg.add_view(row=1, col=1, bgcolor="white", border_color="gray")
            from vispy.scene.widgets import AxisWidget

            yaxis = AxisWidget(
                orientation="left",
                axis_label="Y",
                text_color="black",
                axis_color="black",
                tick_color="black",
                font_size=8,
                axis_width=1,
                tick_width=1,
                major_tick_length=5,
                minor_tick_length=2,
            )
            yaxis.width_max = 45
            xaxis = AxisWidget(
                orientation="bottom",
                axis_label="X",
                text_color="black",
                axis_color="black",
                tick_color="black",
                font_size=8,
                axis_width=1,
                tick_width=1,
                major_tick_length=5,
                minor_tick_length=2,
            )
            xaxis.height_max = 45
            sg.add_widget(yaxis, row=1, col=0)
            sg.add_widget(xaxis, row=2, col=1)
            self.__dict__.setdefault("_axisWidgets", {})[label] = (xaxis, yaxis)
            self._subgrids[label] = sg
            self._views[label] = view

        return self._canvas

    def get_axes(self) -> Dict[str, Tuple[int, int]]:
        return self._axes

    # -------------------------------------------------------------------------
    # Per-panel rendering
    # -------------------------------------------------------------------------

    @_serialized_canvas
    def render_panel(
        self,
        panel_key: str,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: PlottingOptions,
        _preparedFrame=None,
    ) -> VispyVisualizationState:
        import vispy.scene.visuals as visuals  # noqa: PLC0415

        from ._render_util import prepare_frame  # noqa: PLC0415

        view = self._views[panel_key]

        # ── preprocessing ────────────────────────────────────────────────────
        frame = _preparedFrame or prepare_frame(
            particleState, domain, quantity, options
        )
        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = (
            frame.domain,
            frame.options,
            frame.assembled,
            frame.particles.quantities,
            frame.fluid,
            frame.boundary,
            frame.ghost,
        )
        if opts.gridVisualization is not None and domain_.dim != 2:
            raise ValueError(
                "Vispy grid rendering currently requires a 2D domain; use PyVista for 3D grids."
            )

        cmap = _cmap_name(opts)

        marker_scaling, fluid_size, boundary_size = _marker_sizes(
            opts, fluid, boundary, self._point_size
        )

        # ── camera ────────────────────────────────────────────────────────────
        view.camera = "panzoom" if domain_.dim == 2 else _turntable_camera(domain_)
        if domain_.dim == 2:
            view.camera.aspect = 1.0  # enforce equal x/y scale

        for i, axis_widget in enumerate(self._axisWidgets[panel_key]):
            axis_widget.visible = domain_.dim == 2
            if domain_.dim == 2:
                axis_widget.axis.axis_label = "XYZ"[(frame.displayAxes or [0, 1])[i]]
                axis_widget.link_view(view)
        # ── panel title ───────────────────────────────────────────────────────
        title_visual = self._panel_titles[panel_key]
        title_visual.text = opts.plotTitle or ""
        title_visual.height_min = title_visual.height_max = 26 if opts.plotTitle else 0

        # ── domain bounding box ────────────────────────────────────────────────
        domain_lines = None
        if opts.plotDomain:
            if domain_.dim == 2:
                line_pts = _domain_line_positions_2d(domain_)
                domain_lines = visuals.Line(
                    line_pts, color="blue", connect="strip", parent=view.scene
                )
            else:
                line_pts = _domain_line_positions_3d(domain_)
                domain_lines = visuals.Line(
                    line_pts, color="blue", connect="segments", parent=view.scene
                )
            domain_lines.order = 20
            if domain_.dim == 2:
                domain_lines.set_gl_state("translucent", depth_test=False)

        # ── grid or scatter path ───────────────────────────────────────────────
        markers_fluid, markers_boundary, image_grid = None, None, None
        # Set on the marker path below; stays None on the grid path, where
        # the colorbar takes its range from the grid norm instead.
        fluidQs = boundaryQs = sharedNorm = None

        if opts.gridVisualization is not None:
            # ── grid (Image visual) ───────────────────────────────────────────
            from ..grid import mapFrameToGrid, gridDisplayBounds  # noqa: PLC0415

            gridState, gridQuantity, nxs, gridExtent = mapFrameToGrid(frame, rotated_q)

            rgba_img, (ox, oy, dx, dy) = _grid_to_rgba_image(
                gridState, gridQuantity, nxs, gridExtent, opts, cmap
            )
            image_grid = visuals.Image(rgba_img, parent=view.scene)
            image_grid.set_gl_state("translucent", depth_test=False)
            _position_image(image_grid, ox, oy, dx, dy)

        else:
            # One norm across fluid AND boundary -- see `math.getSharedBounds`.
            # Normalising them separately and drawing both through the same
            # colormap makes wall colours incomparable to fluid colours.
            fluidQs, boundaryQs, sharedNorm = frame.sharedBounds or _sharedMarkerBounds(
                fluid, boundary, opts
            )

            # ── fluid markers ─────────────────────────────────────────────────
            if (
                opts.fluidVisualization != VisualizeOptions.Hide
                and fluid.positions.shape[0] > 0
            ):
                pts = _to_3d(fluid.positions)
                if opts.fluidVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(fluidQs, sharedNorm, cmap)
                markers_fluid = visuals.Markers(antialias=0, parent=view.scene)
                markers_fluid.scaling = marker_scaling
                markers_fluid.set_data(
                    pts,
                    face_color=face_color,
                    edge_color=face_color,
                    size=fluid_size,
                    edge_width=0,
                )

            # ── boundary markers ──────────────────────────────────────────────
            if (
                opts.boundaryVisualization != VisualizeOptions.Hide
                and boundary.positions.shape[0] > 0
            ):
                pts = _to_3d(boundary.positions)
                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(boundaryQs, sharedNorm, cmap)
                markers_boundary = visuals.Markers(antialias=0, parent=view.scene)
                markers_boundary.scaling = marker_scaling
                markers_boundary.set_data(
                    pts,
                    face_color=face_color,
                    edge_color=face_color,
                    size=boundary_size,
                    edge_width=0,
                )

        # ── camera fit ────────────────────────────────────────────────────────
        from ._render_util import view_domain

        _fit_camera_to_domain(view, view_domain(frame), margin=opts.domainEpsilon)

        # ── colorbar ──────────────────────────────────────────────────────────
        colorbar_widget = None
        if opts.showColorBar:
            # Determine which scalar range and cmap to use for the colorbar.
            # Prefer the fluid (active) range; fall back to grid range.

            cb_range: Optional[Tuple[float, float]] = None
            cb_cmap = cmap
            if opts.gridVisualization is not None:
                # Derive range from the grid quantity's norm so the colorbar
                # always reflects the actual data limits, not the normalised
                # [0, 1] interval.
                if image_grid is not None:
                    _, grid_norm = gridDisplayBounds(gridQuantity, gridExtent, opts)
                    cb_range = (
                        (float(grid_norm.vmin), float(grid_norm.vmax))
                        if grid_norm is not None
                        else None
                    )
            elif sharedNorm is not None:
                # The same norm both marker sets were coloured with, so the bar
                # describes every particle on screen rather than just the fluid.
                cb_range = (float(sharedNorm.vmin), float(sharedNorm.vmax))

            if cb_range is not None:
                colorbar_widget = _make_colorbar(cb_cmap, cb_range)
                self._subgrids[panel_key].add_widget(colorbar_widget, row=1, col=2)

        state = VispyVisualizationState(
            canvas=self._canvas,
            view=view,
            markers_fluid=markers_fluid,
            markers_boundary=markers_boundary,
            domain_lines=domain_lines,
            image_grid=image_grid,
            colorbar=colorbar_widget,
            domain=domain_,
            options=opts,
            fluidParticles=fluid,
            boundaryParticles=boundary,
            ghostParticles=ghost,
            assembledQuantity=assembled,
            rotatedQuantities=rotated_q,
            title_visual=title_visual,
        )
        state.cameraGroup = (
            domain_.dim,
            tuple(frame.displayAxes or range(domain_.dim)),
        )
        state.displayAxes = tuple(frame.displayAxes or range(domain_.dim))
        self._states[panel_key] = state
        self._link_compatible_cameras()
        self._update_streamlines(state, frame)
        return state

    @_serialized_canvas
    def update_panel(
        self,
        panel_key: str,
        panel_state: VispyVisualizationState,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: PlottingOptions,
        **kwargs: Any,
    ) -> VispyVisualizationState:
        import copy as _copy  # noqa: PLC0415
        import vispy.scene.visuals as visuals  # noqa: PLC0415

        from ._render_util import (
            prepare_frame,
            view_needs_refit,
            view_domain,
        )  # noqa: PLC0415

        cached_frame = kwargs.pop("_preparedFrame", None)
        old_domain = panel_state.domain
        # Apply any option overrides passed as keyword arguments
        if kwargs:
            options = _copy.deepcopy(options)
            for k, v in kwargs.items():
                if hasattr(options, k):
                    setattr(options, k, v)

        frame = cached_frame or prepare_frame(particleState, domain, quantity, options)
        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = (
            frame.domain,
            frame.options,
            frame.assembled,
            frame.particles.quantities,
            frame.fluid,
            frame.boundary,
            frame.ghost,
        )
        if opts.gridVisualization is not None and domain_.dim != 2:
            raise ValueError(
                "Vispy grid rendering currently requires a 2D domain; use PyVista for 3D grids."
            )

        cmap = _cmap_name(opts)

        marker_scaling, fluid_size, boundary_size = _marker_sizes(
            opts, fluid, boundary, self._point_size
        )

        view = panel_state.view
        if old_domain.dim != domain_.dim:
            self._unlink_camera(view.camera)
            view.camera = "panzoom" if domain_.dim == 2 else _turntable_camera(domain_)
            if domain_.dim == 2:
                view.camera.aspect = 1.0
        group = (domain_.dim, tuple(frame.displayAxes or range(domain_.dim)))
        if group != getattr(panel_state, "cameraGroup", None):
            self._unlink_camera(view.camera)
            panel_state.cameraGroup = group
            self._link_compatible_cameras()
        for i, axis_widget in enumerate(self._axisWidgets[panel_key]):
            axis_widget.visible = domain_.dim == 2
            if domain_.dim == 2:
                label = "XYZ"[(frame.displayAxes or [0, 1])[i]]
                if axis_widget.axis.axis_label != label:
                    axis_widget.axis.axis_label = label
                axis_widget.link_view(view)

        # Set on the marker path below; stays None on the grid path, where the
        # colorbar takes its range from the grid norm instead.
        fluidQs = boundaryQs = sharedNorm = None

        if opts.gridVisualization is not None:
            for marker in (panel_state.markers_fluid, panel_state.markers_boundary):
                if marker is not None:
                    marker.visible = False
            # ── grid update ────────────────────────────────────────────────────
            from ..grid import mapFrameToGrid, gridDisplayBounds  # noqa: PLC0415

            gridState, gridQuantity, nxs, gridExtent = mapFrameToGrid(frame, rotated_q)

            rgba_img, (ox, oy, dx, dy) = _grid_to_rgba_image(
                gridState, gridQuantity, nxs, gridExtent, opts, cmap
            )

            if panel_state.image_grid is not None:
                panel_state.image_grid.set_data(rgba_img)
                panel_state.image_grid.visible = True
                panel_state.image_grid.set_gl_state("translucent", depth_test=False)
                _position_image(panel_state.image_grid, ox, oy, dx, dy)
            else:
                panel_state.image_grid = visuals.Image(rgba_img, parent=view.scene)
                panel_state.image_grid.set_gl_state("translucent", depth_test=False)
                _position_image(panel_state.image_grid, ox, oy, dx, dy)

        else:
            if panel_state.image_grid is not None:
                panel_state.image_grid.visible = False
            # One norm across fluid AND boundary -- see `math.getSharedBounds`.
            fluidQs, boundaryQs, sharedNorm = frame.sharedBounds or _sharedMarkerBounds(
                fluid, boundary, opts
            )

            # ── fluid markers update ───────────────────────────────────────────
            if (
                opts.fluidVisualization != VisualizeOptions.Hide
                and fluid.positions.shape[0] > 0
            ):
                pts = _to_3d(fluid.positions)
                if opts.fluidVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(fluidQs, sharedNorm, cmap)

                if panel_state.markers_fluid is not None:
                    panel_state.markers_fluid.visible = True
                    panel_state.markers_fluid.scaling = marker_scaling
                    panel_state.markers_fluid.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=fluid_size,
                        edge_width=0,
                    )
                else:
                    panel_state.markers_fluid = visuals.Markers(
                        antialias=0, parent=view.scene
                    )
                    panel_state.markers_fluid.scaling = marker_scaling
                    panel_state.markers_fluid.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=fluid_size,
                        edge_width=0,
                    )
            elif panel_state.markers_fluid is not None:
                panel_state.markers_fluid.visible = False

            # ── boundary markers update ────────────────────────────────────────
            if (
                opts.boundaryVisualization != VisualizeOptions.Hide
                and boundary.positions.shape[0] > 0
            ):
                pts = _to_3d(boundary.positions)
                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(boundaryQs, sharedNorm, cmap)

                if panel_state.markers_boundary is not None:
                    panel_state.markers_boundary.visible = True
                    panel_state.markers_boundary.scaling = marker_scaling
                    panel_state.markers_boundary.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=boundary_size,
                        edge_width=0,
                    )
                else:
                    panel_state.markers_boundary = visuals.Markers(
                        antialias=0, parent=view.scene
                    )
                    panel_state.markers_boundary.scaling = marker_scaling
                    panel_state.markers_boundary.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=boundary_size,
                        edge_width=0,
                    )
            elif panel_state.markers_boundary is not None:
                panel_state.markers_boundary.visible = False

        panel_state.fluidParticles = fluid
        panel_state.boundaryParticles = boundary
        panel_state.ghostParticles = ghost
        panel_state.assembledQuantity = assembled
        panel_state.rotatedQuantities = rotated_q
        panel_state.options = opts

        # Update colorbar presence as well as range. All-passive/empty plots
        # have no scalar norm, so they must not access unassigned limits.
        new_range = None
        if opts.gridVisualization is not None:
            _, grid_norm = gridDisplayBounds(gridQuantity, gridExtent, opts)
            new_range = (
                (float(grid_norm.vmin), float(grid_norm.vmax))
                if grid_norm is not None
                else None
            )
        elif sharedNorm is not None:
            new_range = (float(sharedNorm.vmin), float(sharedNorm.vmax))
        cb = panel_state.colorbar
        if not opts.showColorBar or new_range is None:
            if cb is not None:
                self._subgrids[panel_key].remove_widget(cb)
                cb.parent = None
                # Text visuals own GL font resources. Keep one widget per
                # panel rather than allocating new atlases on every reveal.
                panel_state.cached_colorbar = cb
                panel_state.colorbar = None
        else:
            if cb is None:
                cb = panel_state.cached_colorbar
                if cb is None:
                    cb = _make_colorbar(cmap, new_range)
                panel_state.cached_colorbar = None
                self._subgrids[panel_key].add_widget(cb, row=1, col=2)
                panel_state.colorbar = cb
            last = self.__dict__.setdefault("_colorbarLast", {}).setdefault(
                panel_key, {}
            )
            signature = (id(cb), new_range, cmap)
            if last.get("signature") != signature:
                color_signature = (id(cb), cmap)
                if last.get("color_signature") != color_signature:
                    cb.cmap = _mpl_to_vispy_cmap(cmap)
                    last["color_signature"] = color_signature
                # Updating limits also binds the LUT: install a changed cmap
                # first so its texture is visible in this frame, not the next.
                cb.clim = tuple(f"{v:.6g}" for v in new_range)
                _style_colorbar_tick_labels(cb)
                last["signature"] = signature
        changed = view_needs_refit(
            old_domain, frame, getattr(panel_state, "displayAxes", None)
        )
        if changed:
            _fit_camera_to_domain(view, view_domain(frame), margin=opts.domainEpsilon)
        panel_state.displayAxes = tuple(frame.displayAxes or range(domain_.dim))
        panel_state.domain = domain_
        if opts.plotDomain:
            pts = (
                _domain_line_positions_2d(domain_)
                if domain_.dim == 2
                else _domain_line_positions_3d(domain_)
            )
            if panel_state.domain_lines is None:
                panel_state.domain_lines = visuals.Line(
                    pts,
                    color="blue",
                    connect="strip" if domain_.dim == 2 else "segments",
                    parent=view.scene,
                )
            else:
                panel_state.domain_lines.visible = True
                panel_state.domain_lines.set_data(pts)
            panel_state.domain_lines.order = 20
            panel_state.domain_lines.set_gl_state(
                "translucent", depth_test=domain_.dim == 3
            )
        elif panel_state.domain_lines is not None:
            panel_state.domain_lines.visible = False
        title = panel_state.title_visual
        title.text = opts.plotTitle or ""
        title.height_min = title.height_max = 26 if opts.plotTitle else 0
        self._update_streamlines(panel_state, frame)

        # Trigger a canvas redraw so the updated data is displayed
        self._canvas.update()

        return panel_state

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    @_serialized_canvas
    def export(self, filepath: str, **kwargs: Any) -> None:
        """Export the canvas to *filepath*.

        The canvas is rendered off-screen via :meth:`vispy.scene.SceneCanvas.render`
        and written to *filepath* using PIL (Pillow).  Any raster format
        supported by Pillow (``png``, ``jpg``, ``tiff``, …) is accepted.

        Parameters
        ----------
        filepath:
            Full path including extension.
        dpi:
            Dots per inch written into the image metadata (where supported by
            the format, e.g. PNG/TIFF).  Has no effect on actual pixel
            dimensions, which are determined by the canvas size.
        **kwargs:
            Remaining keyword arguments are forwarded to
            :meth:`PIL.Image.Image.save`.
        """
        if self._canvas is None:
            raise RuntimeError("No canvas to export; call create_figure first.")
        img_array = self._canvas.render()
        if self._refresh_axes():
            img_array = self._canvas.render()
        dpi = kwargs.pop("dpi", None)
        # `background=True`: encode + write on a worker thread (PIL releases
        # the GIL while compressing) so a simulation's step loop does not wait
        # on the PNG encoder; `asyncExport.waitForExports()` drains the queue
        # (call it before reading the files, e.g. before encoding a video).
        background = bool(kwargs.pop("background", False))
        try:
            import PIL.Image  # noqa: PLC0415

            img = PIL.Image.fromarray(img_array)
            if _os.path.splitext(str(filepath))[1].lower() in (".jpg", ".jpeg"):
                img = img.convert("RGB")
            save_kwargs = dict(kwargs)
            if dpi is not None:
                save_kwargs.setdefault("dpi", (dpi, dpi))
            if background:
                submitImageWrite(img, filepath, save_kwargs)
            else:
                img.save(filepath, **save_kwargs)
        except ImportError:
            # Pillow not available — fall back to vispy's built-in PNG writer
            from vispy.io import write_png  # noqa: PLC0415

            write_png(filepath, img_array)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    @_serialized_canvas
    def show(self) -> None:
        """Flush / display the canvas.

        First call
            Shows the canvas (native OS window or inline Jupyter widget).
        Subsequent calls
            For ``"native"`` mode: pumps the app event loop so pending redraws
            are processed immediately — required for live updates inside a
            synchronous ``for`` loop.
            For ``"notebook"`` mode: directly calls the jupyter_rfb draw
            pipeline without asyncio scheduling, so frames are pushed
            synchronously via ZMQ and appear in the widget mid-loop.
            (``nest_asyncio`` is intentionally avoided: it is broken in
            Python 3.13 due to tightened ``contextvars.Context.run``
            re-entrancy guards.)
        """
        if self._shown:
            # Subsequent call: flush pending redraws
            self._canvas.update()
            if self._jupyter_mode == "native":
                try:
                    self._ensure_native_window_visible()
                    self._canvas.app.process_events()
                except Exception:
                    pass
            elif self._jupyter_mode == "notebook":
                self._flush_notebook_frame()
            elif self._jupyter_mode == "image":
                self._flush_image_frame()
            return
        if self._jupyter_mode != "image":
            # "image" mode never has a window/widget to map -- rendering is
            # pure offscreen `canvas.render()`, which needs no prior `show()`.
            self._canvas.show()
        if self._jupyter_mode == "native":
            # Force the OS window to the foreground once so it reliably maps
            # on-screen before entering long synchronous simulation loops.
            self._ensure_native_window_visible()
            try:
                self._canvas.app.process_events()
            except Exception:
                pass
        if self._jupyter_mode == "notebook":
            try:
                from IPython.display import display as ip_display  # noqa: PLC0415

                ip_display(self._canvas)
            except Exception:
                pass
            # Push the first frame synchronously so the widget shows content
            # immediately rather than displaying an empty placeholder.
            self._flush_notebook_frame()
            # The browser reports the resulting logical/physical size through
            # the RFB resize event. Fixed pixel width clips bars in narrow cells.
            self._canvas._backend.css_width = "100%"
        if self._jupyter_mode == "image":
            self._flush_image_frame()
        self._shown = True

    @_serialized_canvas
    def update_figure_title(self, title: Optional[str]) -> None:
        """Update figure title text for both canvas content and native window."""
        text = "" if title is None else str(title)
        self._figure_title = text

        has_title = bool(text.strip())
        if self._figure_title_widget is not None:
            self._figure_title_widget.text = text
            # Grow the band with the line count. `_figure_title_height_px` (36)
            # was a fixed single-line allowance, so a multi-line title -- e.g.
            # warpSPH's provenance banner (state / physics / run identity) --
            # was drawn over the panel borders. 18 px per line + 18 px padding
            # reproduces the old 36 px for a one-line title exactly.
            nLines = text.count("\n") + 1 if has_title else 0
            perLine = self._figure_title_height_px // 2
            h = (perLine * nLines + perLine) if has_title else 0
            self._figure_title_widget.height_min = h
            self._figure_title_widget.height_max = h

        window_title = text if has_title else "warpSPHPlotting"
        try:
            self._canvas.title = window_title
        except Exception:
            pass

        try:
            native = getattr(self._canvas, "native", None)
            if native is not None and hasattr(native, "setWindowTitle"):
                native.setWindowTitle(window_title)
        except Exception:
            pass

        if self._canvas is not None:
            self._canvas.update()
        if self._shown and self._jupyter_mode == "notebook":
            self._flush_notebook_frame()

    @_serialized_canvas
    def close(self):
        if self._canvas is not None:
            try:
                self._canvas.close()
            finally:
                self._canvas_access.closed = True
                self._canvas_access.redraw_pending = False
                self._canvas_access.schedule_redraw = None
            self._states.clear()
            self._canvas = None

    @staticmethod
    def _unlink_camera(camera):
        # Vispy exposes link() but no unlink(). Remove both sides to release
        # retired cameras when a panel changes between planar and 3D views.
        for other in list(camera._linked_cameras):
            other._linked_cameras.pop(camera, None)
        camera._linked_cameras.clear()

    def _link_compatible_cameras(self):
        if not (self._sharex or self._sharey):
            return
        groups = {}
        for state in self._states.values():
            camera = state.view.camera
            group = state.cameraGroup
            if group[0] == 3 and not (self._sharex and self._sharey):
                continue  # Turntable cameras cannot link individual x/y limits.
            if group in groups:
                groups[group].link(
                    camera,
                    axis=(
                        None
                        if self._sharex and self._sharey
                        else ("x" if self._sharex else "y")
                    ),
                )
            else:
                groups[group] = camera

    def _refresh_axes(self):
        # AxisWidget listens to the scene transform, but mosaic widget
        # placement can settle after that signal. Refresh from the final
        # transforms so later panels never retain startup coordinates.
        changed = False
        for state in self._states.values():
            title = state.title_visual
            height = 26 if state.options.plotTitle else 0
            if state.options.plotTitle and state.options.plotTitleGap is not None:
                # Preserve the public gap's fraction-of-domain-height units
                # while reserving the title outside the data viewport.
                rect = getattr(state.view.camera, "_real_rect", None)
                if rect is not None and rect.height > 0:
                    domain_height = float(state.domain.max[1] - state.domain.min[1])
                    gap = (
                        state.options.plotTitleGap
                        * domain_height
                        * state.view.size[1]
                        / rect.height
                    )
                    height = 22 + max(0, round(gap))
            if title.height_min != height:
                title.height_min = title.height_max = height
                changed = True
        for axes in getattr(self, "_axisWidgets", {}).values():
            for axis in axes:
                if not axis.visible or axis._linked_view is None:
                    continue
                before = tuple(axis.axis.domain)
                axis._view_changed()
                changed |= not np.allclose(before, axis.axis.domain)
        return changed

    def _update_streamlines(self, state, frame):
        grid = frame.options.gridVisualization
        if grid is None or not grid.streamLines:
            if state.stream_visual is not None:
                state.stream_visual.visible = False
            return
        from ..grid import mapFrameToGrid
        from ..streamLines import streamlineSegments
        from vispy.scene.visuals import Line

        gs, gq, n, extent = mapFrameToGrid(frame, frame.streamlineQuantity)
        edges = streamlineSegments(gs, gq, n, extent).reshape(-1, 2)
        pts = np.column_stack((edges, np.zeros(len(edges), dtype=np.float32)))
        if state.stream_visual is None:
            state.stream_visual = Line(
                pts, color="black", connect="segments", parent=state.view.scene
            )
            state.stream_visual.order = 10
            state.stream_visual.set_gl_state("translucent", depth_test=False)
        else:
            state.stream_visual.visible = True
            state.stream_visual.set_data(pts)

    def _ensure_native_window_visible(self) -> None:
        """Best-effort native widget raise/activate for GUI backends."""
        try:
            native = getattr(self._canvas, "native", None)
            if native is None:
                return

            # Optional explicit placement used to avoid stale/off-screen window
            # coordinates when monitor layouts change between runs.
            if self._window_pos is not None and hasattr(native, "move"):
                native.move(*self._window_pos)

            # Safety net: move to the primary screen when requested, or when
            # the current top-left corner is outside all known screen rects.
            if hasattr(native, "geometry") and hasattr(native, "screen"):
                try:
                    from PyQt5.QtGui import QGuiApplication  # noqa: PLC0415
                except Exception:
                    QGuiApplication = None
                if QGuiApplication is not None:
                    g = native.geometry()
                    x, y = int(g.x()), int(g.y())
                    screens = QGuiApplication.screens() or []
                    in_any_screen = False
                    for s in screens:
                        sg = s.geometry()
                        if sg.x() <= x < (sg.x() + sg.width()) and sg.y() <= y < (
                            sg.y() + sg.height()
                        ):
                            in_any_screen = True
                            break
                    if self._force_primary_window or not in_any_screen:
                        ps = QGuiApplication.primaryScreen()
                        if ps is not None and hasattr(native, "move"):
                            pg = ps.availableGeometry()
                            native.move(int(pg.x()) + 40, int(pg.y()) + 40)

            if hasattr(native, "show"):
                native.show()
            if hasattr(native, "raise_"):
                native.raise_()
            if hasattr(native, "activateWindow"):
                native.activateWindow()
            if hasattr(native, "repaint"):
                native.repaint()
        except Exception:
            pass

    @_serialized_canvas
    def _flush_notebook_frame(self) -> None:
        """Push the current frame to the notebook widget synchronously.

        jupyter_rfb throttles rendering via a ``max_buffered_frames`` (= 2)
        check: if more frames are *in-flight* (sent but not yet ACK-ed by the
        browser) it skips drawing.  The browser can only ACK frames between
        kernel messages, so inside a synchronous ``for`` loop every frame
        after the second would be silently dropped.

        We bypass the throttle by calling ``get_frame()`` and
        ``_rfb_send_frame()`` directly. Browser acknowledgments remain intact;
        replacing them with a synthetic index breaks asynchronous RFB stats. The
        render and ZMQ send are fully synchronous; no asyncio is involved.

        On the very first call (before the browser has sent a resize event),
        ``get_frame()`` would return ``None`` because ``_physical_size`` is
        still ``(1, 1)``.  We pre-set it from ``self._canvas_size`` so the
        first frame is rendered at the correct resolution immediately after
        ``ip_display(canvas)`` is called.
        """
        backend = self._canvas._backend
        if not (hasattr(backend, "get_frame") and hasattr(backend, "_rfb_send_frame")):
            return
        # Pre-set physical size if the browser hasn't sent a resize yet.
        if getattr(backend, "_physical_size", (1, 1))[0] <= 1:
            px_w, px_h = self._canvas_size
            backend._physical_size = (px_w, px_h)
            # Keep logical and canvas sizes in sync so the first resize
            # event does not use a degenerate 1x1 viewbox layout.
            if hasattr(backend, "_logical_size"):
                backend._logical_size = (px_w, px_h)
            if hasattr(self._canvas, "size"):
                self._canvas.size = (px_w, px_h)
            if hasattr(backend, "_helper"):
                backend._helper.set_physical_size(px_w, px_h)
            if hasattr(backend, "_emit_resize_event"):
                backend._emit_resize_event()

        # If any panel still has a zero-sized viewbox, defer this frame.
        # A subsequent browser resize/layout event will provide valid
        # geometry and normal drawing can proceed without singular matrices.
        if any(
            _view_rect_is_degenerate(v) or _camera_rect_is_degenerate(v)
            for v in self._views.values()
        ):
            return

        with backend._output_context:
            array = backend.get_frame()
            if array is not None:
                backend._rfb_send_frame(array)

    @_serialized_canvas
    def _flush_image_frame(self) -> None:
        """Render offscreen and push the frame as a static Jupyter image.

        Same ``canvas.render()`` offscreen path :meth:`export` uses -- works
        with any GL context vispy can create (including a headless EGL/OSMesa
        one, see ``"image"`` mode in the module docstring), no window or
        widget required. The image is pushed through IPython's ordinary
        display-data update (``display(..., display_id=...)`` /
        ``DisplayHandle.update``), the same iopub message every notebook
        frontend renders -- unlike ``"notebook"`` mode's jupyter_rfb comm,
        nothing here depends on a live websocket/comm channel back to the
        browser.
        """
        try:
            from IPython.display import Image as _IPImage  # noqa: PLC0415
            from IPython.display import display as ip_display  # noqa: PLC0415

            array = self._canvas.render()
            if self._refresh_axes():
                array = self._canvas.render()
            try:
                import PIL.Image  # noqa: PLC0415

                buf = _io.BytesIO()
                PIL.Image.fromarray(array).save(buf, format="png")
                png_bytes = buf.getvalue()
            except ImportError:
                from vispy.io.image import _make_png  # noqa: PLC0415

                png_bytes = _make_png(array)
            image = _IPImage(data=png_bytes, format="png")
            if self._image_display_handle is None:
                self._image_display_handle = ip_display(image, display_id=True)
            else:
                self._image_display_handle.update(image)
        except Exception as exc:
            raise RuntimeError("Vispy image rendering failed") from exc

    # -------------------------------------------------------------------------
    # Capability flags
    # -------------------------------------------------------------------------

    @property
    def supports_streamlines(self) -> bool:
        return True

    @property
    def supports_grid(self) -> bool:
        return True

    @property
    def supports_notebook_inline(self) -> bool:
        # Static image display works independently of jupyter_rfb.
        return True

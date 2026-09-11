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

Current limitations (Phase 4)
------------------------------
- Streamlines are not yet supported (``supports_streamlines = False``).
- Axis tick-label widgets are not wired (vispy does not replicate
  matplotlib-style annotated axes).
- ``sharex`` / ``sharey`` camera linking between panels is not yet wired.
- No colour-bar widget; the norm/cmap used is the same as the matplotlib
  backend so scalar ranges are still deterministic.
- ``matplotlib`` must be installed for the colormap / norm pipeline
  (same implicit dependency as the pyvista backend).
"""

from __future__ import annotations

import contextlib
import io as _io
import os as _os
import sys as _sys
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
class VispyVisualizationState:
    """Per-panel rendering state kept by the Vispy backend."""

    canvas: Any                    # shared vispy.scene.SceneCanvas reference
    view: Any                      # ViewBox for this panel

    # Marker visuals; None if the particle type is hidden or absent
    markers_fluid: Optional[Any]
    markers_boundary: Optional[Any]

    # Line visual for the domain bounding box; None when plotDomain=False
    domain_lines: Optional[Any]

    # Image visual for grid mode; None when gridVisualization is off
    image_grid: Optional[Any]

    # ColorBarWidget placed to the right of the view; None if showColorBar=False
    colorbar: Optional[Any]

    domain: Any                    # DomainDescription (detached copy)
    options: PlottingOptions

    # Per-type particle states (for update bookkeeping)
    fluidParticles: Any
    boundaryParticles: Any
    ghostParticles: Any
    assembledQuantity: Any
    rotatedQuantities: Any


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
    return VispyColormap(lut)


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
    q_flat = np.asarray(q_clipped).ravel()  # guarantee 1-D → output is (N, 4)
    cmap = _get_cmap(cmap_name)
    normed = norm(q_flat)
    if hasattr(normed, "data"):   # masked array returned by some norms
        normed = normed.data
    normed = np.clip(normed, 0.0, 1.0)
    return cmap(normed).astype(np.float32)


def _cmap_name(options: PlottingOptions) -> str:
    return options.colorMap.value + ("_r" if options.flipColorMap else "")


_MPL_PTS2_TO_VISPY_PX = 2.0 * (1.0 / np.pi) ** 0.5 * (96.0 / 72.0)
"""Conversion factor from matplotlib scatter ``s`` (pts²) to vispy pixel diameter.

Derivation:
  diameter_pts = 2 * sqrt(s / π)
  diameter_px  = diameter_pts * (screen_dpi / 72)  # 96 DPI assumed
"""


def _point_size(
    options: PlottingOptions,
    default_size: float,
) -> float:
    """Resolve marker size in vispy pixel diameter.

    When ``options.markerSize`` is ``None``, return *default_size*.
    When explicitly set, convert from matplotlib scatter area units (pts²) to
    vispy pixel diameter so that the same ``PlottingOptions.markerSize`` value
    produces visually comparable results across backends.
    """
    if options.markerSize is None:
        return float(default_size)
    # matplotlib s = area in pts²; vispy size = diameter in px
    return max(1.0, float(options.markerSize) ** 0.5 * _MPL_PTS2_TO_VISPY_PX)


def _auto_point_size(
    domain_,
    n_particles: int,
    panel_w_px: float,
    panel_h_px: float,
) -> float:
    """Estimate a vispy pixel diameter so each particle covers roughly one cell.

    Computes the inter-particle spacing from the domain area and particle count,
    then converts to pixels using the panel size and domain extent.  Mirrors
    the logic of ``scatter_util.computeMarkerSize`` but avoids a matplotlib
    axes object.
    """
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    domain_w = max(float(mx[0] - mn[0]), 1e-9)
    domain_h = max(float(mx[1] - mn[1]), 1e-9)
    if n_particles > 1:
        # Estimated square-packing spacing
        dx = (domain_w * domain_h / n_particles) ** 0.5
    else:
        dx = domain_w / 32.0
    # Pixels per domain unit — use the more constrained axis
    px_per_unit = min(panel_w_px / domain_w, panel_h_px / domain_h)
    size_px = dx * px_per_unit * 0.8
    return max(2.0, float(size_px))


def _domain_line_positions_2d(domain_) -> np.ndarray:
    """Return (5, 3) float32 positions for a closed 2-D bounding-box rectangle."""
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    return np.array([
        [mn[0], mn[1], 0.0],
        [mx[0], mn[1], 0.0],
        [mx[0], mx[1], 0.0],
        [mn[0], mx[1], 0.0],
        [mn[0], mn[1], 0.0],   # close the loop
    ], dtype=np.float32)


def _domain_line_positions_3d(domain_) -> np.ndarray:
    """Return (24, 3) float32 positions for all 12 edges of a 3-D bbox.

    Arranged as 12 disjoint segments (use ``connect='segments'``).
    """
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])
    return np.array([
        # bottom face
        [x0, y0, z0], [x1, y0, z0],
        [x1, y0, z0], [x1, y1, z0],
        [x1, y1, z0], [x0, y1, z0],
        [x0, y1, z0], [x0, y0, z0],
        # top face
        [x0, y0, z1], [x1, y0, z1],
        [x1, y0, z1], [x1, y1, z1],
        [x1, y1, z1], [x0, y1, z1],
        [x0, y1, z1], [x0, y0, z1],
        # verticals
        [x0, y0, z0], [x0, y0, z1],
        [x1, y0, z0], [x1, y0, z1],
        [x1, y1, z0], [x1, y1, z1],
        [x0, y1, z0], [x0, y1, z1],
    ], dtype=np.float32)


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
    # For 3-D, allow vispy's default auto-fit; an explicit extent-based fit
    # is deferred as a Phase 4 follow-up.


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
    from ..math import getBounds  # noqa: PLC0415

    qs, norm = getBounds(gridQuantity, opts)
    nx, ny = nxs[0], nxs[1]

    # Reshape: ij-order (nx, ny) → transpose → (ny, nx) = (H, W)
    q_img = qs.reshape(nx, ny).T
    normed = norm(q_img)
    if hasattr(normed, "data"):
        normed = normed.data
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


def _style_colorbar_tick_labels(colorbar_widget: Any) -> None:
    """Apply custom tick styling for vispy ColorBarWidget.

    For vertical colorbars (orientation right/left), rotate ticks to read
    vertically and mirror the anchors so the upper-limit tick is right-
    aligned and the lower-limit tick is left-aligned.
    """
    if colorbar_widget is None:
        return
    try:
        cb = colorbar_widget._colorbar
        ticks = cb._ticks
        if cb.orientation in ("right", "left") and len(ticks) >= 2:
            ticks[0].rotation = -90
            ticks[1].rotation = -90
            ticks[1].pos = (float(ticks[0].pos[0]), float(ticks[0].pos[1]))

            # Vispy recomputes tick positions during layout; choose top/bottom
            # by y-position and set mirrored horizontal anchors.
            y0 = float(ticks[0].pos[1])
            y1 = float(ticks[1].pos[1])
            top_idx = 0 if y0 >= y1 else 1
            bottom_idx = 1 - top_idx

            # Use vertical top/bottom anchors so rotated labels are nudged
            # inward from the colorbar ends instead of straddling the limits.
            # ticks[top_idx].anchors = ("right", "top")
            # ticks[bottom_idx].anchors = ("left", "bottom")

            # Anchors alone are not always sufficient in vispy; explicitly
            # move labels inward by a few pixels for consistent placement.
            # inward_px = 160.0
            # top_pos = np.array(ticks[top_idx].pos, dtype=np.float32)
            # bot_pos = np.array(ticks[bottom_idx].pos, dtype=np.float32)
            # top_pos[1] -= inward_px
            # bot_pos[1] += inward_px
            # ticks[top_idx].pos = top_pos
            # ticks[bottom_idx].pos = bot_pos

            # Ensure the text visuals pick up the modified transforms.
            # ticks[top_idx].update()
            # ticks[bottom_idx].update()
            cb.update()
    except Exception:
        # Styling is best-effort; skip if vispy internals differ.
        pass


def _install_colorbar_style_hook(colorbar_widget: Any) -> None:
    """Install a post-resize hook so custom tick alignment persists."""
    if colorbar_widget is None or getattr(colorbar_widget, "_warpSPHPlotting_style_hook", False):
        return
    try:
        original_on_resize = colorbar_widget.on_resize

        def _on_resize_and_style(event):
            original_on_resize(event)
            _style_colorbar_tick_labels(colorbar_widget)

        colorbar_widget.on_resize = _on_resize_and_style
        colorbar_widget._warpSPHPlotting_style_hook = True
    except Exception:
        pass


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
    fluidQuantities = (fluid.quantities
                       if opts.fluidVisualization not in coloured
                       and fluid.positions.shape[0] > 0 else None)
    boundaryQuantities = (boundary.quantities
                          if opts.boundaryVisualization not in coloured
                          and boundary.positions.shape[0] > 0 else None)
    return getSharedBounds(fluidQuantities, boundaryQuantities, opts)


def _no_display_available() -> bool:
    """True on a Linux session with neither X11 nor Wayland reachable.

    The signal vispy's own GUI-toolkit probe (glfw/Qt/...) has no way to
    check for cheaply up front -- it just tries to open a window and raises.
    Non-Linux platforms are left alone: macOS/Windows native windowing
    doesn't depend on `DISPLAY`, and EGL is a Linux-specific escape hatch.
    """
    return (_sys.platform.startswith("linux")
            and not _os.environ.get("DISPLAY")
            and not _os.environ.get("WAYLAND_DISPLAY"))


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
        self._canvas: Any = None
        self._grid_widget: Any = None       # top-level vispy Grid
        self._subgrids: Dict[str, Any] = {}  # label → per-panel sub-Grid
        self._axes: Dict[str, Tuple[int, int]] = {}
        self._views: Dict[str, Any] = {}    # label → ViewBox (inside sub-grid col 0)
        self._figure_title_widget: Any = None
        self._figure_title_height_px: int = 36
        self._figure_title: str = ""
        self._jupyter_mode: str = "native"
        self._app_backend: Optional[str] = None
        self._image_display_handle: Any = None
        self._window_pos: Optional[Tuple[int, int]] = None
        self._force_primary_window: bool = False
        self._point_size: float = 10.0
        self._shown: bool = False
        self._canvas_size: Tuple[int, int] = (960, 480)  # (px_w, px_h)
        self._shape: Tuple[int, int] = (1, 1)            # (nrows, ncols)

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
        if (self._app_backend is None and self._jupyter_mode in ("native", "image")
                and _no_display_available()):
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

        self._point_size = float(opts.get("point_size", 10.0))
        self._shown = False
        self._figure_title = figTitle or ""

        shape, label_to_pos = _parse_mosaic(mosaic)
        self._axes = label_to_pos
        self._shape = shape

        px_w = int(figsize[0] * 96)
        px_h = int(figsize[1] * 96)
        self._canvas_size = (px_w, px_h)

        with _suppress_vispy_init_warnings():
            self._canvas = vs.SceneCanvas(
                title=figTitle or "warpSPHPlotting",
                size=(px_w, px_h),
                keys="interactive",
                show=False,
                bgcolor="white",
            )

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
        for label, (r, c) in label_to_pos.items():
            # Each mosaic panel gets a sub-grid so the colorbar column is
            # fully contained within the panel's allocated space.  Within the
            # sub-grid: col 0 = ViewBox, col 1 = ColorBarWidget slot (added
            # lazily in render_panel when showColorBar is True).
            sg = self._grid_widget.add_grid(row=r + 1, col=c, spacing=0)
            view = sg.add_view(row=0, col=0, bgcolor="white", border_color="gray")
            self._subgrids[label] = sg
            self._views[label] = view

        return self._canvas

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
    ) -> VispyVisualizationState:
        import vispy.scene.visuals as visuals  # noqa: PLC0415

        from ._render_util import prepare_particle_states  # noqa: PLC0415
        from ..math import getBounds  # noqa: PLC0415

        view = self._views[panel_key]

        # ── preprocessing ────────────────────────────────────────────────────
        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = prepare_particle_states(
            particleState, domain, quantity, options
        )

        cmap = _cmap_name(opts)

        # ── auto marker size ──────────────────────────────────────────────────
        nrows, ncols = self._shape
        pw, ph = self._canvas_size
        # Approximate panel pixel size (colorbars take ~65 px per column)
        cb_px = 65 if opts.showColorBar else 0
        panel_w_px = max(1.0, pw / max(ncols, 1) - cb_px)
        panel_h_px = max(1.0, ph / max(nrows, 1))
        n_total = int(fluid.positions.shape[0]) + int(boundary.positions.shape[0])
        if opts.markerSize is None:
            point_size = _auto_point_size(domain_, n_total, panel_w_px, panel_h_px)
        else:
            point_size = _point_size(opts, self._point_size)

        # ── camera ────────────────────────────────────────────────────────────
        view.camera = "panzoom" if domain_.dim == 2 else "turntable"
        if domain_.dim == 2:
            view.camera.aspect = 1.0  # enforce equal x/y scale

        # ── panel title ───────────────────────────────────────────────────────
        if opts.plotTitle is not None:
            mn = domain_.min.cpu().numpy()
            mx = domain_.max.cpu().numpy()
            cx = 0.5 * (float(mn[0]) + float(mx[0]))
            pad_frac = 0.04 if opts.plotTitleGap is None else float(opts.plotTitleGap)
            pad = (float(mx[1]) - float(mn[1])) * pad_frac
            visuals.Text(
                opts.plotTitle,
                color="black",
                font_size=10,
                pos=(cx, float(mx[1]) + pad, 0.0),
                anchor_x="center",
                anchor_y="bottom",
                parent=view.scene,
            )

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

        # ── grid or scatter path ───────────────────────────────────────────────
        markers_fluid, markers_boundary, image_grid = None, None, None
        # Set on the marker path below; stays None on the grid path, where
        # the colorbar takes its range from the grid norm instead.
        fluidQs = boundaryQs = sharedNorm = None

        if opts.gridVisualization is not None:
            # ── grid (Image visual) ───────────────────────────────────────────
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

            rgba_img, (ox, oy, dx, dy) = _grid_to_rgba_image(
                gridState, gridQuantity, nxs, gridExtent, opts, cmap
            )
            image_grid = visuals.Image(rgba_img, parent=view.scene)
            image_grid.transform = _image_transform(ox, oy, dx, dy)

        else:
            # One norm across fluid AND boundary -- see `math.getSharedBounds`.
            # Normalising them separately and drawing both through the same
            # colormap makes wall colours incomparable to fluid colours.
            fluidQs, boundaryQs, sharedNorm = _sharedMarkerBounds(fluid, boundary, opts)

            # ── fluid markers ─────────────────────────────────────────────────
            if opts.fluidVisualization != VisualizeOptions.Hide and fluid.positions.shape[0] > 0:
                pts = _to_3d(fluid.positions)
                if opts.fluidVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(fluidQs, sharedNorm, cmap)
                markers_fluid = visuals.Markers(antialias=0, parent=view.scene)
                markers_fluid.set_data(
                    pts,
                    face_color=face_color,
                    edge_color=face_color,
                    size=point_size,
                    edge_width=0,
                )

            # ── boundary markers ──────────────────────────────────────────────
            if opts.boundaryVisualization != VisualizeOptions.Hide and boundary.positions.shape[0] > 0:
                pts = _to_3d(boundary.positions)
                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(boundaryQs, sharedNorm, cmap)
                markers_boundary = visuals.Markers(antialias=0, parent=view.scene)
                markers_boundary.set_data(
                    pts,
                    face_color=face_color,
                    edge_color=face_color,
                    size=point_size,
                    edge_width=0,
                )

        # ── camera fit ────────────────────────────────────────────────────────
        _fit_camera_to_domain(view, domain_)

        # ── colorbar ──────────────────────────────────────────────────────────
        colorbar_widget = None
        if opts.showColorBar:
            from vispy.scene.widgets.colorbar import ColorBarWidget  # noqa: PLC0415

            # Determine which scalar range and cmap to use for the colorbar.
            # Prefer the fluid (active) range; fall back to grid range.
            from ..math import getBounds  # noqa: PLC0415

            cb_range: Optional[Tuple[float, float]] = None
            cb_cmap = cmap
            if opts.gridVisualization is not None:
                # Derive range from the grid quantity's norm so the colorbar
                # always reflects the actual data limits, not the normalised
                # [0, 1] interval.
                if image_grid is not None:
                    _, grid_norm = getBounds(gridQuantity, opts)
                    cb_range = (float(grid_norm.vmin), float(grid_norm.vmax))
            elif sharedNorm is not None:
                # The same norm both marker sets were coloured with, so the bar
                # describes every particle on screen rather than just the fluid.
                cb_range = (float(sharedNorm.vmin), float(sharedNorm.vmax))

            if cb_range is not None:
                vmin, vmax = cb_range
                colorbar_widget = ColorBarWidget(
                    cmap=_mpl_to_vispy_cmap(cb_cmap),
                    orientation="right",
                    label="",
                    label_color="black",
                    clim=(f"{vmin:.6g}", f"{vmax:.6g}"),
                    # padding = (0.5,0)
                )
                # Add colorbar into the panel's sub-grid at col 1.
                # Fix its width so the view (col 0) gets all remaining space.
                self._subgrids[panel_key].add_widget(colorbar_widget, row=0, col=1)
                colorbar_widget.width_min = 65
                colorbar_widget.width_max = 65
                _install_colorbar_style_hook(colorbar_widget)
                # Rotate tick labels so they read vertically (-90° CCW, same
                # as the main colorbar label).  Horizontal numbers like
                # "0.0123" are wider than the 65-px column; rotated they fit
                # comfortably.
                _style_colorbar_tick_labels(colorbar_widget)
                colorbar_widget.update()

        return VispyVisualizationState(
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
        )

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

        from ._render_util import prepare_particle_states  # noqa: PLC0415
        from ..math import getBounds  # noqa: PLC0415

        # Apply any option overrides passed as keyword arguments
        if kwargs:
            options = _copy.deepcopy(options)
            for k, v in kwargs.items():
                if hasattr(options, k):
                    setattr(options, k, v)

        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = prepare_particle_states(
            particleState, domain, quantity, options
        )

        cmap = _cmap_name(opts)

        # ── auto marker size ──────────────────────────────────────────────────
        nrows, ncols = self._shape
        pw, ph = self._canvas_size
        cb_px = 65 if opts.showColorBar else 0
        panel_w_px = max(1.0, pw / max(ncols, 1) - cb_px)
        panel_h_px = max(1.0, ph / max(nrows, 1))
        n_total = int(fluid.positions.shape[0]) + int(boundary.positions.shape[0])
        if opts.markerSize is None:
            point_size = _auto_point_size(domain_, n_total, panel_w_px, panel_h_px)
        else:
            point_size = _point_size(opts, self._point_size)

        view = panel_state.view

        # Set on the marker path below; stays None on the grid path, where the
        # colorbar takes its range from the grid norm instead.
        fluidQs = boundaryQs = sharedNorm = None

        if opts.gridVisualization is not None:
            # ── grid update ────────────────────────────────────────────────────
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

            rgba_img, (ox, oy, dx, dy) = _grid_to_rgba_image(
                gridState, gridQuantity, nxs, gridExtent, opts, cmap
            )

            if panel_state.image_grid is not None:
                panel_state.image_grid.set_data(rgba_img)
            else:
                panel_state.image_grid = visuals.Image(rgba_img, parent=view.scene)
                panel_state.image_grid.transform = _image_transform(ox, oy, dx, dy)

        else:
            # One norm across fluid AND boundary -- see `math.getSharedBounds`.
            fluidQs, boundaryQs, sharedNorm = _sharedMarkerBounds(fluid, boundary, opts)

            # ── fluid markers update ───────────────────────────────────────────
            if opts.fluidVisualization != VisualizeOptions.Hide and fluid.positions.shape[0] > 0:
                pts = _to_3d(fluid.positions)
                if opts.fluidVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(fluidQs, sharedNorm, cmap)

                if panel_state.markers_fluid is not None:
                    panel_state.markers_fluid.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
                        edge_width=0,
                    )
                else:
                    panel_state.markers_fluid = visuals.Markers(antialias=0, parent=view.scene)
                    panel_state.markers_fluid.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
                        edge_width=0,
                    )
            elif panel_state.markers_fluid is not None:
                panel_state.markers_fluid.visible = False

            # ── boundary markers update ────────────────────────────────────────
            if opts.boundaryVisualization != VisualizeOptions.Hide and boundary.positions.shape[0] > 0:
                pts = _to_3d(boundary.positions)
                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    face_color = _scalars_to_rgba(boundaryQs, sharedNorm, cmap)

                if panel_state.markers_boundary is not None:
                    panel_state.markers_boundary.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
                        edge_width=0,
                    )
                else:
                    panel_state.markers_boundary = visuals.Markers(antialias=0, parent=view.scene)
                    panel_state.markers_boundary.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
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

        # ── update colorbar clim / cmap ────────────────────────────────────────
        if panel_state.colorbar is not None and opts.showColorBar:
            from vispy.color import Colormap as VispyColormap  # noqa: PLC0415

            new_range: Optional[Tuple[float, float]] = None
            if opts.gridVisualization is not None:
                # gridQuantity was computed in the grid update block above
                _, grid_norm = getBounds(gridQuantity, opts)
                new_range = (float(grid_norm.vmin), float(grid_norm.vmax))
            elif sharedNorm is not None:
                # The same norm both marker sets were coloured with, so the bar
                # describes every particle on screen rather than just the fluid.
                new_range = (float(sharedNorm.vmin), float(sharedNorm.vmax))
            if new_range is not None:
                vmin, vmax = new_range
                # 6 significant digits, not 3: a density plot sits near 1.0 and
            # `.3g` renders both ends of [0.997, 1.003] as "1".
            panel_state.colorbar.clim = (f"{vmin:.6g}", f"{vmax:.6g}")
            panel_state.colorbar.cmap = _mpl_to_vispy_cmap(cmap)
            _style_colorbar_tick_labels(panel_state.colorbar)

        # -- update camera to fit new domain
        _fit_camera_to_domain(view, domain_)
        # -- update domain bounding box
        if panel_state.domain_lines is not None and opts.plotDomain:
            if domain_.dim == 2:
                line_pts = _domain_line_positions_2d(domain_)
                panel_state.domain_lines.set_data(line_pts)
            else:
                line_pts = _domain_line_positions_3d(domain_)
                panel_state.domain_lines.set_data(line_pts)
        # -- adjust panel title position if plotTitleGap changed or domain changed
        if opts.plotTitle is not None:
            mn = domain_.min.cpu().numpy()
            mx = domain_.max.cpu().numpy()
            cx = 0.5 * (float(mn[0]) + float(mx[0]))
            pad_frac = 0.04 if opts.plotTitleGap is None else float(opts.plotTitleGap)
            pad = (float(mx[1]) - float(mn[1])) * pad_frac
            # Find the existing title text visual and update its position
            for child in view.scene.children:
                if isinstance(child, visuals.Text) and child.text == opts.plotTitle:
                    child.pos = (cx, float(mx[1]) + pad, 0.0)
                    break

        # Trigger a canvas redraw so the updated data is displayed
        self._canvas.update()

        return panel_state

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

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
        dpi = kwargs.pop("dpi", None)
        try:
            import PIL.Image  # noqa: PLC0415
            img = PIL.Image.fromarray(img_array)
            save_kwargs = dict(kwargs)
            if dpi is not None:
                save_kwargs.setdefault("dpi", (dpi, dpi))
            img.save(filepath, **save_kwargs)
        except ImportError:
            # Pillow not available — fall back to vispy's built-in PNG writer
            from vispy.io import write_png  # noqa: PLC0415
            write_png(filepath, img_array)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

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
        if self._jupyter_mode == "image":
            self._flush_image_frame()
        self._shown = True

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

    def update_domain(self, newDomain: Any) -> None:
        """Update the domain for all panels.

        This is a best-effort update: if the new domain is incompatible with
        the existing particle states (e.g. different dimensionality), the
        update is skipped and a warning is logged.
        """
        from ..math import getBounds  # noqa: PLC0415
        import vispy.scene.visuals as visuals  # noqa: PLC0415

        try:
            for panel_key, view in self._views.items():
                print(f"Updating domain for panel '{panel_key}' to {newDomain}")
                # Get the current panel state to access its options and particles
                panel_state = getattr(self, "_views", {}).get(panel_key, None)
                if panel_state is None:
                    print(f"Warning: No panel state found for '{panel_key}'; skipping domain update.")
                    continue

                # Check if the new domain is compatible with the existing particles
                if newDomain.dim != panel_state.domain.dim:
                    print(f"Warning: Incompatible domain dimension for panel '{panel_key}'; skipping domain update.")
                    continue

                # Update the domain in the panel state
                panel_state.domain = newDomain
                print(f"Updated domain for panel '{panel_key}' to {newDomain}")

                # Recompute bounds and update camera
                _fit_camera_to_domain(view, newDomain)

                # Trigger a canvas redraw so the updated domain is displayed
                self._canvas.update()
                

            # if newDomain.dim == 2:
                
            #     line_pts = _domain_line_positions_2d(newDomain)
            #     self.domain_lines = visuals.Line(
            #         line_pts, color="blue", connect="strip", parent=view.scene
            #     )
        except Exception:
            pass

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
                        if sg.x() <= x < (sg.x() + sg.width()) and sg.y() <= y < (sg.y() + sg.height()):
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

    def _flush_notebook_frame(self) -> None:
        """Push the current frame to the notebook widget synchronously.

        jupyter_rfb throttles rendering via a ``max_buffered_frames`` (= 2)
        check: if more frames are *in-flight* (sent but not yet ACK-ed by the
        browser) it skips drawing.  The browser can only ACK frames between
        kernel messages, so inside a synchronous ``for`` loop every frame
        after the second would be silently dropped.

        We bypass the throttle by calling ``get_frame()`` and
        ``_rfb_send_frame()`` directly and resetting ``_frame_feedback`` to
        pretend all previous frames were acknowledged before each call.  The
        render and ZMQ send are fully synchronous; no asyncio is involved.

        On the very first call (before the browser has sent a resize event),
        ``get_frame()`` would return ``None`` because ``_physical_size`` is
        still ``(1, 1)``.  We pre-set it from ``self._canvas_size`` so the
        first frame is rendered at the correct resolution immediately after
        ``ip_display(canvas)`` is called.
        """
        try:
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

            # Pretend all in-flight frames were acknowledged so the throttle
            # never blocks us.
            backend._frame_feedback = {"index": backend._rfb_frame_index}
            with backend._output_context:
                array = backend.get_frame()
                if array is not None:
                    backend._rfb_send_frame(array)
        except Exception:
            pass

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
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Capability flags
    # -------------------------------------------------------------------------

    @property
    def supports_streamlines(self) -> bool:
        # TODO (Phase 4 follow-up): implement with vispy Line visuals + seed
        # integration; leave False until that work lands.
        return False

    @property
    def supports_grid(self) -> bool:
        return True

    @property
    def supports_notebook_inline(self) -> bool:
        try:
            import jupyter_rfb  # noqa: F401

            return True
        except ImportError:
            return False

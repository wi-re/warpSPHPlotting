"""Backend factory for warpSPHPlotting.

Call :func:`get_backend` with a backend name (or :class:`~warpSPHPlotting.Backend`
enum member) to receive a fully initialised backend instance ready for use
by :func:`~warpSPHPlotting.visualize`.

Adding a new backend
--------------------
1. Create ``src/warpSPHPlotting/backends/<name>_backend.py`` subclassing
   :class:`~warpSPHPlotting.backends.base.AbstractBackend`.
2. Add a branch in :func:`get_backend` below.
3. Add the corresponding extras entry in ``pyproject.toml``.
"""

from typing import Optional
from importlib.util import find_spec
import warnings

from .base import AbstractBackend

# Canonical string names (lower-cased) recognised by the factory.
_MATPLOTLIB = "matplotlib"
_PYVISTA = "pyvista"
_VISPY = "vispy"

_KNOWN_BACKENDS = {_MATPLOTLIB, _PYVISTA, _VISPY}

# Count rendered particles across panels, rather than particles hidden by options
# or used only for grid interpolation. This is a routing heuristic, not a cap.
AUTO_PARTICLE_THRESHOLD = 50_000


def select_backend(backend, frames, sharex=True, sharey=True):
    """Resolve ``auto`` once, before allocating any graphics resources.

    Small 2D plots retain Matplotlib. Prefer PyVista for native 3D or large
    scatter plots, then Vispy where its capabilities suffice. Explicit backend
    choices are never replaced, and an existing figure never changes backend.
    """
    from ..enumTypes import VisualizeOptions

    name = str(getattr(backend, "value", backend)).lower()
    if name != "auto":
        return name
    frames = list(frames)
    native_3d = any(f.domain.dim == 3 for f in frames)
    volume = any(f.domain.dim == 3 and f.options.gridVisualization for f in frames)
    partial_sharing = bool(sharex) != bool(sharey)
    if volume and partial_sharing:
        raise ValueError(
            "Native 3D grids require PyVista, which links whole cameras; "
            "set sharex and sharey both True or both False."
        )
    count = sum(
        len(getattr(f, kind).positions)
        for f in frames
        if f.options.gridVisualization is None
        for kind, variant in (
            ("fluid", f.options.fluidVisualization),
            ("boundary", f.options.boundaryVisualization),
        )
        if variant != VisualizeOptions.Hide
    )
    if not native_3d and count < AUTO_PARTICLE_THRESHOLD:
        return _MATPLOTLIB
    if not partial_sharing and find_spec(_PYVISTA) is not None:
        return _PYVISTA
    if not volume and find_spec(_VISPY) is not None:
        return _VISPY
    if native_3d:
        raise ImportError(
            "Automatic 3D rendering requires PyVista"
            + (
                " with compatible camera sharing."
                if volume
                else " or Vispy with compatible camera sharing."
            )
            + " Install with: pip install 'warpSPHPlotting[plot-pyvista]'"
        )
    warnings.warn(
        f"Rendering {count:,} particles with Matplotlib because no compatible "
        "accelerated backend is available. Install 'warpSPHPlotting[plot-vispy]' "
        "or use 'warpSPHPlotting[plot-pyvista]' with matching sharex/sharey for "
        "large particle plots.",
        RuntimeWarning,
        stacklevel=3,
    )
    return _MATPLOTLIB


def get_backend(backend: str, backendOptions: Optional[dict] = None) -> AbstractBackend:
    """Return an ``AbstractBackend`` instance for *backend*.

    Parameters
    ----------
    backend:
        One of ``"matplotlib"``, ``"pyvista"``, or ``"vispy"``.
        A :class:`~warpSPHPlotting.Backend` enum value is also accepted (it
        compares equal to its string value because the enum inherits ``str``).
    backendOptions:
        Forwarded verbatim to the backend; currently unused by the factory
        itself but reserved for future use (e.g. choosing a pyvista renderer).

    Raises
    ------
    ValueError
        If *backend* is not one of the recognised names.
    ImportError
        If the optional dependency for the chosen backend is not installed.
        The error message includes the corresponding ``pip install`` hint.
    """
    name = str(getattr(backend, "value", backend)).lower()

    if name not in _KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'.  "
            f"Choose one of: {sorted(_KNOWN_BACKENDS)}."
        )

    if name == _MATPLOTLIB:
        from .matplotlib_backend import MatplotlibBackend  # noqa: PLC0415

        return MatplotlibBackend()

    if name == _PYVISTA:
        try:
            import pyvista  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The pyvista backend requires pyvista (and optionally trame).  "
                "Install it with:  pip install 'warpSPHPlotting[plot-pyvista]'"
            ) from exc
        from .pyvista_backend import PyVistaBackend  # noqa: PLC0415

        return PyVistaBackend()

    if name == _VISPY:
        try:
            import vispy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The vispy backend requires vispy (and optionally jupyter_rfb).  "
                "Install it with:  pip install 'warpSPHPlotting[plot-vispy]'"
            ) from exc
        from .vispy_backend import VispyBackend  # noqa: PLC0415

        return VispyBackend()

    # Unreachable – kept as a safety net.
    raise ValueError(f"Unhandled backend name '{name}'.")  # pragma: no cover

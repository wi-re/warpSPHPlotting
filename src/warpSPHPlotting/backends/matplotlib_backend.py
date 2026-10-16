"""Matplotlib rendering backend.

Wraps the existing ``visualizeParticlesNew`` / ``updatePlot`` functions so
that the rest of the library can treat matplotlib as just one of several
interchangeable backends.

No new rendering logic lives here — all heavy lifting stays in
:mod:`warpSPHPlotting.visualize` and :mod:`warpSPHPlotting.update`.  The late imports
inside each method prevent circular-import issues while keeping this module
importable even if matplotlib is not installed (the error surfaces only when
a method is actually called).
"""

from typing import Any, Dict, Optional, Tuple

from .base import AbstractBackend


class MatplotlibBackend(AbstractBackend):
    """Backend that delegates to the matplotlib rendering pipeline."""

    def __init__(self) -> None:
        self._fig: Any = None
        self._axes: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Figure management
    # ------------------------------------------------------------------

    def create_figure(
        self,
        mosaic: str,
        figsize: Tuple[float, float],
        sharex: bool,
        sharey: bool,
        figTitle: Optional[str],
        backendOptions: Optional[dict],
    ) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "The matplotlib backend requires matplotlib.  "
                "Install it with:  pip install 'warpSPHPlotting[plot-matplotlib]'"
            ) from exc

        opts = dict(backendOptions or {})
        if not opts.keys() & {"layout", "constrained_layout", "tight_layout"}:
            opts["layout"] = "constrained"
        self._fig, self._axes = plt.subplot_mosaic(
            mosaic,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            **{
                k: v
                for k, v in opts.items()
                if k not in ("figsize", "sharex", "sharey")
            },
        )
        if figTitle is not None:
            self._fig.suptitle(figTitle, fontsize=16)
        return self._fig

    def get_axes(self) -> Dict[str, Any]:
        return self._axes

    # ------------------------------------------------------------------
    # Per-panel rendering
    # ------------------------------------------------------------------

    def render_panel(
        self,
        panel_key: str,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: Any,
        _preparedFrame=None,
    ) -> Any:
        # Late import avoids circular dependency:
        # visualize.py → backends.factory → matplotlib_backend → visualize (here)
        from ..visualize import visualizeParticlesNew  # noqa: PLC0415

        return visualizeParticlesNew(
            self._fig,
            self._axes[panel_key],
            particleState=particleState,
            domain=domain,
            quantity=quantity,
            options=options,
            _preparedFrame=_preparedFrame,
        )

    def update_panel(
        self,
        panel_key: str,
        panel_state: Any,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: Any,
        **kwargs: Any,
    ) -> Any:
        from ..update import updatePlot  # noqa: PLC0415

        return updatePlot(
            panel_state,
            particles=particleState,
            domain=domain,
            quantity=quantity,
            options=options,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show(self) -> None:
        if self._fig is not None:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def update_figure_title(self, title: Optional[str]) -> None:
        if self._fig is None:
            return
        self._fig.suptitle("" if title is None else title, fontsize=16)
        import matplotlib.pyplot as plt

        if plt.isinteractive():
            self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, filepath: str, **kwargs: Any) -> None:
        """Save the figure to *filepath*.

        The format is inferred from the file extension (``png``, ``pdf``,
        ``svg``, ``eps``, …).  All *kwargs* are forwarded verbatim to
        :meth:`matplotlib.figure.Figure.savefig`, so ``dpi``,
        ``transparent``, ``bbox_inches``, etc. are all supported.
        """
        if self._fig is None:
            raise RuntimeError("No figure to export; call create_figure first.")
        self._fig.savefig(filepath, **kwargs)

    def close(self):
        import matplotlib.pyplot as plt

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_streamlines(self) -> bool:
        return True

    @property
    def supports_grid(self) -> bool:
        return True

    @property
    def supports_notebook_inline(self) -> bool:
        return True

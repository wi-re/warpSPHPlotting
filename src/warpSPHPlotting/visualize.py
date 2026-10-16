"""Public visualization API and backend-neutral update dispatch."""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from warpSPHCore import DomainDescription
from .options import PlottingOptions
from .state import VisualizationState
from .update import updatePlot


def visualizeParticlesNew(
    fig,
    axis,
    particleState,
    domain,
    quantity,
    options=None,
    verbose=False,
    _preparedFrame=None,
    **kwargs,
):
    from .backends._render_util import prepare_frame
    from .update import _renderFrame

    opts = copy.deepcopy(options if options is not None else PlottingOptions())
    for key, value in kwargs.items():
        if not hasattr(opts, key):
            raise TypeError(f"Unknown plotting option: {key}")
        setattr(opts, key, value)
    frame = _preparedFrame or prepare_frame(particleState, domain, quantity, opts)
    state = VisualizationState(fig, axis, frame.domain, frame.options, frame.assembled)
    result = _renderFrame(state, frame, initial=True)
    result._sourceDomain = domain
    return result


def _yield_notebook_events(seconds: float = 0.0) -> None:
    """Yield control to the notebook event loop from sync code.

    This keeps widget-based backends (e.g. pyvista+trame) responsive inside
    plain ``for`` loops without forcing users to write async notebook cells.
    """
    delay = max(float(seconds), 0.0)

    try:
        from IPython import get_ipython  # noqa: PLC0415

        ip = get_ipython()
        in_notebook = ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        in_notebook = False

    if not in_notebook:
        return

    try:
        import asyncio  # noqa: PLC0415

        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio  # noqa: PLC0415

                nest_asyncio.apply(loop)
                loop.run_until_complete(asyncio.sleep(delay))
            except Exception:
                # Last resort: pause briefly if we cannot re-enter the loop.
                if delay > 0.0:
                    time.sleep(delay)
        else:
            loop.run_until_complete(asyncio.sleep(delay))
    except Exception:
        if delay > 0.0:
            time.sleep(delay)


@dataclass
class PlotState:
    fig: Any  # backend-specific figure handle; inspect backend for the resolved choice
    axes: Dict[str, Any]  # backend-specific per-panel handles
    domain: DomainDescription
    options: PlottingOptions
    quantities: Union[torch.Tensor, Dict[str, torch.Tensor]]
    particleState: Any
    mosaic: str
    sharex: bool
    sharey: bool
    figTitle: Optional[str]

    plotStates: Dict[str, VisualizationState]

    backend: str = "matplotlib"
    backendOptions: Optional[dict] = None
    _backend_instance: Any = field(default=None, repr=False, compare=False)
    _update_counter: int = field(default=0, repr=False, compare=False)

    def updateTitle(self, newTitle: str):
        self.figTitle = newTitle
        if self._backend_instance is not None:
            self._backend_instance.update_figure_title(newTitle)
            return

        # Fallback for legacy PlotState objects created without a backend.
        if hasattr(self.fig, "suptitle"):
            self.fig.suptitle(newTitle, fontsize=16)

    def updateDomain(self, newDomain):
        self.updateQuantities(self.quantities, newDomain=newDomain)

    def close(self):
        if self._backend_instance is not None:
            self._backend_instance.close()

    def show(self) -> None:
        """Display or re-display the figure.

        For the matplotlib backend this is equivalent to calling
        ``fig.tight_layout()`` and is typically a no-op after the initial
        ``visualize()`` call.

        For the pyvista *static* backend, this re-captures an offscreen
        screenshot and displays it inline in the current Jupyter cell —
        useful for showing the updated scene after
        ``updateQuantities(...)``.

        For pyvista *trame* and the pop-out window modes, the scene updates
        are reflected live and this call is also a no-op.
        """
        if self._backend_instance is not None:
            self._backend_instance.show()

    def export(self, filepath: str, **kwargs) -> None:
        """Export the current visualization to *filepath*.

        The file format is inferred from the extension (e.g. ``".png"``,
        ``".pdf"``, ``".svg"``).

        Parameters
        ----------
        filepath:
            Full path including extension.
        dpi:
            Dots per inch (supported by the matplotlib and vispy backends).
        **kwargs:
            Additional keyword arguments forwarded verbatim to the active
            backend.  Common options:

            * ``transparent`` / ``transparent_background`` — transparent
              background (matplotlib / pyvista-raster).
            * ``bbox_inches`` — e.g. ``"tight"`` (matplotlib only).

        Raises
        ------
        NotImplementedError
            If the active backend does not support file export.
        RuntimeError
            If no backend instance is available.
        """
        if self._backend_instance is None:
            raise RuntimeError("No backend instance available.")
        self._backend_instance.export(filepath, **kwargs)

    def updateQuantities(
        self,
        newQuantities: Union[torch.Tensor, Dict[str, torch.Tensor]],
        key: Optional[str] = None,
        newParticleState: Optional[Any] = None,
        newDomain: Optional[DomainDescription] = None,
        newOptions: Optional[Dict[str, Any]] = None,
        redraw: bool = True,
        redrawEvery: int = 1,
        yieldNotebookEvents: bool = False,
        yieldSeconds: float = 0.02,
        resetView: bool = False,
        **kwargs,
    ):
        """Update data, retaining the camera unless its domain changes.

        Use resetView=True for a new case or a one-off refit. Set the plotting
        option dynamicBounds=True to follow visible particle bounds each frame.
        """
        from .state import snapshotParticleState, snapshotQuantity

        pending_domain = newDomain if newDomain is not None else self.domain
        pending_particles = (
            snapshotParticleState(newParticleState)
            if newParticleState is not None
            else self.particleState
        )
        if key is not None:
            if key not in self.plotStates:
                raise KeyError(f"Unknown panel: {key}")
            replacements = {
                key: (
                    newQuantities[key]
                    if isinstance(newQuantities, dict)
                    else newQuantities
                )
            }
        elif isinstance(newQuantities, dict):
            replacements = dict(newQuantities)
        else:
            replacements = {k: newQuantities for k in self.plotStates}
        unknown = replacements.keys() - self.plotStates.keys()
        if unknown:
            raise KeyError(f"Unknown panels: {sorted(unknown)}")
        quantities = {
            k: (
                self.quantities[k]
                if isinstance(self.quantities, dict)
                else self.quantities
            )
            for k in self.plotStates
        }
        quantities.update(
            {key: snapshotQuantity(value) for key, value in replacements.items()}
        )
        # Geometry changes affect all panels, even with a partial quantity update.
        affected = (
            list(self.plotStates)
            if newDomain is not None or newParticleState is not None
            else list(replacements)
        )
        from .backends._render_util import prepare_frame

        prepared_options = {}
        frames = {}
        spacing_cache = {}
        for k in affected:
            opts = copy.deepcopy(self.plotStates[k].options)
            for name, value in dict((newOptions or {}).get(k, {}), **kwargs).items():
                if not hasattr(opts, name):
                    raise TypeError(f"Unknown plotting option: {name}")
                setattr(opts, name, value)
            # Validate every requested panel before mutating renderer state.
            frames[k] = prepare_frame(
                pending_particles,
                pending_domain,
                quantities[k],
                opts,
                _spacingCache=spacing_cache,
            )
            self._backend_instance.validate_frame(frames[k])
            frames[k].resetView = resetView
            prepared_options[k] = opts
        for k in affected:
            self.plotStates[k] = self._backend_instance.update_panel(
                k,
                self.plotStates[k],
                particleState=pending_particles,
                domain=pending_domain,
                quantity=quantities[k],
                options=prepared_options[k],
                _preparedFrame=frames[k],
            )
        self.domain, self.particleState, self.quantities = (
            pending_domain,
            pending_particles,
            quantities,
        )
        self.options = {key: state.options for key, state in self.plotStates.items()}

        # Flush the display after all panels are updated.
        # For pyvista-static this captures a new screenshot; for trame it
        # calls render(); for matplotlib it calls tight_layout() (no-op).
        if self._backend_instance is not None and redraw:
            n = max(int(redrawEvery), 1)
            self._update_counter += 1
            if self._update_counter % n == 0:
                self._backend_instance.show()

        if yieldNotebookEvents or self.backend == "pyvista":
            _yield_notebook_events(yieldSeconds)
        # if self.backend == 'matplotlib':
        #     self.fig.canvas.draw()
        #     self.fig.canvas.flush_events()

    def updatePlot(
        self,
        key: Union[str, List[str]],
        newOptions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if isinstance(key, list):
            for k in key:
                self.updatePlot(k, newOptions, **kwargs)
            return

        panel_state = self.plotStates[key]
        particleState = self.particleState
        quantity = (
            self.quantities[key]
            if isinstance(self.quantities, dict)
            else self.quantities
        )

        if self._backend_instance is not None:
            updatedState = self._backend_instance.update_panel(
                key,
                panel_state,
                particleState=particleState,
                domain=self.domain,
                quantity=quantity,
                options=panel_state.options,
                **newOptions if newOptions is not None else {},
                **kwargs,
            )
        else:
            # Fallback: direct matplotlib updatePlot (keeps backward-compat if
            # PlotState was constructed without a backend instance).
            updatedState = updatePlot(
                panel_state,
                particles=particleState,
                domain=self.domain,
                quantity=quantity,
                options=panel_state.options,
                **newOptions if newOptions is not None else {},
                **kwargs,
            )
        self.plotStates[key] = updatedState


def visualize(
    particleState: Any,
    domain: DomainDescription,
    quantities: Union[torch.Tensor, Dict[str, torch.Tensor]],
    plotOptions: Union[PlottingOptions, Dict[str, PlottingOptions]],
    mosaic: str = "A",
    figsize: Tuple[float, float] = (7, 6),
    sharex: bool = True,
    sharey: bool = True,
    figTitle: Optional[str] = None,
    backend: str = "auto",
    backendOptions: Optional[dict] = None,
) -> "PlotState":
    """Create a multi-panel SPH visualization.

    Args:
        backend: ``"auto"`` (default) retains Matplotlib for small 2D plots and
            selects an installed accelerated backend for native 3D or at least
            50,000 visible scatter particles across panels. Explicit
            ``"matplotlib"``, ``"pyvista"``, and ``"vispy"`` choices are honored.
            Enum values are also accepted. Inspect ``PlotState.backend`` for
            the resolved name; updates retain that backend.
        backendOptions: Optional dict of keyword arguments forwarded verbatim to
            the chosen backend's ``create_figure`` call.
    """
    from .backends.factory import get_backend, select_backend

    from .state import snapshotParticleState, snapshotQuantity
    from .backends._mosaic import parse_mosaic
    from .backends._render_util import prepare_frame

    visParticleState = snapshotParticleState(particleState)
    quantities = snapshotQuantity(quantities)
    _, panel_keys, _ = parse_mosaic(mosaic)
    frames = {}
    spacing_cache = {}
    for key in panel_keys:
        frames[key] = prepare_frame(
            visParticleState,
            domain,
            quantities[key] if isinstance(quantities, dict) else quantities,
            plotOptions[key] if isinstance(plotOptions, dict) else plotOptions,
            _spacingCache=spacing_cache,
        )
    backend = select_backend(backend, frames.values(), sharex=sharex, sharey=sharey)
    be = get_backend(backend, backendOptions)
    for frame in frames.values():
        be.validate_frame(frame)
    fig = be.create_figure(
        mosaic=mosaic,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        figTitle=figTitle,
        backendOptions=backendOptions,
    )
    axis = be.get_axes()

    plotStates = {}
    for key in axis:
        panel_state = be.render_panel(
            key,
            particleState=visParticleState,
            domain=domain,
            quantity=quantities[key] if isinstance(quantities, dict) else quantities,
            options=plotOptions[key] if isinstance(plotOptions, dict) else plotOptions,
            _preparedFrame=frames[key],
        )
        plotStates[key] = panel_state

    if backend == "matplotlib" and fig.get_layout_engine() is None:
        fig.tight_layout()
    be.show()

    return PlotState(
        fig=fig,
        axes=axis,
        domain=domain,
        options=plotOptions,
        quantities=quantities,
        mosaic=mosaic,
        sharex=sharex,
        sharey=sharey,
        figTitle=figTitle,
        plotStates=plotStates,
        particleState=visParticleState,
        backend=str(getattr(backend, "value", backend)).lower(),
        backendOptions=backendOptions,
        _backend_instance=be,
    )

"""Matplotlib updates, using the same preparation as every other backend."""

import copy
from .state import VisualizationState
from .enumTypes import VisualizeOptions
from .domain import processDomain
from .math import getSharedBounds
from .scatter import updateScatterVisualize, removeResult
from .grid import mapFrameToGrid, updateGridVisualize
from .streamLines import updateStreamLinePlot, removeStreamLines
from .backends._render_util import prepare_frame, view_needs_refit, view_domain


def _renderFrame(state, frame, initial=False):
    options = frame.options
    old_domain = state.domain
    changed = initial or view_needs_refit(
        old_domain, frame, getattr(state, "displayAxes", tuple(range(old_domain.dim)))
    )
    processDomain(state.fig, state.axis, frame.domain, options, 0, resetLimits=False)
    if changed:
        fit = view_domain(frame)
        eps = (fit.max - fit.min) * options.domainEpsilon
        state.axis.set_xlim(float(fit.min[0] - eps[0]), float(fit.max[0] + eps[0]))
        state.axis.set_ylim(float(fit.min[1] - eps[1]), float(fit.max[1] + eps[1]))
    state.displayAxes = tuple(frame.displayAxes or range(frame.domain.dim))
    fig, ax = state.fig, state.axis
    if frame.displayAxes is not None:
        ax.set_xlabel("XYZ"[frame.displayAxes[0]])
        ax.set_ylabel("XYZ"[frame.displayAxes[1]])
    if options.gridVisualization is None:
        removeResult(state.gridResult)
        state.gridResult = None
        removeStreamLines(state.streamLines)
        state.streamLines = None
        fv = (
            options.fluidVisualization == VisualizeOptions.Visualize
            and len(frame.fluid.positions) > 0
        )
        bv = (
            options.boundaryVisualization == VisualizeOptions.Visualize
            and len(frame.boundary.positions) > 0
        )
        fq, bq, norm = frame.sharedBounds or getSharedBounds(
            frame.fluid.quantities if fv else None,
            frame.boundary.quantities if bv else None,
            options,
        )
        state.fluidScatterResult = updateScatterVisualize(
            state.fluidScatterResult,
            fig,
            ax,
            frame.fluid,
            frame.domain,
            options,
            options.fluidVisualization,
            fq,
            norm,
            fv,
        )
        state.boundaryScatterResult = updateScatterVisualize(
            state.boundaryScatterResult,
            fig,
            ax,
            frame.boundary,
            frame.domain,
            options,
            options.boundaryVisualization,
            bq,
            norm,
            not fv and bv,
        )
    else:
        removeResult(state.fluidScatterResult)
        removeResult(state.boundaryScatterResult)
        state.fluidScatterResult = state.boundaryScatterResult = None
        gs, gq, n, ext = mapFrameToGrid(frame, frame.particles.quantities)
        state.gridResult = updateGridVisualize(
            state.gridResult, fig, ax, gs, gq, n, ext, options
        )
        if options.gridVisualization.streamLines:
            vs, vq, vn, ve = mapFrameToGrid(frame, frame.streamlineQuantity)
            state.streamLines = updateStreamLinePlot(
                state.streamLines, fig, ax, vs, vq, vn, ve, options
            )
        else:
            removeStreamLines(state.streamLines)
            state.streamLines = None
    ax.set_title(
        options.plotTitle or "",
        **({"pad": options.plotTitleGap} if options.plotTitleGap is not None else {}),
    )
    state.domain = frame.domain
    state.options = options
    state.assembledQuantity = frame.assembled
    state.fluidParticles, state.boundaryParticles, state.ghostParticles = (
        frame.fluid,
        frame.boundary,
        frame.ghost,
    )
    return state


def updatePlot(
    plotState,
    particles,
    quantity=None,
    domain=None,
    options=None,
    _preparedFrame=None,
    resetView=False,
    **kwargs,
):
    requested = copy.deepcopy(options if options is not None else plotState.options)
    for key, value in kwargs.items():
        if not hasattr(requested, key):
            raise TypeError(f"Unknown plotting option: {key}")
        setattr(requested, key, value)
    original_domain = (
        domain
        if domain is not None
        else getattr(plotState, "_sourceDomain", plotState.domain)
    )
    frame = _preparedFrame or prepare_frame(
        particles,
        original_domain,
        plotState.assembledQuantity if quantity is None else quantity,
        requested,
    )
    frame.resetView |= resetView
    result = _renderFrame(plotState, frame)
    result._sourceDomain = original_domain
    return result


def updateVisualization(
    plotStates, particleState, domain, quantities, newOptions=None, **kwargs
):
    return {
        key: updatePlot(
            state,
            particleState,
            quantities[key] if isinstance(quantities, dict) else quantities,
            domain=domain,
            **((newOptions or {}).get(key, {})),
            **kwargs,
        )
        for key, state in plotStates.items()
    }

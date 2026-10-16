"""Streamline geometry and Matplotlib artist ownership."""

import numpy as np


def removeStreamLines(streamLines):
    if streamLines is None:
        return
    streamLines.lines.remove()
    for arrow in getattr(streamLines, "_warpsph_arrows", []):
        arrow.remove()


def streamLinePlot(
    fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options
):
    if len(gridResolution) != 2 or gridQuantity.ndim != 2 or gridQuantity.shape[1] != 2:
        raise ValueError("Streamline rendering requires a 2D vector grid.")
    nx, ny = gridResolution
    pts = gridState.positions.detach().cpu().numpy().reshape(nx, ny, 2)
    velocity = gridQuantity.detach().cpu().numpy().reshape(nx, ny, 2)
    velocity = np.ma.masked_invalid(velocity)
    if gridExtent is not None and gridExtent.get("coverage") is not None:
        covered = gridExtent["coverage"].detach().cpu().numpy().reshape(nx, ny)
        velocity = np.ma.masked_where(
            np.broadcast_to(~covered[..., None], velocity.shape), velocity
        )
    before = set(axis.patches)
    result = axis.streamplot(
        pts[:, 0, 0],
        pts[0, :, 1],
        velocity[:, :, 0].T,
        velocity[:, :, 1].T,
        color="k",
        density=1.5,
        linewidth=0.5,
    )
    result._warpsph_arrows = [p for p in axis.patches if p not in before]
    return result


def updateStreamLinePlot(
    streamLines, fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options
):
    removeStreamLines(streamLines)
    return streamLinePlot(
        fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options
    )


def streamlineSegments(gridState, gridQuantity, resolution, extent=None):
    """Deterministic 2D integration shared by GPU renderers via streamplot.

    Matplotlib's tested integrator runs on an unattached Agg figure, so this
    does not open windows or affect the user's pyplot figure registry.
    """
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.subplots()
    result = streamLinePlot(fig, ax, gridState, gridQuantity, resolution, extent, None)
    # New Matplotlib can emit variable-length polylines; split into edges.
    edges = []
    for path in result.lines.get_segments():
        path = np.asarray(path)
        if len(path) > 1:
            edges.extend(np.stack((path[:-1], path[1:]), axis=1))
    return np.asarray(edges, dtype=np.float32).reshape(-1, 2, 2)

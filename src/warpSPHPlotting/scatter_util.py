from typing import Optional
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


def computeMarkerSize(
    axis: plt.Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    dx: float,
):
    # We want to compute the marker size such for a plot with the given limits
    # each marker appears as a circle with diameter approximately equal to dx
    # xRange = xlim[1] - xlim[0]
    # yRange = ylim[1] - ylim[0]

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_aspect("equal", adjustable="box")

    # Because the marker size in matplotlib is specified in points^2, we need to convert the desired size in data units to points
    # We can use the axis transformation to do this conversion

    # Get the transformation from data units to display units (points)
    dataToDisplay = axis.transData
    # Compute the size of dx in display units
    dxDisplay = dataToDisplay.transform((dx, 0)) - dataToDisplay.transform((0, 0))
    dyDisplay = dataToDisplay.transform((0, dx)) - dataToDisplay.transform((0, 0))
    # The marker size in points^2 is then given by the area of the circle with diameter equal to the average of dxDisplay and dyDisplay
    markerSize = (
        (np.linalg.norm(dxDisplay) + np.linalg.norm(dyDisplay)) ** 2 * (np.pi / 4) * 0.8
    )  # We multiply by pi/4 to convert from diameter to radius and by 0.8 as a scaling factor to make the markers slightly smaller than the grid cells for better visibility
    return markerSize / 6

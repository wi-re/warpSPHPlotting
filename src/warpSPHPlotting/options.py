from dataclasses import dataclass
from enum import Enum, auto
from .colorMaps import ColorMap
from .enumTypes import VisualizeOptions, PlotScaling, Mapping, StreamLineLocation
from warpSPHCore import *
import torch
from typing import List, Optional, Union


@dataclass
class GridVisualization:
    sliceAxis: Optional[int] = (
        None  # For 3D input, sample a 2D plane normal to this axis.
    )
    slicePosition: Optional[float] = (
        None  # Defaults to the domain midpoint on sliceAxis.
    )
    resolution: int = (
        128  # Number of grid points along each dimension, the grid will be square in 2D and cubic in 3D
    )
    streamLines: bool = (
        False  # Whether to visualize streamlines based on the quantity gradient
    )
    gridSupport: Optional[SupportScheme] = (
        None  # The support scheme to use when mapping particle quantities to the grid for visualization. This can affect the appearance of the grid-based visualization, with different schemes providing different levels of smoothing or detail in the visualized quantity.'
    )

    streamLineOperation: Optional[WarpOperation] = (
        None  # If specified, the operation will be applied to the quantity before computing the streamlines, allowing visualization of derived quantities such as acceleration. The operation should be defined with the appropriate properties for the quantity being visualized (e.g., if visualizing acceleration, the operation should compute acceleration from positions and velocities).
    )
    streamLineGradientMode: GradientScheme = (
        GradientScheme.Difference
    )  # The scheme used to compute the gradient for the streamlines
    streamLineLaplaceMode: LaplacianScheme = (
        LaplacianScheme.Brookshaw
    )  # The scheme used to compute the Laplacian for the streamlines

    streamLinePositiveDivergence: bool = False
    streamLineOperationLocation: StreamLineLocation = (
        StreamLineLocation.BeforeOperation
    )  # Whether to apply the streamLineOperation before or after the main plotting operation is applied to the quantity. This allows visualization of streamlines based on either the original quantity or the quantity after the plotting operation is applied.

    maskEmpty: bool = True  # Make cells outside particle kernel coverage transparent.

    renderMode: str = (
        "volume"  # Native PyVista 3D grids: volume or surface; ignored for 2D/slices.
    )
    volumeOpacity: float = (
        0.15  # Opacity per tenth of the domain diagonal, independent of grid resolution.
    )


@dataclass
class PlottingOptions:
    plottingKernel: KernelFunctions = (
        KernelFunctions.Wendland4
    )  # The kernel function used for any grid-based computations in the visualization (e.g., for mapping particle quantities to a grid for visualization). This does not affect the underlying simulation computations, only the visualization.
    plottingOperation: Optional[
        Union[OperationProperties, List[OperationProperties]]
    ] = None  # A sequence of warp operations to apply to the quantity being visualized, allowing visualization of derived quantities

    fluidVisualization: VisualizeOptions = (
        VisualizeOptions.Visualize
    )  # Whether to visualize fluid particles, and if so, whether to visualize them as passive (gray) or with the specified color map based on their quantity values
    boundaryVisualization: VisualizeOptions = (
        VisualizeOptions.Hide
    )  # Whether to visualize boundary particles, and if so, whether to visualize them as passive (gray) or with the specified color map based on their quantity values

    showColorBar: bool = (
        True  # Whether to show a color bar for the quantity values when visualizing with a color map
    )
    colorMap: ColorMap = (
        ColorMap.viridis
    )  # The color map to use when visualizing particle quantities, if visualizing with a color map
    flipColorMap: bool = False  # adds _r to the colormap name

    quantityScaling: PlotScaling = (
        PlotScaling.Linear
    )  # The scaling to use for the quantity values when visualizing with a color map (e.g., linear, logarithmic, symmetric logarithmic, etc.)
    quantityLogThreshold: float = (
        1e-3  # Only used if quantityScaling is set to SymmetricLog
    )
    vMin: Optional[float] = (
        None  # The minimum value for scaling the quantity values when visualizing with a color map. If None, the minimum value will be determined from the data.
    )
    vMax: Optional[float] = (
        None  # The maximum value for scaling the quantity values when visualizing with a color map. If None, the maximum value will be determined from the data.
    )
    midPoint: Optional[Union[float, str]] = (
        None  # Used to center the scaling around a value with equal positive and negative bands
    )

    domainEpsilon: float = (
        0.05  # A small buffer added to the domain bounds when visualizing to ensure particles near the boundaries are still visible. This is a fraction of the domain size (e.g., 0.05 means adding a 5% buffer to each side of the domain).
    )
    markerSize: Optional[float] = (
        None  # Matplotlib scatter s (points squared); None fits particle spacing and camera zoom
    )

    markerSizeScale: float = (
        0.65  # Automatic diameter as a fraction of particle spacing; ignored for explicit markerSize
    )

    mapping: Mapping = (
        Mapping.none
    )  # If not none, this specifies how to map the quantity values to a scalar value for visualization (e.g., if the quantity has multiple components or is a vector/tensor, how to reduce it to a single scalar value for coloring). The specific mapping options depend on the type of quantity being visualized (e.g., for a vector quantity, options might include magnitude, individual components, etc.).
    gridVisualization: Optional[GridVisualization] = (
        None  # If specified, this enables grid-based visualization of the quantity, where the particle quantities are mapped to a regular grid for visualization. This can allow for visualizing quantities that are not easily visualized on the particles themselves (e.g., velocity fields with streamlines) or for visualizing in a way that is less noisy than the particle-based visualization. The specific options for the grid visualization are defined in the GridVisualization dataclass.
    )

    plotDomain: bool = (
        True  # Whether to plot the domain boundaries (e.g., as a box in 2D or a cube in 3D) for reference. This can help provide context for the particle positions and quantities being visualized, especially if the domain has non-periodic boundaries or if the particles are not filling the entire domain.
    )
    plotTitle: Optional[str] = (
        None  # An optional title to display on the plot, which can be useful for identifying the quantity being visualized or for distinguishing between different plots when visualizing multiple quantities or time steps.
    )
    plotTitleGap: Optional[float] = (
        None  # Optional title spacing. For matplotlib this is title padding in points; for vispy this is a fraction of the domain height. If None, backend defaults are used.
    )
    dynamicBounds: bool = (
        False  # Refit the viewport to domain/visible particles on every update.
    )
    markerSpacing: str = (
        "volume"  # Automatic sizing: physical volume, or 'neighbors' to cap by close pairs.
    )

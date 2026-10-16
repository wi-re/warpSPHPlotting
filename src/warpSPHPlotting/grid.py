from warpSPHCore import *
import torch
from .math import getBounds
from .enumTypes import VisualizeOptions


def generateGrid(
    domain: DomainDescription,
    resolution: int,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    alignment: str = "center",
    shortEdge: bool = False,
):
    # Generate a grid of points within the specified domain and resolution
    # Alignment can be 'center', 'edge', 'left', or 'right'

    if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 2:
        raise ValueError("Grid resolution must be an integer >= 2.")
    edgeLengths = (domain.max - domain.min).to(device=device, dtype=dtype)
    if not torch.isfinite(edgeLengths).all() or not torch.all(edgeLengths > 0):
        raise ValueError("Grid domain must have finite positive extents.")
    edge = edgeLengths.min() if shortEdge else edgeLengths.max()
    ns = [max(2, int(float(length / edge) * resolution)) for length in edgeLengths]
    denominator = [n - 1 if alignment == "edge" else n for n in ns]
    dx = edgeLengths / torch.tensor(denominator, device=device, dtype=dtype)
    if alignment == "center":
        start = domain.min + dx / 2
        end = domain.max - dx / 2
    elif alignment == "edge":
        start = domain.min
        end = domain.max
    elif alignment == "left":
        start = domain.min
        end = domain.max - dx
    elif alignment == "right":
        start = domain.min + dx
        end = domain.max
    else:
        raise ValueError(f"Unsupported alignment: {alignment}")

    grids = [
        torch.linspace(start[i], end[i], ns[i], device=device, dtype=dtype)
        for i in range(domain.dim)
    ]
    mesh = torch.meshgrid(*grids, indexing="ij")
    gridPoints = torch.stack(mesh, dim=-1).reshape(-1, domain.dim)

    gridArea = torch.prod(dx)

    gridExtent = {
        "min": domain.min,
        "max": domain.max,
    }

    return gridPoints, dx, ns, gridArea, gridExtent


def mapToGrid(
    particleState: ParticleState,
    quantity: torch.Tensor,
    domain: DomainDescription,
    nx: int,
    targetNeighbors: int,
    kernel: KernelFunctions,
    alignment: str = "center",
    includeFluid: bool = True,
    includeBoundary: bool = False,
    gridMode: SupportScheme = SupportScheme.Scatter,
    sliceAxis=None,
    slicePosition=None,
    _cache=None,
    computeCoverage=False,
):
    # Map particle quantities to a grid using SPH-like interpolation
    device = particleState.positions.device
    dtype = particleState.positions.dtype

    # Cache only inside one immutable prepared frame. No search structure
    # survives a particle/domain/support update.
    cache = {} if _cache is None else _cache
    if "_grid" not in cache:
        if sliceAxis is None:
            grid, dx, nxs, gridArea, gridExtent = generateGrid(
                domain, nx, device=device, dtype=dtype, alignment=alignment
            )
        else:
            if domain.dim != 3 or sliceAxis not in (0, 1, 2):
                raise ValueError("sliceAxis must be 0, 1, or 2 for a 3D domain.")
            axes = [i for i in range(3) if i != sliceAxis]
            position = (
                float((domain.min[sliceAxis] + domain.max[sliceAxis]) / 2)
                if slicePosition is None
                else float(slicePosition)
            )
            if not torch.isfinite(torch.tensor(position)) or not float(
                domain.min[sliceAxis]
            ) <= position <= float(domain.max[sliceAxis]):
                raise ValueError("slicePosition must lie inside the domain.")
            plane = DomainDescription(
                domain.min[axes], domain.max[axes], domain.periodic[axes], 2
            )
            points, dx, nxs, area, gridExtent = generateGrid(
                plane, nx, device=device, dtype=dtype, alignment=alignment
            )
            grid = torch.full((len(points), 3), position, device=device, dtype=dtype)
            grid[:, axes] = points
            gridArea = area * torch.sqrt(torch.prod(dx))
            gridExtent.update(axes=axes, sliceAxis=sliceAxis, slicePosition=position)

        h = volumeToSupport(gridArea, targetNeighbors, particleState.positions.shape[1])

        gridState = ParticleState(
            positions=grid,
            supports=torch.full((grid.shape[0],), h, device=device, dtype=dtype),
            masses=torch.full((grid.shape[0],), gridArea, device=device, dtype=dtype),
            densities=torch.ones((grid.shape[0],), device=device, dtype=dtype),
            kinds=torch.full(
                (grid.shape[0],),
                ParticleType.Fluid.value,
                device=device,
                dtype=torch.int32,
            ),
        )
        cache["_grid"] = (gridState, nxs, gridExtent)
    gridState, nxs, gridExtent = cache["_grid"]
    grid = gridState.positions
    gridQuantity = torch.zeros(
        (grid.shape[0],) + quantity.shape[1:], device=device, dtype=dtype
    )

    if gridMode is None:
        # A plane's sample count cannot be compared to a 3D particle count.
        # Gather would derive a tiny 3D kernel from the 2D pixel spacing,
        # producing stripes/holes as the plane passes between particle layers.
        if sliceAxis is None and grid.shape[0] < particleState.positions.shape[0]:
            gridMode = SupportScheme.Gather
        else:
            gridMode = SupportScheme.Scatter

    if (includeFluid or includeBoundary) and "_hash" not in cache:
        cache["_hash"] = buildCompactHashMap(
            queryPositions=gridState.positions,
            referencePositions=particleState.positions,
            querySupports=gridState.supports,
            referenceSupports=particleState.supports,
            periodicity=domain.periodic,
            domainDescription=domain,
            mode=gridMode,
        )

    if includeFluid:
        gridQuantity += warpOperation(
            queryParticles=gridState,
            referenceParticles=particleState,
            queryValues=gridQuantity,
            referenceValues=quantity,
            operationProperties=OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Interpolate,
                operationMode=OperationDirection.FluidToFluid,
                supportMode=gridMode,
            ),
            domain=domain,
            adjacency=cache["_hash"],
        )
    if includeBoundary:
        gridQuantity += warpOperation(
            queryParticles=gridState,
            referenceParticles=particleState,
            queryValues=gridQuantity,
            referenceValues=quantity,
            operationProperties=OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Interpolate,
                operationMode=OperationDirection.BoundaryToFluid,
                supportMode=gridMode,
            ),
            domain=domain,
            adjacency=cache["_hash"],
        )
    if computeCoverage:
        if "_coverage" not in cache:
            weights = torch.zeros(len(grid), device=device, dtype=dtype)
            for include, direction in (
                (includeFluid, OperationDirection.FluidToFluid),
                (includeBoundary, OperationDirection.BoundaryToFluid),
            ):
                if include:
                    weights += warpOperation(
                        queryParticles=gridState,
                        referenceParticles=particleState,
                        queryValues=torch.zeros_like(weights),
                        referenceValues=torch.ones_like(particleState.masses),
                        operationProperties=OperationProperties(
                            kernel=kernel,
                            operation=WarpOperation.Interpolate,
                            operationMode=direction,
                            supportMode=gridMode,
                        ),
                        domain=domain,
                        adjacency=cache["_hash"],
                    )
            cache["_coverage"] = torch.isfinite(weights) & (weights > 0)
        # Keep the raw interpolation (including unsupported zeros) unchanged.
        gridExtent = dict(gridExtent, coverage=cache["_coverage"])
    return gridState, gridQuantity, nxs, gridExtent


def gridDisplayBounds(values, extent, options):
    """Resolve colors from covered samples without changing physical values."""
    import numpy as np

    coverage = extent.get("coverage")
    display = values
    if coverage is not None:
        display = values.masked_fill(~coverage, float("nan"))
        if not bool(coverage.any()):
            # An empty view has no measured color range to advertise.
            return np.ma.masked_all(tuple(values.shape)), None
    return getBounds(display, options)


def gridVisualize(
    fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options
):
    return updateGridVisualize(
        None, fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options
    )


def updateGridVisualize(
    priorResult, fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options
):
    import numpy as np

    if len(gridResolution) != 2:
        raise ValueError("Matplotlib grid rendering requires a 2D domain.")
    qs, norm = gridDisplayBounds(gridQuantity, gridExtent, options)
    # Explicit cell edges also support thin domains with only a few cells.
    nx, ny = gridResolution
    mn = gridExtent["min"].detach().cpu().numpy()
    mx = gridExtent["max"].detach().cpu().numpy()
    geometry = (tuple(gridResolution), tuple(mn), tuple(mx))
    sc, cb, _ = priorResult or (None, None, None)
    if sc is not None and getattr(sc, "_warpsph_geometry", None) != geometry:
        if cb is not None:
            cb.remove()
            cb = None
        sc.remove()
        sc = None
    if sc is None:
        sc = axis.pcolormesh(
            np.linspace(mn[0], mx[0], nx + 1),
            np.linspace(mn[1], mx[1], ny + 1),
            qs.reshape(nx, ny).T,
            shading="flat",
            cmap=options.colorMap.value + ("_r" if options.flipColorMap else ""),
            norm=norm,
        )
        sc._warpsph_geometry = geometry
    else:
        sc.set_array(qs.reshape(nx, ny).T)
        sc.set_cmap(options.colorMap.value + ("_r" if options.flipColorMap else ""))
        sc.set_norm(norm)
    if cb is not None and (not options.showColorBar or norm is None):
        cb.remove()
        cb = None
    if options.showColorBar and norm is not None and cb is None:
        cb = fig.colorbar(sc, ax=axis, pad=0.025)
    if cb is not None:
        from .scatter import updateColorBar

        updateColorBar(cb, sc)
    return sc, cb, norm


def mapFrameToGrid(frame, quantity):
    """Interpolate in physical space, then project an optional 3D slice."""
    import copy

    opts = frame.options
    grid = opts.gridVisualization
    state, values, n, extent = mapToGrid(
        frame.sourceParticles or frame.particles,
        quantity,
        frame.sourceDomain or frame.domain,
        grid.resolution,
        50,
        opts.plottingKernel,
        includeFluid=opts.fluidVisualization != VisualizeOptions.Hide,
        includeBoundary=opts.boundaryVisualization != VisualizeOptions.Hide,
        gridMode=grid.gridSupport,
        sliceAxis=grid.sliceAxis,
        slicePosition=grid.slicePosition,
        _cache=frame.gridCache,
        computeCoverage=grid.maskEmpty,
    )
    if frame.displayAxes is not None:
        state = copy.copy(state)
        state.positions = state.positions[:, frame.displayAxes]
        extent = dict(extent, min=frame.domain.min, max=frame.domain.max)
    return state, values, n, extent

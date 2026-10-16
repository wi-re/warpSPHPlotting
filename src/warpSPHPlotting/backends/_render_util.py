"""Shared preparation for creation and updates on every rendering backend."""

from dataclasses import dataclass, field
import copy
import torch
from warpSPHCore import (
    DomainDescription,
    ParticleType,
    warpOperation,
    OperationProperties,
    OperationDirection,
)
from ..enumTypes import Mapping, StreamLineLocation, VisualizeOptions
from .._sizing import validate_marker_options
from ..math import buildRotationMatrix, mapQuantity
from ..state import PlottingParticleState, assembleQuantity, filterState, rotateState


@dataclass
class PreparedFrame:
    domain: object
    options: object
    assembled: torch.Tensor
    particles: PlottingParticleState
    fluid: PlottingParticleState
    boundary: PlottingParticleState
    ghost: PlottingParticleState
    streamlineQuantity: object = None
    sourceDomain: object = None
    sourceParticles: object = None
    displayAxes: object = None
    sharedBounds: object = None
    gridCache: dict = field(default_factory=dict)
    resetView: bool = False


def view_needs_refit(previous_domain, frame, previous_axes=None):
    """Changing particle buffers alone must not discard a user's camera."""
    axes = tuple(frame.displayAxes or range(frame.domain.dim))
    return (
        frame.resetView
        or frame.options.dynamicBounds
        or previous_domain.dim != frame.domain.dim
        or not torch.equal(previous_domain.min, frame.domain.min)
        or not torch.equal(previous_domain.max, frame.domain.max)
        or (previous_axes is not None and tuple(previous_axes) != axes)
    )


def view_domain(frame):
    """Optional data-following bounds, separate from the physical/SPH domain."""
    if not frame.options.dynamicBounds:
        return frame.domain
    mn, mx = frame.domain.min.clone(), frame.domain.max.clone()
    for particles, visibility in (
        (frame.fluid, frame.options.fluidVisualization),
        (frame.boundary, frame.options.boundaryVisualization),
    ):
        if visibility == VisualizeOptions.Hide:
            continue
        points = particles.positions
        points = points[torch.isfinite(points).all(dim=1)]
        if len(points):
            mn = torch.minimum(mn, points.amin(dim=0))
            mx = torch.maximum(mx, points.amax(dim=0))
    return DomainDescription(mn, mx, frame.domain.periodic, frame.domain.dim)


def prepare_frame(particleState, domain, quantity, options, _spacingCache=None):
    opts = copy.deepcopy(options)
    validate_marker_options(opts)
    domain_ = DomainDescription(
        min=domain.min.detach().clone(),
        max=domain.max.detach().clone(),
        periodic=domain.periodic.detach().clone(),
        dim=domain.dim,
    )
    if domain.dim not in (2, 3):
        raise ValueError("Particle rendering supports 2D or 3D domains.")
    if (
        not torch.isfinite(domain_.min).all()
        or not torch.isfinite(domain_.max).all()
        or not torch.all(domain_.max > domain_.min)
    ):
        raise ValueError("Domain bounds must be finite and have positive extents.")
    n = len(particleState.positions)
    if (
        particleState.positions.ndim != 2
        or particleState.positions.shape[1] != domain.dim
    ):
        raise ValueError("Particle coordinates must match the domain dimension.")
    for name in ("supports", "masses", "densities", "kinds"):
        value = getattr(particleState, name, None)
        if not isinstance(value, torch.Tensor) or value.shape != (n,):
            raise ValueError(f"Particle {name} must be a tensor with {n} entries.")
    if (
        opts.gridVisualization
        and opts.gridVisualization.streamLines
        and domain.dim != 2
        and opts.gridVisualization.sliceAxis is None
    ):
        raise ValueError("Streamline rendering currently requires a 2D vector field.")
    assembled = assembleQuantity(particleState, quantity).detach()
    p = PlottingParticleState(
        particleState.positions.detach(),
        particleState.supports.detach(),
        particleState.masses.detach(),
        particleState.densities.detach(),
        particleState.kinds.detach(),
        assembled,
    )
    if getattr(domain, "angles", None) is not None:
        angles = torch.as_tensor(
            domain.angles, dtype=p.positions.dtype, device=p.positions.device
        )
        rotation = buildRotationMatrix(
            angles, domain.dim, device=p.positions.device, dtype=p.positions.dtype
        )
        p = rotateState(p, rotation.inverse())
    p.positions = torch.stack(
        [
            (
                torch.remainder(
                    p.positions[:, i] - domain_.min[i], domain_.max[i] - domain_.min[i]
                )
                + domain_.min[i]
                if domain_.periodic[i]
                else p.positions[:, i]
            )
            for i in range(domain.dim)
        ],
        dim=-1,
    )
    grid = opts.gridVisualization
    if grid is not None:
        if grid.renderMode not in ("volume", "surface"):
            raise ValueError("grid renderMode must be volume or surface.")
        opacity = float(grid.volumeOpacity)
        if not 0 <= opacity <= 1:
            raise ValueError("volumeOpacity must be finite and between 0 and 1.")
    stream = None
    if (
        grid
        and grid.streamLines
        and grid.streamLineOperationLocation == StreamLineLocation.BeforeOperation
    ):
        stream = p.quantities
    if opts.plottingOperation is not None:
        ops = (
            opts.plottingOperation
            if isinstance(opts.plottingOperation, list)
            else [opts.plottingOperation]
        )
        for op in ops:
            p.quantities = warpOperation(
                queryParticles=p,
                queryValues=p.quantities,
                operationProperties=op,
                adjacency=None,
                domain=domain_,
            )
    if (
        grid
        and grid.streamLines
        and grid.streamLineOperationLocation == StreamLineLocation.BeforeMapping
    ):
        stream = p.quantities
    p.quantities = mapQuantity(p.quantities, opts.mapping)
    if grid and grid.streamLines:
        if grid.streamLineOperationLocation == StreamLineLocation.AfterMapping:
            stream = p.quantities
        if grid.streamLineOperation is not None:
            stream = warpOperation(
                queryParticles=p,
                queryValues=stream,
                operationProperties=OperationProperties(
                    kernel=opts.plottingKernel,
                    operation=grid.streamLineOperation,
                    gradientMode=grid.streamLineGradientMode,
                    laplacianMode=grid.streamLineLaplaceMode,
                    operationMode=OperationDirection.NoGhost,
                ),
                adjacency=None,
                domain=domain_,
            )
        if stream is None or stream.ndim != 2 or stream.shape[1] != domain.dim:
            raise ValueError(
                "Streamlines require a vector field; choose BeforeMapping or a gradient operation."
            )
    if p.quantities.ndim != 1 or p.quantities.shape[0] != p.positions.shape[0]:
        raise ValueError(
            f"Expected one scalar per particle, got {tuple(p.quantities.shape)}; specify options.mapping."
        )
    source_domain, source_particles, axes = None, None, None
    if grid is not None and grid.sliceAxis is not None:
        if domain.dim != 3 or grid.sliceAxis not in (0, 1, 2):
            raise ValueError("sliceAxis requires a 3D domain and axis 0, 1, or 2.")
        position = (
            float((domain_.min[grid.sliceAxis] + domain_.max[grid.sliceAxis]) / 2)
            if grid.slicePosition is None
            else float(grid.slicePosition)
        )
        if not torch.isfinite(torch.tensor(position)) or not float(
            domain_.min[grid.sliceAxis]
        ) <= position <= float(domain_.max[grid.sliceAxis]):
            raise ValueError("slicePosition must lie inside the domain.")
        axes = [i for i in range(3) if i != grid.sliceAxis]
        source_domain, source_particles = domain_, p
        domain_ = DomainDescription(
            domain_.min[axes], domain_.max[axes], domain_.periodic[axes], 2
        )
        p = copy.copy(p)
        p.positions = p.positions[:, axes]
        if stream is not None:
            stream = stream[:, axes]
    frame = PreparedFrame(
        domain_,
        opts,
        assembled,
        p,
        filterState(p, p.quantities, kind=ParticleType.Fluid),
        filterState(p, p.quantities, kind=ParticleType.Boundary),
        filterState(p, p.quantities, kind=ParticleType.Ghost),
        stream,
        source_domain,
        source_particles,
        axes,
    )
    spacing_cache = {} if _spacingCache is None else _spacingCache
    for kind in ("fluid", "boundary", "ghost"):
        getattr(frame, kind)._spacingCache = spacing_cache.setdefault(kind, {})
    return frame


def prepare_particle_states(particleState, domain, quantity, options):
    frame = prepare_frame(particleState, domain, quantity, options)
    return (
        frame.domain,
        frame.options,
        frame.assembled,
        frame.particles.quantities,
        frame.fluid,
        frame.boundary,
        frame.ghost,
    )

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
from warpSPHCore import *
from .options import PlottingOptions
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
import matplotlib.colors as colors
from matplotlib.streamplot import StreamplotSet


@dataclass
class PlottingParticleState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor
    kinds: torch.Tensor
    quantities: torch.Tensor
    _spacingCache: Optional[dict] = field(default=None, repr=False, compare=False)


def rotateState(
    state: PlottingParticleState, rotMat: Optional[torch.Tensor]
) -> PlottingParticleState:
    if rotMat is None:
        return state
    else:
        rotatedPositions = torch.einsum("ij, ni->nj", rotMat, state.positions)
        rotatedQuantities = None
        if state.quantities.ndim == 2 and state.quantities.shape[1] == rotMat.shape[0]:
            rotatedQuantities = torch.einsum("ij, ni->nj", rotMat, state.quantities)
        else:
            rotatedQuantities = state.quantities
        return PlottingParticleState(
            positions=rotatedPositions,
            supports=state.supports,
            masses=state.masses,
            densities=state.densities,
            kinds=state.kinds,
            quantities=rotatedQuantities,
        )


def filterState(
    particleState: ParticleState,
    quantity: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    kind: ParticleType = ParticleType.Fluid,
    batch=None,
    rotMat: Optional[torch.Tensor] = None,
):
    mask = particleState.kinds == kind.value
    if batch is not None:
        mask = torch.logical_and(mask, particleState.batches == batch)
    state = PlottingParticleState(
        positions=particleState.positions[mask],
        supports=particleState.supports[mask],
        masses=particleState.masses[mask],
        densities=particleState.densities[mask],
        kinds=particleState.kinds[mask],
        quantities=assembleQuantity(particleState, quantity)[mask],
    )
    if rotMat is not None and torch.sum(mask) > 0:
        state = rotateState(state, rotMat)
    return state


def assembleQuantity(
    particleState: ParticleState,
    quantity: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
):
    n = particleState.positions.shape[0]
    if isinstance(quantity, torch.Tensor):
        if quantity.ndim == 0 or quantity.shape[0] != n:
            raise ValueError(f"Quantity must contain {n} particle samples.")
        return quantity
    if not isinstance(quantity, (tuple, list)):
        raise TypeError("quantity must be a tensor or a per-kind tuple/list.")
    first = next((q for q in quantity if q is not None), None)
    if first is None:
        raise ValueError("At least one per-kind quantity is required.")
    assembled = torch.zeros(
        (n,) + first.shape[1:], device=first.device, dtype=first.dtype
    )
    for kind in ParticleType:
        mask = particleState.kinds == kind.value
        count = int(mask.sum())
        if count == 0:
            continue
        if kind.value >= len(quantity) or quantity[kind.value] is None:
            continue
        q = quantity[kind.value]
        if q.shape[1:] != first.shape[1:]:
            raise ValueError("Per-kind quantities must have matching component shapes.")
        if q.shape[0] == n:
            assembled[mask] = q[mask]
        elif q.shape[0] == count:
            assembled[mask] = q
        else:
            raise ValueError(
                f"Quantity for {kind.name} must have {count} or {n} samples."
            )
    return assembled


@dataclass
class VisualizationState:
    fig: plt.Figure
    axis: plt.Axes
    domain: DomainDescription
    options: PlottingOptions

    assembledQuantity: torch.Tensor

    fluidParticles: Optional[PlottingParticleState] = None
    boundaryParticles: Optional[PlottingParticleState] = None
    ghostParticles: Optional[PlottingParticleState] = None

    fluidScatterResult: Optional[
        Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]
    ] = None
    boundaryScatterResult: Optional[
        Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]
    ] = None
    ghostScatterResult: Optional[
        Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]
    ] = None
    gridResult: Optional[
        Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]
    ] = None

    streamLines: Optional[StreamplotSet] = None


def snapshotParticleState(state):
    """Detached plotting buffers, including differentiable non-leaf tensors."""
    from copy import copy

    result = copy(state)
    for name in ("positions", "supports", "masses", "densities", "kinds"):
        value = getattr(state, name, None)
        if isinstance(value, torch.Tensor):
            setattr(result, name, value.detach().clone())
    return result


def snapshotQuantity(quantity):
    """Own display values without retaining a simulation's autograd graph."""
    if isinstance(quantity, torch.Tensor):
        return quantity.detach().clone()
    if isinstance(quantity, dict):
        return {key: snapshotQuantity(value) for key, value in quantity.items()}
    if isinstance(quantity, (tuple, list)):
        return tuple(
            snapshotQuantity(value) if value is not None else None for value in quantity
        )
    raise TypeError("quantity must be a tensor, dictionary, or per-kind tuple/list")

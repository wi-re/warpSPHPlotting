from typing import List
import matplotlib.colors as colors
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
import torch
import numpy as np
from .options import PlottingOptions
from .enumTypes import Mapping, PlotScaling


def buildRotationMatrix(
    angles: List[float],
    dim: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    if dim == 1:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    elif dim == 2:
        return torch.tensor(
            [
                [torch.cos(angles), -torch.sin(angles)],
                [torch.sin(angles), torch.cos(angles)],
            ],
            device=device,
            dtype=dtype,
        )
    elif dim == 3:
        angle_phi = angles[0]
        angle_theta = angles[1]
        return torch.tensor(
            [
                [
                    torch.cos(angle_phi) * torch.sin(angle_theta),
                    -torch.sin(angle_phi),
                    torch.cos(angle_phi) * torch.cos(angle_theta),
                ],
                [
                    torch.sin(angle_phi) * torch.sin(angle_theta),
                    torch.cos(angle_phi),
                    torch.sin(angle_phi) * torch.cos(angle_theta),
                ],
                [torch.cos(angle_theta), 0, -torch.sin(angle_theta)],
            ],
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def getBounds(values: torch.Tensor, options: PlottingOptions):
    """Resolve display bounds without modifying options or physical values.

    Nonfinite samples, and nonpositive samples on log scales, are masked.
    Empty/all-invalid fields require explicit bounds so invalid data never
    silently determines a color range.
    """
    raw = values.detach().cpu().numpy()
    if np.iscomplexobj(raw):
        raise ValueError(
            "Complex quantities require a real-valued mapping such as magnitude."
        )
    q = raw.astype(float, copy=False)
    valid = np.isfinite(q)
    scaling = options.quantityScaling
    if scaling == PlotScaling.Logarithmic:
        valid &= q > 0
    finite = q[valid]
    if finite.size == 0 and (options.vMin is None or options.vMax is None):
        raise ValueError("No valid quantity samples; supply finite vMin and vMax.")
    lo = float(options.vMin) if options.vMin is not None else float(finite.min())
    hi = float(options.vMax) if options.vMax is not None else float(finite.max())
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("vMin and vMax must be finite with vMin <= vMax.")
    if scaling in (PlotScaling.Symmetric, PlotScaling.SymmetricLog):
        center = options.midPoint
        if isinstance(center, str):
            if center != "median":
                raise ValueError("midPoint must be a number, None, or 'median'.")
            if not finite.size:
                raise ValueError("Cannot compute a median without valid samples.")
            center = float(np.median(finite))
        center = 0.0 if center is None else float(center)
        if not np.isfinite(center):
            raise ValueError("midPoint must be finite.")
        if options.vMin is None or options.vMax is None:
            radius = float(np.max(np.abs(finite - center))) if finite.size else 0.0
            radius = max(radius, abs(center) * 1e-6, 1e-12)
            lo = center - radius if options.vMin is None else lo
            hi = center + radius if options.vMax is None else hi
        if scaling == PlotScaling.Symmetric:
            radius = max(abs(lo - center), abs(hi - center), 1e-12)
            lo, hi = center - radius, center + radius
            norm = colors.CenteredNorm(vcenter=center, halfrange=radius)
        else:
            if options.quantityLogThreshold <= 0:
                raise ValueError("quantityLogThreshold must be positive.")
            norm = colors.SymLogNorm(
                linthresh=options.quantityLogThreshold, linscale=0.03, vmin=lo, vmax=hi
            )
    elif scaling == PlotScaling.Logarithmic:
        if lo <= 0 or hi <= 0:
            raise ValueError("Logarithmic bounds must be positive.")
        if lo == hi:
            lo, hi = lo / 1.01, hi * 1.01
        norm = colors.LogNorm(vmin=lo, vmax=hi)
    else:
        if lo == hi:
            pad = max(abs(lo) * 1e-6, 1e-12)
            lo, hi = lo - pad, hi + pad
        norm = colors.Normalize(vmin=lo, vmax=hi)
    return np.ma.array(np.clip(q, lo, hi), mask=~valid), norm


def mapQuantity(
    quantity: torch.Tensor,
    mapping: Mapping,
):
    if not isinstance(quantity, torch.Tensor) or quantity.ndim == 0:
        raise ValueError("quantity must be a tensor with one leading particle axis.")
    if mapping == Mapping.none:
        return (
            quantity.squeeze(-1)
            if quantity.ndim == 2 and quantity.shape[1] == 1
            else quantity
        )
    if mapping in (Mapping.x, Mapping.y, Mapping.z):
        component = {Mapping.x: 0, Mapping.y: 1, Mapping.z: 2}[mapping]
        if quantity.ndim == 1 and component == 0:
            return quantity
        if quantity.ndim != 2 or quantity.shape[1] <= component:
            raise ValueError(
                f"Mapping {mapping.value} is incompatible with shape {tuple(quantity.shape)}."
            )
        return quantity[:, component]
    if mapping in (Mapping.L2Norm, Mapping.magnitude, Mapping.L2, Mapping.L1):
        if quantity.ndim < 2:
            return quantity.abs()
        if not quantity.is_floating_point() and not quantity.is_complex():
            quantity = quantity.to(torch.get_default_dtype())
        return torch.linalg.vector_norm(
            quantity.flatten(1), dim=1, ord=1 if mapping == Mapping.L1 else 2
        )
    raise ValueError(f"Unsupported mapping: {mapping}")


def getSharedBounds(fluidQuantities, boundaryQuantities, options: PlottingOptions):
    """One norm across fluid **and** boundary, so the two are comparable.

    Returns ``(fluidQs, boundaryQs, norm)``; pass ``None`` (or an empty tensor)
    for a set that is hidden, passive or absent and the norm is taken from the
    other alone. ``(None, None, None)`` when neither is drawn.

    Normalising the two sets independently and then drawing them through the
    same colormap -- which the vispy and pyvista backends did -- makes wall
    colours incomparable to fluid colours. A dry mDBC wall particle sitting at
    exactly ``rho0`` is the *minimum of the boundary set*, so it saturates one
    end of the map, while the fluid's own minimum is a different number
    entirely; the two meet at the waterline as a hard edge that is not in the
    data. The single colorbar can only describe one of the two norms, leaving
    the other unlabelled. ``visualize.py`` / ``update.py`` (matplotlib) already
    shared a norm; this is the same thing for the other backends.
    """
    hasFluid = fluidQuantities is not None and fluidQuantities.shape[0] > 0
    hasBoundary = boundaryQuantities is not None and boundaryQuantities.shape[0] > 0

    if hasFluid and hasBoundary:
        combinedQs, norm = getBounds(
            torch.cat((fluidQuantities, boundaryQuantities), dim=0), options
        )
        nFluid = fluidQuantities.shape[0]
        return combinedQs[:nFluid], combinedQs[nFluid:], norm
    if hasFluid:
        qs, norm = getBounds(fluidQuantities, options)
        return qs, None, norm
    if hasBoundary:
        qs, norm = getBounds(boundaryQuantities, options)
        return None, qs, norm
    return None, None, None

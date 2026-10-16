"""Matplotlib domain geometry owned by the plotting axes."""

import torch
from matplotlib import patches
from warpSPHCore import DomainDescription
from .math import buildRotationMatrix
from ._sizing import particle_diameter
import numpy as np


def processDomain(fig, axis, domain, options, dx, resetLimits=True):
    if domain.dim != 2:
        raise ValueError(
            "Matplotlib particle plots require a 2D domain; use PyVista for 3D."
        )
    domain_ = DomainDescription(
        domain.min.detach().clone(),
        domain.max.detach().clone(),
        domain.periodic.detach().clone(),
        domain.dim,
    )
    rot = None
    if getattr(domain, "angles", None) is not None:
        rot = buildRotationMatrix(
            torch.as_tensor(domain.angles),
            domain.dim,
            device=domain.min.device,
            dtype=domain.min.dtype,
        )
    mn, mx = domain_.min.cpu().numpy(), domain_.max.cpu().numpy()
    eps = (mx - mn) * options.domainEpsilon
    if resetLimits:
        axis.set_xlim(mn[0] - eps[0], mx[0] + eps[0])
        axis.set_ylim(mn[1] - eps[1], mx[1] + eps[1])
    square = next((p for p in axis.patches if p.get_gid() == "warpsph-domain"), None)
    if options.plotDomain:
        if square is None:
            square = patches.Rectangle(
                mn,
                *(mx - mn),
                linewidth=1,
                edgecolor="b",
                facecolor="none",
                ls="--",
                gid="warpsph-domain",
            )
            axis.add_patch(square)
        square.set_bounds(mn[0], mn[1], mx[0] - mn[0], mx[1] - mn[1])
        square.set_visible(True)
    elif square is not None:
        square.set_visible(False)
    axis.set_aspect("equal")
    return domain_, rot, rot.inverse() if rot is not None else None


def markerSize(axis, particles, options):
    if options.markerSize is not None:
        return options.markerSize
    diameter = particle_diameter(particles, options)
    transform = axis.transData
    origin = transform.transform((0, 0))
    pixels = min(
        np.linalg.norm(transform.transform(p) - origin)
        for p in ((diameter, 0), (0, diameter))
    )
    return (pixels * 72 / axis.figure.dpi) ** 2

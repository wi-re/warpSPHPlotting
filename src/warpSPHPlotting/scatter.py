"""Matplotlib scatter lifecycle with shared color normalization."""

import numpy as np
from matplotlib.markers import MarkerStyle
from .enumTypes import VisualizeOptions
from .math import getBounds
from matplotlib.collections import PathCollection
from matplotlib.transforms import IdentityTransform
from ._sizing import particle_diameter


class ParticleCollection(PathCollection):
    """Resolve automatic diameters at draw time, after layout and camera changes."""

    world_diameter = None
    passive = False

    def draw(self, renderer):
        if self.world_diameter is not None:
            origin = self.axes.transData.transform((0, 0))
            diameter_px = min(
                np.linalg.norm(self.axes.transData.transform(p) - origin)
                for p in ((self.world_diameter, 0), (0, self.world_diameter))
            )
            diameter_pts = diameter_px * 72 / self.figure.dpi
            self.set_sizes([diameter_pts**2])
            self.set_linewidth(min(0.5, diameter_pts * 0.15) if self.passive else 0)
        super().draw(renderer)


def updateColorBar(colorbar, mappable):
    """Refresh physical ticks, thinning crowded symlog labels around zero."""
    from matplotlib.colors import SymLogNorm

    colorbar.update_normal(mappable)
    norm = mappable.norm
    if isinstance(norm, SymLogNorm) and norm.vmin < 0 < norm.vmax:
        ticks = [v for v in colorbar.get_ticks() if norm.vmin <= v <= norm.vmax]
        selected = [0.0]
        gap = max(0.04, 18 / max(1, colorbar.ax.bbox.height))
        # Preserve zero and the outer decades before considering ticks in
        # the compressed linear region, whose labels otherwise overlap zero.
        for value in sorted(
            ticks, key=lambda v: abs(float(norm(v)) - 0.5), reverse=True
        ):
            if all(abs(float(norm(value)) - float(norm(v))) >= gap for v in selected):
                selected.append(value)
        colorbar.set_ticks(sorted(selected))


def removeResult(result):
    if result is None:
        return
    sc, cb, _ = result
    if cb is not None:
        cb.remove()
    if sc is not None:
        sc.remove()


def scatterVisualize(
    fig,
    axis,
    particleState,
    domain,
    options,
    variant=VisualizeOptions.Visualize,
    precomputedQs=None,
    precomputedNorm=None,
    attachColorBar=None,
):
    return updateScatterVisualize(
        None,
        fig,
        axis,
        particleState,
        domain,
        options,
        variant,
        precomputedQs,
        precomputedNorm,
        attachColorBar,
    )


def updateScatterVisualize(
    priorResult,
    fig,
    axis,
    particleState,
    domain,
    options,
    variant=VisualizeOptions.Visualize,
    precomputedQs=None,
    precomputedNorm=None,
    attachColorBar=None,
):
    if variant == VisualizeOptions.Hide or not len(particleState.positions):
        removeResult(priorResult)
        return None, None, None
    pos = particleState.positions.detach().cpu().numpy()
    sc, cb, _ = priorResult or (None, None, None)
    passive = variant == VisualizeOptions.Passive
    if passive:
        qs, norm = None, None
    elif precomputedNorm is not None:
        qs, norm = precomputedQs, precomputedNorm
    else:
        qs, norm = getBounds(particleState.quantities, options)
    attach = options.showColorBar and not passive and (attachColorBar is not False)
    if sc is None:
        sc = ParticleCollection(
            [], sizes=[1], offsets=pos[:, :2], offset_transform=axis.transData
        )
        sc.set_transform(IdentityTransform())
        axis.add_collection(sc)
    sc.set_offsets(pos[:, :2])
    sc.passive = passive
    sc.world_diameter = (
        particle_diameter(particleState, options)
        if options.markerSize is None
        else None
    )
    if options.markerSize is not None:
        sc.set_sizes([options.markerSize])
        sc.set_linewidth(1.0)
    marker = MarkerStyle("x" if passive else "o")
    sc.set_paths([marker.get_path().transformed(marker.get_transform())])
    sc.set_alpha(0.5 if passive else None)
    if passive:
        sc.set_array(None)
        sc.set_facecolor("gray")
        sc.set_edgecolor("gray")
    else:
        sc.set_edgecolor("face")
        sc.set_cmap(options.colorMap.value + ("_r" if options.flipColorMap else ""))
        sc.set_norm(norm)
        sc.set_array(qs)
    if cb is not None and not attach:
        cb.remove()
        cb = None
    if attach and cb is None:
        cb = fig.colorbar(sc, ax=axis, pad=0.025)
    if cb is not None:
        updateColorBar(cb, sc)
    return sc, cb, norm

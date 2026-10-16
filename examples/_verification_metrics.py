"""Small, dependency-free resource probes for local rendering checks."""

from pathlib import Path
import os


def resident_bytes():
    """Current Linux resident memory, rather than the process high-water mark."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
            "SC_PAGE_SIZE"
        )
    except (OSError, IndexError, ValueError):
        return None


def scene_counts(plot):
    if plot.backend == "matplotlib":
        return dict(
            axes=len(plot.fig.axes),
            collections=sum(len(a.collections) for a in plot.fig.axes),
            patches=sum(len(a.patches) for a in plot.fig.axes),
            lines=sum(len(a.lines) for a in plot.fig.axes),
        )
    if plot.backend == "pyvista":
        actors = sum(len(r.actors) for r in plot.fig.renderers)
        streams = sum(f"stream_{key}" in s.actors for key, s in plot.plotStates.items())
        return dict(
            actors=actors, streamline_actors=streams, other_actors=actors - streams
        )
    seen = set()

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for child in node.children:
            visit(child)

    visit(plot.fig.scene)
    return dict(nodes=len(seen))

"""Small notebook UI for reviewing real saved frontend frames.

Run interactive.ipynb from the repository root. Missing recordings fall back
to an explicitly labeled analytical Taylor-Green field. No simulation runs in
this UI; changing a control updates the same PlotState.
"""

from collections import deque
import copy
import json
from pathlib import Path
from _artifacts import development_dir, recording_paths
import re
import time
import numpy as np
import torch
import ipywidgets as widgets
from IPython.display import display
from warpSPHCore import DomainDescription, ParticleState
from warpSPHPlotting import (
    ColorMap,
    GridVisualization,
    Mapping,
    PlotScaling,
    PlottingOptions,
    VisualizeOptions,
    visualize,
)


class NotebookReview:
    def __init__(self, root=None, backend="pyvista"):
        self.root = Path(root or Path.cwd())
        self.backend = backend
        self.plot = None
        self.history = deque(maxlen=500)
        self._changing = False
        self._loaded = None
        self._reset_after_update = False
        self._case_viewports = {}
        self.sources = {}
        names = {
            "tgv": "Taylor-Green · periodic",
            "sod": "Sod · sparse viewport",
            "marrone31": "Marrone 3.1 · wall impact",
            "khh": "KHH · multiphase",
        }
        for case, label in names.items():
            paths = recording_paths(case, root=self.root)
            if paths:
                self.sources[label] = paths
        self.sources["Analytical Taylor-Green (not integrated)"] = [None]
        self.case = widgets.Dropdown(
            options=list(self.sources),
            description="Case:",
            layout=widgets.Layout(width="350px"),
        )
        if names["marrone31"] in self.sources:
            self.case.value = names["marrone31"]
        self.frame = widgets.IntSlider(
            description="Frame:",
            min=0,
            max=len(self.sources[self.case.value]) - 1,
            continuous_update=False,
            layout=widgets.Layout(width="550px"),
        )
        self.play = widgets.Play(min=0, max=self.frame.max, interval=500, repeat=True)
        self._play_link = widgets.jslink((self.play, "value"), (self.frame, "value"))
        self.field = widgets.Dropdown(
            options=["Speed", "Density"], description="Field:"
        )
        self.mode = widgets.Dropdown(
            options=["Particles", "Grid", "Streamlines"], description="View:"
        )
        self.cmap = widgets.Dropdown(
            options=["viridis", "plasma", "magma", "coolwarm"], description="Colors:"
        )
        self.scaling = widgets.Dropdown(
            options=["Linear", "SymmetricLog"], description="Scale:"
        )
        self.automatic = widgets.Checkbox(
            value=True, description="Automatic marker sizes"
        )
        self.spacing = widgets.Dropdown(
            options=[("Physical volume", "volume"), ("Close neighbors", "neighbors")],
            description="Size rule:",
            layout=widgets.Layout(width="260px"),
        )
        self.marker = widgets.FloatSlider(
            value=9,
            min=1,
            max=64,
            step=1,
            description="Fixed size:",
            continuous_update=False,
            disabled=True,
        )
        self.boundary = widgets.Checkbox(value=True, description="Show boundaries")
        self.colorbar = widgets.Checkbox(value=True, description="Show colorbar")
        self.dynamic = widgets.Checkbox(
            value=False, description="Follow bounds each frame"
        )
        self.reset = widgets.Button(description="Reset camera")
        self.save = widgets.Button(
            description="Save view + diagnostics", layout=widgets.Layout(width="210px")
        )
        self.status = widgets.HTML()
        self.errors = widgets.Output()
        self.output = widgets.Output(layout=widgets.Layout(width="100%"))
        self.ui = widgets.VBox(
            [
                widgets.HTML(f"<b>{backend} · interactive renderer review</b>"),
                widgets.HBox([self.case, self.field, self.mode]),
                widgets.HBox([self.play, self.frame]),
                widgets.HBox([self.cmap, self.scaling, self.colorbar]),
                widgets.HBox([self.automatic, self.spacing, self.marker]),
                widgets.HBox([self.boundary, self.dynamic]),
                widgets.HBox([self.reset, self.save]),
                self.status,
                self.output,
                self.errors,
            ]
        )
        for row in self.ui.children:
            if isinstance(row, widgets.HBox):
                row.layout.flex_flow = "row wrap"
        self.case.observe(self._case_changed, names="value")
        for control in (
            self.frame,
            self.field,
            self.mode,
            self.cmap,
            self.scaling,
            self.automatic,
            self.spacing,
            self.marker,
            self.boundary,
            self.colorbar,
            self.dynamic,
        ):
            control.observe(self._changed, names="value")
        self.reset.on_click(self._reset)
        self.save.on_click(self._save)

    def show(self):
        display(self.ui)
        self._update()
        return self

    def _load(self):
        key = (self.case.value, self.frame.value, self.dynamic.value)
        if self._loaded is not None and self._loaded[0] == key:
            return self._loaded[1]
        path = self.sources[self.case.value][self.frame.value]
        if path is None:
            from three_dimensions import field

            p, d, v = field(nx=48, dim=2)
        else:
            with np.load(path) as data:
                p = ParticleState(
                    **{
                        name: torch.from_numpy(data[name].copy())
                        for name in (
                            "positions",
                            "supports",
                            "masses",
                            "densities",
                            "kinds",
                        )
                    }
                )
                v = torch.from_numpy(data["velocities"].copy())
                d = DomainDescription(
                    torch.from_numpy(data["domain_min"].copy()),
                    torch.from_numpy(data["domain_max"].copy()),
                    torch.from_numpy(data["periodic"].copy()),
                    2,
                )
            if "Sod" in self.case.value:
                # These stress-test recordings store an intentionally animated
                # display viewport, not a changing physical simulation domain.
                # Keep its initial sparse viewport unless following is enabled.
                if not self.dynamic.value:
                    if self.case.value not in self._case_viewports:
                        first = self.sources[self.case.value][0]
                        with np.load(first) as data:
                            self._case_viewports[self.case.value] = DomainDescription(
                                torch.from_numpy(data["domain_min"].copy()),
                                torch.from_numpy(data["domain_max"].copy()),
                                torch.from_numpy(data["periodic"].copy()),
                                2,
                            )
                    d = self._case_viewports[self.case.value]
                center = (d.min + d.max) / 2
                half = (d.max - d.min) * 3.75
                d = DomainDescription(center - half, center + half, d.periodic, 2)
        self._loaded = (key, (p, d, v))
        return p, d, v

    def _case_changed(self, change):
        self.play.playing = False
        self._changing = True
        self.play.value = self.frame.value = 0
        self.play.max = self.frame.max = len(self.sources[self.case.value]) - 1
        self._changing = False
        self._reset_after_update = True
        self._update()

    def _changed(self, change):
        if not self._changing:
            self.marker.disabled = self.automatic.value
            self.spacing.disabled = not self.automatic.value
            self._changing = True
            if self.mode.value == "Streamlines":
                self.field.value = "Speed"
            self.field.disabled = self.mode.value == "Streamlines"
            self._changing = False
            self._update()

    def _options(self):
        grid = (
            None
            if self.mode.value == "Particles"
            else GridVisualization(
                resolution=96, streamLines=self.mode.value == "Streamlines"
            )
        )
        # Streamlines always integrate the velocity field, and color speed.
        speed = self.field.value == "Speed" or self.mode.value == "Streamlines"
        return speed, PlottingOptions(
            mapping=Mapping.L2 if speed else Mapping.none,
            boundaryVisualization=(
                VisualizeOptions.Passive
                if self.boundary.value
                else VisualizeOptions.Hide
            ),
            gridVisualization=grid,
            markerSize=None if self.automatic.value else self.marker.value,
            colorMap=ColorMap[self.cmap.value],
            quantityScaling=PlotScaling[self.scaling.value],
            quantityLogThreshold=0.01,
            showColorBar=self.colorbar.value,
            plotTitle="Speed" if speed else "Density",
            dynamicBounds=self.dynamic.value,
            markerSpacing=self.spacing.value,
        )

    def _update(self):
        start = time.perf_counter()
        try:
            p, d, velocity = self._load()
            speed, opts = self._options()
            q = velocity if speed else p.densities
            title = f"{self.case.value} · frame {self.frame.value}/{self.frame.max}"
            if self.plot is None:
                beopts = (
                    {"jupyter_backend": "trame", "off_screen": True}
                    if self.backend == "pyvista"
                    else {"jupyter_backend": "notebook"}
                )
                with self.output:
                    self.plot = visualize(
                        p,
                        d,
                        q,
                        opts,
                        backend=self.backend,
                        backendOptions=beopts,
                        figsize=(10, 5),
                        figTitle=title,
                    )
                self._initial_camera = self.camera()
            else:
                overrides = {
                    name: getattr(opts, name) for name in opts.__dataclass_fields__
                }
                # Supplying particle buffers ensures replay exercises the same
                # ownership/invalidation path as a moving simulation.
                self.plot.updateTitle(title)
                self.plot.updateQuantities(
                    q,
                    newParticleState=p,
                    newDomain=d,
                    newOptions={"A": overrides},
                    redraw=True,
                    yieldSeconds=0,
                    resetView=self._reset_after_update,
                )
                if self._reset_after_update:
                    self._initial_camera = self.camera()
                    self._reset_after_update = False
            elapsed = 1000 * (time.perf_counter() - start)
            # Leave time for transport and input between frames. A fixed fast
            # timer queues stale updates when interpolation takes longer.
            self.play.interval = max(500, int(elapsed * 1.5 + 100))
            self.history.append(
                {
                    "case": self.case.value,
                    "frame": self.frame.value,
                    "field": "Speed" if speed else "Density",
                    "view": self.mode.value,
                    "colors": self.cmap.value,
                    "update_ms": elapsed,
                    "particles": len(p.positions),
                    "automatic_sizes": self.automatic.value,
                    "colorbar": self.colorbar.value,
                    "dynamic_bounds": self.dynamic.value,
                    "marker_spacing": self.spacing.value,
                }
            )
            self.status.value = f"<span>{len(p.positions):,} particles · {elapsed:.1f} ms Python update · {self.frame.max+1} saved frames</span>"
            self.errors.clear_output()
        except Exception:
            import traceback

            self.play.playing = False
            with self.errors:
                self.errors.clear_output()
                traceback.print_exc()
            raise

    def camera(self):
        if self.backend == "pyvista":
            return {
                "position": [list(x) for x in self.plot.fig.camera_position],
                "parallel_scale": self.plot.fig.camera.parallel_scale,
                "window_center": list(self.plot.fig.camera.GetWindowCenter()),
            }
        rect = self.plot.plotStates["A"].view.camera.rect
        return {"rect": [rect.left, rect.bottom, rect.width, rect.height]}

    def _reset(self, button):
        if self.backend == "pyvista":
            self.plot.fig.camera_position = self._initial_camera["position"]
            self.plot.fig.camera.parallel_scale = self._initial_camera["parallel_scale"]
            self.plot.fig.camera.SetWindowCenter(*self._initial_camera["window_center"])
        else:
            from vispy.geometry import Rect

            self.plot.plotStates["A"].view.camera.rect = Rect(
                *self._initial_camera["rect"]
            )
        self.plot.show()

    def _save(self, button):
        destination = development_dir(self.root) / "output" / "notebook-review"
        destination.mkdir(parents=True, exist_ok=True)
        case = re.sub(r"[^a-z0-9]+", "-", self.case.value.lower()).strip("-")
        stem = (
            f"{self.backend}-{case}-{self.mode.value.lower()}-"
            f"{self.field.value.lower()}-frame-{self.frame.value:04d}-{time.time_ns()}"
        )
        self.plot.export(str(destination / f"{stem}.png"))
        (destination / f"{stem}.json").write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "camera": self.camera(),
                    "events": list(self.history),
                },
                indent=2,
            )
        )
        self.status.value = f"Saved {destination / stem}.png and diagnostics JSON"

    def close(self):
        self.play.playing = False
        if self.plot is not None:
            self.plot.close()
            self.plot = None
        self._play_link.unlink()
        self.ui.close()

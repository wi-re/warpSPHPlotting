"""Coordinate our windows with trame's synchronous VTK capture boundary.

Notebook execution, widget callbacks and trame's event loop can use different
threads. A completed render/capture must release its native context before the
next thread can acquire it. Window EndEvent is too early: image readback can
make the context current again after Render() returns.
"""

from functools import wraps
from contextlib import contextmanager
import threading
import weakref

_active = threading.local()


@contextmanager
def rendering_context(window, lock):
    """Acquire once and release after the complete outer render/readback call."""
    with lock:
        depths = getattr(_active, "depths", None)
        if depths is None:
            depths = _active.depths = {}
        key = window.GetAddressAsString("")
        depth = depths.get(key, 0)
        depths[key] = depth + 1
        try:
            if depth == 0 and window.GetContextCreationTime():
                window.SetForceMakeCurrent()
                window.MakeCurrent()
            yield
        finally:
            if depth:
                depths[key] = depth
            else:
                depths.pop(key)
                window.ReleaseCurrent()


class TrameContextGuard:
    """Scope capture/interaction cleanup to explicitly registered windows.

    trame-vtk's protocols look up these methods on their shared vtkWebApplication
    for each call. Wrap that instance, leaving other applications and unregistered
    windows alone. Restore the instance when its last managed window closes.
    """

    _methods = ("StillRenderToBuffer", "StillRenderToString", "HandleInteractionEvent")

    def __init__(self, application):
        self.application = application
        self.windows = {}
        self.plotters = {}
        self.originals = {}
        for name in self._methods:
            original = getattr(application, name)
            self.originals[name] = original
            setattr(application, name, self._wrap(original))

    def _wrap(self, original):
        @wraps(original)
        def call(window, *args, **kwargs):
            lock = self.windows.get(window.GetAddressAsString(""))
            if lock is None:
                return original(window, *args, **kwargs)
            with rendering_context(window, lock):
                return original(window, *args, **kwargs)

        return call

    def add(self, window, lock, plotter=None):
        key = window.GetAddressAsString("")
        self.windows[key] = lock
        if plotter is not None and key not in self.plotters:
            # Toolbar camera/axes callbacks invoke Plotter.render directly.
            # They need the same boundary as our backend and VTK image RPCs.
            original = plotter.render
            previous = plotter.__dict__.get("render")

            @wraps(original)
            def render(*args, **kwargs):
                with rendering_context(window, lock):
                    return original(*args, **kwargs)

            plotter.render = render
            self.plotters[key] = (weakref.ref(plotter), previous)

    def remove(self, window):
        key = window.GetAddressAsString("")
        self.windows.pop(key, None)
        restored = self.plotters.pop(key, None)
        if restored is not None:
            plotter = restored[0]()
            if plotter is not None:
                if restored[1] is None:
                    del plotter.render
                else:
                    plotter.render = restored[1]
        if not self.windows and self.originals:
            for name in self._methods:
                delattr(self.application, name)
            self.originals.clear()
            if getattr(self.application, "_warpsph_context_guard", None) is self:
                del self.application._warpsph_context_guard


def register_trame_window(plotter, lock, server_name=None):
    """Use the same shared application as PyVista's server-rendered viewer."""
    from trame.app import get_server
    from trame_vtk.modules import vtk
    import pyvista as pv

    server = get_server(
        server_name
        if server_name is not None
        else pv.global_theme.trame.jupyter_server_name
    )
    server.enable_module(vtk)
    application = vtk.get_helper(server)._vtk_core
    guard = getattr(application, "_warpsph_context_guard", None)
    if guard is None:
        guard = TrameContextGuard(application)
        application._warpsph_context_guard = guard
    guard.add(plotter.render_window, lock, plotter)
    return guard


def serialized_notebook_render(method):
    """Keep scene mutations and remote captures from overlapping on our window."""

    @wraps(method)
    def call(self, *args, **kwargs):
        with self._render_lock:
            notebook = getattr(self, "_jupyter_backend", None) == "trame"
            window = self._plotter.render_window if self._plotter is not None else None
            if notebook and window is not None:
                with rendering_context(window, self._render_lock):
                    return method(self, *args, **kwargs)
            else:
                return method(self, *args, **kwargs)

    return call

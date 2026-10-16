"""Background image writes for frame export.

Rendering a frame is cheap next to encoding it: a 2688x768 PNG takes ~26 ms
to compress even at zlib level 1, on the thread that runs the simulation.
`submitImageWrite` moves the encode + write to a small worker pool (PIL
releases the GIL inside the encoder, so it genuinely overlaps with the step
loop); `waitForExports` blocks until every submitted write has finished and
re-raises the first error, and must be called before the files are read
(e.g. before encoding a video from them).
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List

__all__ = ["submitImageWrite", "waitForExports"]

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pending: List[Future] = []
_first_error: BaseException | None = None
#: bound on queued frames, so a renderer faster than the encoder cannot grow
#: memory without limit (each queued frame holds its pixel buffer)
MAX_PENDING = 8


def _write(img, path: str, kwargs: Dict[str, Any]) -> None:
    img.save(path, **kwargs)


def submitImageWrite(img, path: str, kwargs: Dict[str, Any]) -> None:
    global _pool, _first_error
    # Serialize queue admission so concurrent producers cannot exceed the bound.
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frameWriter")
        # Retain the first error without retaining completed image buffers.
        active = []
        for f in _pending:
            if not f.done():
                active.append(f)
            else:
                error = f.exception()
                if error is not None and _first_error is None:
                    _first_error = error.with_traceback(None)
        _pending[:] = active
        active = [f for f in _pending if not f.done()]
        if len(active) >= MAX_PENDING:
            active[0].result()
        _pending.append(_pool.submit(_write, img, path, dict(kwargs)))


def waitForExports() -> None:
    global _first_error
    with _lock:
        pending = list(_pending)
        _pending.clear()
        first_error = _first_error
        _first_error = None
    for f in pending:
        try:
            f.result()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error

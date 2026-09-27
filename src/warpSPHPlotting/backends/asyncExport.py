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

__all__ = ['submitImageWrite', 'waitForExports']

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pending: List[Future] = []
#: bound on queued frames, so a renderer faster than the encoder cannot grow
#: memory without limit (each queued frame holds its pixel buffer)
MAX_PENDING = 8


def _write(img, path: str, kwargs: Dict[str, Any]) -> None:
    img.save(path, **kwargs)


def submitImageWrite(img, path: str, kwargs: Dict[str, Any]) -> None:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='frameWriter')
        pending = [f for f in _pending if not f.done()]
        _pending[:] = pending
    if len(pending) >= MAX_PENDING:
        pending[0].result()
    fut = _pool.submit(_write, img, path, kwargs)
    with _lock:
        _pending.append(fut)


def waitForExports() -> None:
    with _lock:
        pending = list(_pending)
        _pending.clear()
    for f in pending:
        f.result()

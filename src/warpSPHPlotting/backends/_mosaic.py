"""Validate the rectangular subset of Matplotlib's string mosaic syntax."""


def parse_mosaic(mosaic):
    if not isinstance(mosaic, str) or not mosaic.strip():
        raise ValueError("mosaic must be a nonempty string")
    rows = mosaic.strip().replace(";", "\n").splitlines()
    if len({len(row) for row in rows}) != 1:
        raise ValueError("mosaic rows must have equal lengths")
    axes = {}
    spans = {}
    for r, row in enumerate(rows):
        for c, key in enumerate(row):
            if key == "." or key.isspace():
                continue
            axes.setdefault(key, (r, c))
    for key in axes:
        cells = [
            (r, c)
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value == key
        ]
        r0 = min(r for r, c in cells)
        r1 = max(r for r, c in cells)
        c0 = min(c for r, c in cells)
        c1 = max(c for r, c in cells)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise ValueError(f"mosaic panel {key!r} must be rectangular")
        spans[key] = (r0, c0, r1, c1)
    if not axes:
        raise ValueError("mosaic must contain at least one panel")
    return (len(rows), len(rows[0])), axes, spans

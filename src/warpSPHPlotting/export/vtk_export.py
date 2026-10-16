"""Dependency-light VTK XML export, independent of graphics backends."""

from pathlib import Path
import json
import os
import xml.etree.ElementTree as ET
import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _array(parent, name, value):
    a = _numpy(value)
    if a.dtype.kind not in "fiu":
        raise ValueError(f"VTK field {name!r} must be numeric.")
    components = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
    vtk_type = (
        "Float64"
        if a.dtype.kind == "f"
        else ("UInt64" if a.dtype.kind == "u" else "Int64")
    )
    node = ET.SubElement(
        parent,
        "DataArray",
        type=vtk_type,
        Name=str(name),
        NumberOfComponents=str(components),
        format="ascii",
    )
    node.text = " ".join(
        format(float(x), ".17g") if a.dtype.kind == "f" else str(int(x))
        for x in a.ravel()
    )
    return node


def _metadata(dataset, metadata):
    if not metadata:
        return
    fields = ET.SubElement(dataset, "FieldData")
    # VTK string arrays are ASCII code points terminated by zero.
    text = json.dumps(metadata, sort_keys=True)
    node = ET.SubElement(
        fields,
        "Array",
        type="String",
        Name="metadata",
        NumberOfTuples="1",
        format="ascii",
    )
    node.text = " ".join(str(x) for x in text.encode("utf-8") + b"\0")


def _save(root, path, suffix):
    path = Path(path)
    if path.suffix != suffix:
        raise ValueError(f"Expected a {suffix} path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return str(path)


def export_particles_vtp(path, particleState, quantities, domain=None, metadata=None):
    points = _numpy(particleState.positions)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Particle positions must be N x 2 or N x 3.")
    if not np.isfinite(points).all():
        raise ValueError("Particle positions must be finite.")
    n = len(points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(n)))
    root = ET.Element(
        "VTKFile", type="PolyData", version="1.0", byte_order="LittleEndian"
    )
    data = ET.SubElement(root, "PolyData")
    info = dict(metadata or {})
    if domain is not None:
        info["domain"] = {
            "min": _numpy(domain.min).tolist(),
            "max": _numpy(domain.max).tolist(),
            "periodic": _numpy(domain.periodic).tolist(),
        }
    _metadata(data, info)
    piece = ET.SubElement(
        data,
        "Piece",
        NumberOfPoints=str(n),
        NumberOfVerts=str(n),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys="0",
    )
    _array(ET.SubElement(piece, "Points"), "Points", points)
    fields = ET.SubElement(piece, "PointData")
    for name, q in quantities.items():
        a = _numpy(q)
        if a.ndim == 0 or len(a) != n:
            raise ValueError(f"{name!r} must have {n} samples.")
        _array(fields, name, a)
    verts = ET.SubElement(piece, "Verts")
    _array(verts, "connectivity", np.arange(n, dtype=np.int64))
    _array(verts, "offsets", np.arange(1, n + 1, dtype=np.int64))
    return _save(root, path, ".vtp")


def export_grid_vti(path, gridState, gridQuantity, resolution, extent, metadata=None):
    n = tuple(int(x) for x in resolution)
    if len(n) not in (2, 3) or any(x < 2 for x in n):
        raise ValueError("resolution must have 2 or 3 axes with >= 2 points.")
    points = _numpy(gridState.positions)
    if len(points) != np.prod(n):
        raise ValueError("Grid point count does not match resolution.")
    shaped = points.reshape(n + (len(n),))
    origin = points[0]
    spacing = np.array(
        [
            shaped[tuple(1 if j == i else 0 for j in range(len(n)))][i] - origin[i]
            for i in range(len(n))
        ]
    )
    if not np.isfinite(points).all() or not np.all(spacing > 0):
        raise ValueError("Grid coordinates must be finite and increasing.")
    expected = np.stack(
        np.meshgrid(
            *[origin[i] + spacing[i] * np.arange(n[i]) for i in range(len(n))],
            indexing="ij",
        ),
        -1,
    ).reshape(points.shape)
    if not np.allclose(points, expected, rtol=1e-5, atol=1e-7):
        raise ValueError("VTI requires a regular, axis-aligned grid.")
    dims = n + ((1,) if len(n) == 2 else ())
    sp = tuple(spacing) + ((1.0,) if len(n) == 2 else ())
    org = tuple(origin) + ((0.0,) if len(n) == 2 else ())
    whole = " ".join(str(x) for count in dims for x in (0, count - 1))
    root = ET.Element(
        "VTKFile", type="ImageData", version="1.0", byte_order="LittleEndian"
    )
    data = ET.SubElement(
        root,
        "ImageData",
        WholeExtent=whole,
        Origin=" ".join(map(str, org)),
        Spacing=" ".join(map(str, sp)),
    )
    _metadata(data, metadata)
    piece = ET.SubElement(data, "Piece", Extent=whole)
    fields = ET.SubElement(piece, "PointData")
    arrays = (
        gridQuantity if isinstance(gridQuantity, dict) else {"quantity": gridQuantity}
    )
    for name, q in arrays.items():
        a = _numpy(q)
        if a.ndim == 0 or len(a) != np.prod(n):
            raise ValueError(f"{name!r} has the wrong grid sample count.")
        components = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
        ordered = (
            a.reshape(n + (components,))
            .transpose(tuple(reversed(range(len(n)))) + (len(n),))
            .reshape(-1, components)
        )
        _array(fields, name, ordered)
    return _save(root, path, ".vti")


def write_time_series_pvd(path, entries):
    """Write `(simulation_time, VTK_path)` entries using relative file links."""
    root = ET.Element(
        "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
    )
    collection = ET.SubElement(root, "Collection")
    previous = -np.inf
    for time, filename in entries:
        time = float(time)
        if not np.isfinite(time) or time < previous:
            raise ValueError("Series times must be finite and nondecreasing.")
        previous = time
        file = Path(filename).resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        ET.SubElement(
            collection,
            "DataSet",
            timestep=str(time),
            group="",
            part="0",
            file=os.path.relpath(file, Path(path).resolve().parent),
        )
    return _save(root, path, ".pvd")

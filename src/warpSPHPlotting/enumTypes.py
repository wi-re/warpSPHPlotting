from .colorMaps import ColorMap
from enum import Enum


class Backend(str, Enum):
    """Rendering backend selector.

    Pass either the enum member or its string value to the ``backend``
    parameter of :func:`visualize`.  Using ``str, Enum`` as the base
    allows transparent string comparisons so legacy code passing plain
    strings continues to work.
    """

    Auto = "auto"
    Matplotlib = "matplotlib"
    PyVista = "pyvista"
    Vispy = "vispy"


class VisualizeOptions(Enum):
    Hide = 0
    Visualize = 1
    Passive = 2


class PlotScaling(Enum):
    Linear = "linear"
    Logarithmic = "log"
    SymmetricLog = "symlog"
    Symmetric = "sym"


class Mapping(Enum):
    none = "none"
    x = "x"
    y = "y"
    z = "z"
    L2Norm = "L2Norm"
    L2 = "L2"
    L1 = "L1"
    magnitude = "magnitude"


class StreamLineLocation(Enum):
    BeforeOperation = 0
    BeforeMapping = 1
    AfterMapping = 2

from enum import Enum

uniformColorMaps = ["viridis", "plasma", "inferno", "magma", "cividis"]


class UniformColorMap(Enum):
    viridis = "viridis"
    plasma = "plasma"
    inferno = "inferno"
    magma = "magma"
    cividis = "cividis"
    # seaborn colormaps (if seaborn is available)
    rocket = "rocket"
    mako = "mako"
    flare = "flare"
    crest = "crest"


sequentialColorMaps = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
]


class SequentialColorMap(Enum):
    Greys = "Greys"
    Purples = "Purples"
    Blues = "Blues"
    Greens = "Greens"
    Oranges = "Oranges"
    Reds = "Reds"
    YlOrBr = "YlOrBr"
    YlOrRd = "YlOrRd"
    OrRd = "OrRd"
    PuRd = "PuRd"
    RdPu = "RdPu"
    BuPu = "BuPu"
    GnBu = "GnBu"
    PuBu = "PuBu"
    YlGnBu = "YlGnBu"
    PuBuGn = "PuBuGn"
    BuGn = "BuGn"
    YlGn = "YlGn"


sequential2ColorMaps = [
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
]


class Sequential2ColorMap(Enum):
    binary = "binary"
    gist_yarg = "gist_yarg"
    gist_gray = "gist_gray"
    gray = "gray"
    bone = "bone"
    pink = "pink"
    spring = "spring"
    summer = "summer"
    autumn = "autumn"
    winter = "winter"
    cool = "cool"
    Wistia = "Wistia"
    hot = "hot"
    afmhot = "afmhot"
    gist_heat = "gist_heat"
    copper = "copper"


divergingColorMaps = [
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "berlin",
    "managua",
    "vanimo",
]


class DivergingColorMap(Enum):
    PiYG = "PiYG"
    PRGn = "PRGn"
    BrBG = "BrBG"
    PuOr = "PuOr"
    RdGy = "RdGy"
    RdBu = "RdBu"
    RdYlBu = "RdYlBu"
    RdYlGn = "RdYlGn"
    Spectral = "Spectral"
    coolwarm = "coolwarm"
    bwr = "bwr"
    seismic = "seismic"
    berlin = "berlin"
    managua = "managua"
    vanimo = "vanimo"
    # seaborn colormaps (if seaborn is available)
    vlag = "vlag"
    icefire = "icefire"


cyclicColorMaps = ["twilight", "twilight_shifted", "hsv"]


class CyclicColorMap(Enum):
    twilight = "twilight"
    twilight_shifted = "twilight_shifted"
    hsv = "hsv"


qualitativeColorMaps = [
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]


class QualitativeColorMap(Enum):
    Pastel1 = "Pastel1"
    Pastel2 = "Pastel2"
    Paired = "Paired"
    Accent = "Accent"
    Dark2 = "Dark2"
    Set1 = "Set1"
    Set2 = "Set2"
    Set3 = "Set3"
    tab10 = "tab10"
    tab20 = "tab20"
    tab20b = "tab20b"
    tab20c = "tab20c"


miscellaneousColorMaps = [
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "turbo",
    "nipy_spectral",
    "gist_ncar",
]


class MiscellaneousColorMap(Enum):
    flag = "flag"
    prism = "prism"
    ocean = "ocean"
    gist_earth = "gist_earth"
    terrain = "terrain"
    gist_stern = "gist_stern"
    gnuplot = "gnuplot"
    gnuplot2 = "gnuplot2"
    CMRmap = "CMRmap"
    cubehelix = "cubehelix"
    brg = "brg"
    gist_rainbow = "gist_rainbow"
    rainbow = "rainbow"
    jet = "jet"
    turbo = "turbo"
    nipy_spectral = "nipy_spectral"
    gist_ncar = "gist_ncar"


seabornColorMaps = [
    "rocket",
    "mako",
    "flare",
    "crest",  # Perceptually uniform sequential
    "vlag",
    "icefire",  # Perceptually uniform diverging
    # "Spectral", "coolwarm" # Perceptually uniform diverging (also in matplotlib)
]

all_color_maps = (
    uniformColorMaps
    + sequentialColorMaps
    + sequential2ColorMaps
    + divergingColorMaps
    + cyclicColorMaps
    + qualitativeColorMaps
    + miscellaneousColorMaps
)

try:
    import seaborn as sns

    all_color_maps += seabornColorMaps
except ImportError:
    pass


# Define an enum to hold the different colormaps for easy access
class ColorMap(Enum):
    # Uniform colormaps
    viridis = "viridis"
    plasma = "plasma"
    inferno = "inferno"
    magma = "magma"
    cividis = "cividis"

    # Sequential colormaps
    Greys = "Greys"
    Purples = "Purples"
    Blues = "Blues"
    Greens = "Greens"
    Oranges = "Oranges"
    Reds = "Reds"
    YlOrBr = "YlOrBr"
    YlOrRd = "YlOrRd"
    OrRd = "OrRd"
    PuRd = "PuRd"
    RdPu = "RdPu"
    BuPu = "BuPu"
    GnBu = "GnBu"
    PuBu = "PuBu"
    YlGnBu = "YlGnBu"
    PuBuGn = "PuBuGn"
    BuGn = "BuGn"
    YlGn = "YlGn"

    # Sequential2 colormaps
    binary = "binary"
    gist_yarg = "gist_yarg"
    gist_gray = "gist_gray"
    gray = "gray"
    bone = "bone"
    pink = "pink"
    spring = "spring"
    summer = "summer"
    autumn = "autumn"
    winter = "winter"
    cool = "cool"
    Wistia = "Wistia"
    hot = "hot"
    afmhot = "afmhot"
    gist_heat = "gist_heat"
    copper = "copper"

    # Diverging colormaps
    PiYG = "PiYG"
    PRGn = "PRGn"
    BrBG = "BrBG"
    PuOr = "PuOr"
    RdGy = "RdGy"
    RdBu = "RdBu"
    RdYlBu = "RdYlBu"
    RdYlGn = "RdYlGn"
    Spectral = "Spectral"
    coolwarm = "coolwarm"
    bwr = "bwr"
    seismic = "seismic"
    berlin = "berlin"
    managua = "managua"
    vanimo = "vanimo"

    # Cyclic colormaps
    twilight = "twilight"
    twilight_shifted = "twilight_shifted"
    hsv = "hsv"

    # Qualitative colormaps
    Pastel1 = "Pastel1"
    Pastel2 = "Pastel2"
    Paired = "Paired"
    Accent = "Accent"
    Dark2 = "Dark2"
    Set1 = "Set1"
    Set2 = "Set2"
    Set3 = "Set3"
    tab10 = "tab10"
    tab20 = "tab20"
    tab20b = "tab20b"
    tab20c = "tab20c"

    # Miscellaneous colormaps
    flag = "flag"
    prism = "prism"
    ocean = "ocean"
    gist_earth = "gist_earth"
    terrain = "terrain"
    gist_stern = "gist_stern"
    gnuplot = "gnuplot"
    gnuplot2 = "gnuplot2"
    CMRmap = "CMRmap"
    cubehelix = "cubehelix"
    brg = "brg"
    gist_rainbow = "gist_rainbow"
    rainbow = "rainbow"
    jet = "jet"
    turbo = "turbo"
    nipy_spectral = "nipy_spectral"
    gist_ncar = "gist_ncar"

    # Seaborn colormaps (if seaborn is available)
    rocket = "rocket"
    mako = "mako"
    flare = "flare"
    crest = "crest"
    vlag = "vlag"
    icefire = "icefire"

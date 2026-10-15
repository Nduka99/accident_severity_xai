"""One chart style for every notebook figure.

Colours are the validated reference palette from the dataviz method: categorical hues in
a fixed order (never cycled), a one-hue blue ramp for magnitude, and blue<->red with a
gray midpoint for signed values such as SHAP contributions.
"""

import logging

import matplotlib as mpl
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots, in order. Scatter and map forms use only the first three.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
DIVERGING_MID = "#f0efec"
RED_ARM = "#e34948"
BLUE_ARM = "#2a78d6"

SEQUENTIAL = LinearSegmentedColormap.from_list("accidents_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "accidents_diverging", [BLUE_ARM, DIVERGING_MID, RED_ARM]
)


def use_style() -> None:
    """Quiet chrome so the data is the loudest thing: hairline solid grid, no top/right
    spines, 2px lines, text in ink colours."""
    # The fallback font has no semibold weight; matplotlib substitutes bold and would log
    # that on every title lookup.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            # A list turns on per-glyph fallback: DejaVu Sans (bundled with matplotlib) draws
            # symbols such as ∈ that the preferred font lacks.
            "font.family": ["sans-serif", "DejaVu Sans"],
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "image.cmap": "accidents_blue",
        }
    )
    if "accidents_blue" not in mpl.colormaps:
        mpl.colormaps.register(SEQUENTIAL)
        mpl.colormaps.register(DIVERGING)


def subtitle(ax: Axes, text: str) -> None:
    """A secondary line under the title that says what to read from the chart."""
    ax.text(0, 1.01, text, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom")

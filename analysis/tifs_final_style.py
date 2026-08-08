"""Shared visual constants for the final IEEE TIFS PRISM-Gait figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


STYLE_PATH = Path(__file__).with_name("publication-hybrid.mplstyle")

PRISM_COLOR = "#355C7D"
PRISM_LIGHT = "#7EA6C2"
PRISM_PALE = "#E3EDF4"
SMALL_SWAY_COLOR = "#2A9D8F"
SMALL_SWAY_LIGHT = "#9FD2CA"
SMALL_SWAY_PALE = "#D5EBE7"
LARGE_SWAY_COLOR = "#8E6C8A"
LARGE_SWAY_LIGHT = "#B89FB4"
LARGE_SWAY_PALE = "#DACDD8"
BASELINE_DARK = "#737373"
BASELINE_MID = "#8A8A8A"
BASELINE_LIGHT = "#BDBDBD"
TEXT_COLOR = "#1A1A1A"

# One explicit final-size hierarchy shared by every results figure.
AXIS_LABEL_SIZE = 10.5
LEGEND_SIZE = 10.0
TICK_SIZE = 9.0
PANEL_LABEL_SIZE = 12.5
HEADER_SIZE = 9.5
VALUE_SIZE = 9.0


def apply_final_style() -> None:
    """Load the publication style and enforce one cross-figure type hierarchy."""
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"publication style not found: {STYLE_PATH}")
    plt.style.use(STYLE_PATH)
    mpl.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.handlelength": 1.8,
            "legend.handleheight": 1.0,
            "legend.markerscale": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )

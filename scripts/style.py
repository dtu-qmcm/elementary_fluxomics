"""Shared plotting style for publication figures.

Usage:
    from style import apply_style, COLORS, METABOLITE_COLORS, SINGLE_COL, DOUBLE_COL

    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.6))
"""

import matplotlib.pyplot as plt
from cmcrameri import cm

# ---------------------------------------------------------------------------
# Column widths (inches) — typical two-column journal dimensions
# ---------------------------------------------------------------------------
SINGLE_COL = 3.5   # ~89 mm
DOUBLE_COL = 7.2   # ~183 mm

# ---------------------------------------------------------------------------
# Condition colors — batlow samples with wider perceptual spacing
# ---------------------------------------------------------------------------
# Previous: 0.15, 0.50, 0.85 — navy and olive were too close in luminance.
# Now: 0.05, 0.50, 0.75 — dark blue / olive-green / warm orange.
COLORS = {
    "No Labeling": cm.batlow(0.05),
    "Steady-State": cm.batlow(0.50),
    "Instationary": cm.batlow(0.75),
}

# Ordered list matching the plotting order (background → foreground)
COLOR_ORDER = ["No Labeling", "Steady-State", "Instationary"]

# ---------------------------------------------------------------------------
# Metabolite colors — four-level batlow for per-metabolite plots
# ---------------------------------------------------------------------------
METABOLITE_COLORS = {
    "g6p":  cm.batlow(0.10),
    "f6p":  cm.batlow(0.38),
    "dhap": cm.batlow(0.60),
    "g3p":  cm.batlow(0.80),
}

METABOLITES = ["g6p", "f6p", "dhap", "g3p"]

# ---------------------------------------------------------------------------
# Apply rcParams
# ---------------------------------------------------------------------------
def apply_style():
    """Set matplotlib rcParams for consistent publication figures."""
    plt.rcParams.update({
        # Font
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        # Axes
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,

        # Ticks
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        # Legend
        "legend.fontsize": 7,
        "legend.frameon": False,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Lines / markers
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    })

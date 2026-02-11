"""
Plot combined 2x2 Gibbs landscape heatmaps.

Layout:
| NE Steady-State  | FFE Steady-State  |
| NE Timecourse    | FFE Timecourse    |

Shared color axis across all panels, single colorbar on right.
NLL converted to LL (multiply by -1), timecourse normalized by n_timepoints (100).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm

from style import apply_style, DOUBLE_COL

apply_style()

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
FFE_DIR = RESULTS_DIR / "gibbs_landscape_2d_ffe"
NE_DIR = RESULTS_DIR / "gibbs_landscape_2d_ne"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Number of timepoints for normalization
N_TIMEPOINTS = 100


def load_and_pivot(csv_path: Path):
    """Load CSV and pivot to 2D arrays for heatmap."""
    df = pd.read_csv(csv_path)

    # Get unique values for axes
    fba1_vals = np.sort(df["FBA1_deltaG"].unique())
    fba2_vals = np.sort(df["FBA2_deltaG"].unique())

    # Create pivot tables
    # Convert NLL to LL (multiply by -1)
    # Normalize timecourse by n_timepoints
    ll_final = df.pivot(
        index="FBA1_deltaG", columns="FBA2_deltaG", values="nll_final"
    ).values * -1

    ll_timecourse = df.pivot(
        index="FBA1_deltaG", columns="FBA2_deltaG", values="nll_timecourse"
    ).values * -1 / N_TIMEPOINTS

    return fba1_vals, fba2_vals, ll_final, ll_timecourse


def main():
    """Generate combined 2x2 heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    ne_fba1, ne_fba2, ne_ll_ss, ne_ll_tc = load_and_pivot(NE_DIR / "gibbs_landscape_2d.csv")
    ffe_fba1, ffe_fba2, ffe_ll_ss, ffe_ll_tc = load_and_pivot(FFE_DIR / "gibbs_landscape_2d.csv")

    # Compute separate color scales per column (NE vs FFE have very different ranges)
    ne_vmin = min(ne_ll_ss.min(), ne_ll_tc.min())
    ne_vmax = max(ne_ll_ss.max(), ne_ll_tc.max())
    ffe_vmin = min(ffe_ll_ss.min(), ffe_ll_tc.min())
    ffe_vmax = max(ffe_ll_ss.max(), ffe_ll_tc.max())

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))

    # Data mapping: (row, col) -> (fba1, fba2, data, vmin, vmax)
    panels = {
        (0, 0): (ne_fba1, ne_fba2, ne_ll_ss, ne_vmin, ne_vmax),    # NE Steady-State
        (0, 1): (ffe_fba1, ffe_fba2, ffe_ll_ss, ffe_vmin, ffe_vmax), # FFE Steady-State
        (1, 0): (ne_fba1, ne_fba2, ne_ll_tc, ne_vmin, ne_vmax),    # NE Timecourse
        (1, 1): (ffe_fba1, ffe_fba2, ffe_ll_tc, ffe_vmin, ffe_vmax), # FFE Timecourse
    }

    images = {}
    for (row, col), (fba1_vals, fba2_vals, data, vmin, vmax) in panels.items():
        ax = axes[row, col]

        # imshow expects (rows, cols) where rows are y-axis
        # Our pivot has FBA1 as rows (index) and FBA2 as columns
        # So FBA1 is y-axis, FBA2 is x-axis
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cm.batlow,
            vmin=vmin,
            vmax=vmax,
            extent=[fba2_vals.min(), fba2_vals.max(), fba1_vals.min(), fba1_vals.max()],
        )
        images[(row, col)] = im

        # Heatmaps need all four spines
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        if row == 1:
            ax.set_xlabel(r"FBA2 $\Delta G$ (kJ/mol)")

        if col == 0:
            ax.set_ylabel(r"FBA1 $\Delta G$ (kJ/mol)")

    # Add separate colorbars for NE and FFE below each column
    fig.subplots_adjust(bottom=0.18, hspace=0.25, wspace=0.25)
    # NE colorbar (below left column)
    cbar_ax_ne = fig.add_axes([0.12, 0.06, 0.35, 0.02])
    fig.colorbar(images[(0, 0)], cax=cbar_ax_ne, orientation="horizontal", label="LL (NE)")
    # FFE colorbar (below right column)
    cbar_ax_ffe = fig.add_axes([0.57, 0.06, 0.35, 0.02])
    fig.colorbar(images[(0, 1)], cax=cbar_ax_ffe, orientation="horizontal", label="LL (FFE)")

    # Save outputs
    png_path = FIGURES_DIR / "gibbs_landscape_2d_combined.png"
    pdf_path = FIGURES_DIR / "gibbs_landscape_2d_combined.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

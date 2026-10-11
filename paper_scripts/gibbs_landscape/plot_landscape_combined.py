"""
Plot combined 2x2 Gibbs landscape heatmaps.

Layout:
| NE steady state  | FFE steady state  |
| NE instationary  | FFE instationary  |

Colour is the log-likelihood (LL = -NLL) of each FBA1/FBA2 ΔG pair against the
reference (equal split, black cross); brighter is more likely. The instationary
LL sums over every timepoint, each measuring the same isotopologues as the
steady-state experiment, so it is divided by the number of timepoints to put it
on the steady-state scale. NE and FFE each get their own colour scale, shared by
their SS and INST panels, with the colourbar under the instationary panel.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from cmcrameri import cm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "common"))
from style import apply_style, DOUBLE_COL
from gibbs_landscape_2d import MODE_CONFIGS, TIMEPOINTS, get_reference_condition

apply_style()

FIGURES_DIR = MODE_CONFIGS["ne"]["results_dir"].parent / "figures"

# Instationary measurements per steady-state measurement
N_TIMEPOINTS = len(TIMEPOINTS)

COLUMNS = {"ne": "Near equilibrium", "ffe": "Far from equilibrium"}
ROWS = {"ss": "Steady state", "inst": "Instationary"}

X_LABEL = r"$\Delta_r G_{\mathrm{FBA}_2}$ $\left(\frac{kJ}{mol}\right)$"
Y_LABEL = r"$\Delta_r G_{\mathrm{FBA}_1}$ $\left(\frac{kJ}{mol}\right)$"


def load_landscape(csv_path: Path):
    """Load a landscape CSV and pivot the SS and INST log-likelihoods to 2D.

    Rows are FBA1 ΔG (y-axis), columns FBA2 ΔG (x-axis).
    """
    df = pd.read_csv(csv_path)
    ll_ss = -df.pivot(index="FBA1_deltaG", columns="FBA2_deltaG", values="nll_final")
    ll_inst = -df.pivot(index="FBA1_deltaG", columns="FBA2_deltaG", values="nll_timecourse")
    return {"ss": ll_ss, "inst": ll_inst / N_TIMEPOINTS}


def main():
    """Generate combined 2x2 heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.045])

    for col, (eq, eq_label) in enumerate(COLUMNS.items()):
        landscapes = load_landscape(MODE_CONFIGS[eq]["results_dir"] / "gibbs_landscape_2d.csv")
        ref_fba1, ref_fba2 = get_reference_condition(MODE_CONFIGS[eq]["total_fba_dg"])

        # One colour scale per equilibrium, shared by its SS and INST panels
        norm = plt.Normalize(
            vmin=min(ll.values.min() for ll in landscapes.values()),
            vmax=max(ll.values.max() for ll in landscapes.values()),
        )

        top_ax = None
        for row, (sim_mode, mode_label) in enumerate(ROWS.items()):
            ax = fig.add_subplot(gs[row, col], sharex=top_ax)
            top_ax = top_ax or ax
            ll = landscapes[sim_mode]

            mesh = ax.pcolormesh(
                ll.columns, ll.index, ll.values,
                shading="nearest", cmap=cm.batlow, norm=norm, rasterized=True,
            )
            ax.plot(ref_fba2, ref_fba1, marker="x", color="black", ms=5, mew=1.2,
                    linestyle="none")
            ax.set_aspect("equal")

            # Heatmaps need all four spines
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            if row == 0:
                ax.set_title(eq_label)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(X_LABEL)

            if col == 0:
                ax.set_ylabel(Y_LABEL)
                ax.annotate(
                    mode_label, xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-50, 0), textcoords="offset points",
                    rotation=90, ha="right", va="center",
                    fontsize=plt.rcParams["axes.titlesize"],
                )

        cbar = fig.colorbar(mesh, cax=fig.add_subplot(gs[2, col]), orientation="horizontal")
        cbar.set_label("Log-likelihood")

    for ext in ("png", "pdf", "svg"):
        out = FIGURES_DIR / f"gibbs_landscape_2d_combined.{ext}"
        fig.savefig(out)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

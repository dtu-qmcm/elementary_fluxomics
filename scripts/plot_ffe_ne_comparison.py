"""
Plot FFE vs NE comparison as a 2x2 figure showing isotope fraction differences.

Layout:
| NE Steady-State  | FFE Steady-State  |
| NE Instationary  | FFE Instationary  |

Each panel shows 4 lines (one per metabolite: g6p, f6p, dhap, g3p).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from style import apply_style, METABOLITE_COLORS, METABOLITES, SINGLE_COL

apply_style()

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
FFE_DIR = RESULTS_DIR / "gibbs_fba_split"
NE_DIR = RESULTS_DIR / "gibbs_fba_split_ne"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_data():
    """Load FFE and NE isotope difference data."""
    ffe_isotope = pd.read_csv(FFE_DIR / "max_isotope_diff_summary.csv")
    ne_isotope = pd.read_csv(NE_DIR / "max_isotope_diff_summary.csv")
    return ffe_isotope, ne_isotope


def main():
    """Generate 2x2 comparison plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ffe_isotope, ne_isotope = load_data()

    # Create 2x2 figure — single column width
    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COL, SINGLE_COL * 1.1), sharey="row")

    # Data mapping: (row, col) -> (data, column_prefix)
    panels = {
        (0, 0): (ne_isotope, "max_diff_ss"),   # NE Steady-State
        (0, 1): (ffe_isotope, "max_diff_ss"),  # FFE Steady-State
        (1, 0): (ne_isotope, "max_diff_tc"),   # NE Instationary
        (1, 1): (ffe_isotope, "max_diff_tc"),  # FFE Instationary
    }

    for (row, col), (data, prefix) in panels.items():
        ax = axes[row, col]

        for metabolite in METABOLITES:
            col_name = f"{prefix}_{metabolite}"
            ax.plot(
                data["fraction"],
                data[col_name],
                color=METABOLITE_COLORS[metabolite],
                marker="o",
                markersize=2,
                label=metabolite if (row == 0 and col == 0) else None,
            )

        if row == 1:
            ax.set_xlabel("FBA1 fraction")

        if col == 0:
            ax.set_ylabel(r"max($MID_\mathrm{ref} - MID_\mathrm{agg}$)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc='upper right')
    fig.tight_layout()

    # Save outputs
    png_path = FIGURES_DIR / "ffe_ne_comparison.png"
    pdf_path = FIGURES_DIR / "ffe_ne_comparison.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot flux error heatmaps from fitted results.

Reads saved CSVs from results/fit_mech_vs_agg/ and generates heatmaps
comparing fitted vs true forward and reverse fluxes across alpha partitions
for FFE and NE conditions.
"""

from pathlib import Path
import sys

import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
STD_MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_gly3p"
RESULTS_DIR = PROJECT_DIR / "results" / "fit_mech_vs_agg"

R = 8.314e-3  # Gas constant in kJ/(mol*K)
CONVERSION_FACTOR = 1000 / 3600  # mmol/gDCW/h -> umol/gCDW/s

# True net fluxes (mmol/gDCW/h, converted to umol/gCDW/s)
TRUE_NET_FLUXES_MMOL = {
    "glc_input": 100.0043,
    "GLUT2": 100.0043,
    "HEX": 100.0043,
    "GPI": 82.5514,
    "PFK": 93.52,
    "FBA": 93.52,
    "GLY3PDH": 5.0,
    "TPI": 88.52,
    "GAPD_PGK": 187.5243,
    "PGM": 187.5243,
    "PYK": 187.5243,
    "pyr_drain": 87.5243,
    "LDH": 100.0,
    "G6PDH2r_PGL": 17.4529,
    "GND": 17.4529,
    "RPI": 6.4843,
    "RPE": 10.9686,
    "r5p_drain": 1.0,
    # Transketolase/transaldolase half reactions via the enzyme-bound EC2 and
    # EC3 carriers. Negative means the half reaction runs backwards relative to
    # how it is written in reactions.tsv.
    "X5P_EC2": 10.9686,
    "F6P_EC2": -5.4843,
    "S7P_EC2": -5.4843,
    "F6P_EC3": -5.4843,
    "S7P_EC3": 5.4843,
}

TRUE_NET_FLUXES = {k: v * CONVERSION_FACTOR for k, v in TRUE_NET_FLUXES_MMOL.items()}

# Reversible reactions in the aggregated model
REVERSIBLE_RXNS = [
    "GLUT2", "HEX", "GPI", "PFK", "FBA", "TPI",
    "GAPD_PGK", "PGM", "PYK",
    "G6PDH2r_PGL", "RPI", "RPE",
    "X5P_EC2", "F6P_EC2", "S7P_EC2", "F6P_EC3", "S7P_EC3",
]

# Reaction display order
REACTION_ORDER_NET = [
    "glc_input", "GLUT2", "HEX", "GPI", "PFK", "FBA", "GLY3PDH", "TPI",
    "GAPD_PGK", "PGM", "PYK", "pyr_drain", "LDH",
    "G6PDH2r_PGL", "GND", "RPI", "RPE", "r5p_drain",
    "X5P_EC2", "F6P_EC2", "S7P_EC2", "F6P_EC3", "S7P_EC3",
]

# For reverse fluxes, only reversible reactions
REACTION_ORDER_REV = [r for r in REACTION_ORDER_NET if r in REVERSIBLE_RXNS]

# Thermodynamic conditions (aggregated model thermo files)
CONDITIONS = {
    "FFE": "glycolysis_thermodynamics.toml",
    "NE": "glycolysis_thermodynamics_ne.toml",
}


# ============================================================================
# Helper Functions
# ============================================================================


def net_to_forward_backward(v_net, deltaG_prime, temperature=298.15):
    """Convert net flux to forward and backward fluxes using Gibbs energy."""
    exp_term = np.exp(-deltaG_prime / (R * temperature))

    if exp_term > 1e6:
        return abs(v_net), 0.0
    if abs(exp_term - 1) < 1e-6:
        v_f = abs(v_net) * 1000
        v_b = v_f - v_net
        return v_f, v_b

    v_f = v_net * exp_term / (exp_term - 1)
    v_b = v_net / (exp_term - 1)
    return v_f, v_b


def compute_true_fluxes(condition):
    """Compute true forward and reverse fluxes for a condition.

    Returns:
        dict of {rxn_id: {"forward": v_f, "reverse": v_b}} for reversible rxns.
    """
    thermo_file = CONDITIONS[condition]
    thermo_data = toml.load(str(STD_MODEL_DIR / thermo_file))
    temperature = thermo_data.get("conditions", {}).get("temperature", 298.15)

    true_fluxes = {}
    for rxn_id in REVERSIBLE_RXNS:
        v_net = TRUE_NET_FLUXES[rxn_id]
        dG = thermo_data["reactions"][rxn_id]["deltaG_prime"]
        v_f, v_b = net_to_forward_backward(v_net, dG, temperature)
        true_fluxes[rxn_id] = {"forward": v_f, "reverse": v_b}

    return true_fluxes


def plot_heatmap(ax, pct_diff, alphas, reaction_order, title):
    """Plot a single heatmap panel with symmetric log scale."""
    vmax = np.nanmax(np.abs(pct_diff))
    if vmax < 0.1:
        vmax = 0.1

    # SymLogNorm: linear near zero (within linthresh), log outside
    norm = mcolors.SymLogNorm(linthresh=1.0, linscale=0.5, vmin=-vmax, vmax=vmax)

    im = ax.imshow(
        pct_diff,
        aspect="auto",
        cmap=cmc.vik,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45, ha="right")
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Annotate cells with compact labels
    for i in range(pct_diff.shape[0]):
        for j in range(pct_diff.shape[1]):
            val = pct_diff[i, j]
            if not np.isfinite(val):
                continue
            # Format: use scientific notation for large values
            aval = abs(val)
            if aval >= 1000:
                label = f"{val:.0e}"
            elif aval >= 10:
                label = f"{val:.0f}"
            elif aval >= 1:
                label = f"{val:.1f}"
            else:
                label = f"{val:.2f}"
            # Text color: white on saturated cells
            norm_val = norm(val)
            color = "white" if norm_val < 0.2 or norm_val > 0.8 else "black"
            ax.text(
                j, i, label,
                ha="center", va="center", fontsize=5, color=color,
            )

    return im


# ============================================================================
# Main
# ============================================================================


def main():
    """Generate forward and reverse flux error heatmaps."""
    df = pd.read_csv(RESULTS_DIR / "fitted_total_fluxes.csv")

    # Compute true forward/reverse fluxes per condition
    true_fwd = {cond: compute_true_fluxes(cond) for cond in CONDITIONS}

    # --- Forward flux error heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, cond in zip(axes, ["FFE", "NE"]):
        cond_data = df[df["condition"] == cond].sort_values("alpha")
        alphas = cond_data["alpha"].values

        pct_diff = np.full((len(REACTION_ORDER_NET), len(alphas)), np.nan)
        for j, (_, row) in enumerate(cond_data.iterrows()):
            for i, rxn in enumerate(REACTION_ORDER_NET):
                if rxn not in row or rxn not in TRUE_NET_FLUXES:
                    continue
                # Forward = net + reverse for reversible, net for irreversible
                if rxn in REVERSIBLE_RXNS and f"{rxn}_r" in row:
                    fitted_fwd = row[rxn] + row[f"{rxn}_r"]
                    true_val = true_fwd[cond][rxn]["forward"]
                else:
                    fitted_fwd = row[rxn]
                    true_val = TRUE_NET_FLUXES[rxn]
                if true_val != 0:
                    pct_diff[i, j] = (fitted_fwd - true_val) / true_val * 100

        im = plot_heatmap(ax, pct_diff, alphas, REACTION_ORDER_NET, cond)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Forward flux error (%)", fontsize=10)

    axes[0].set_yticks(range(len(REACTION_ORDER_NET)))
    axes[0].set_yticklabels(REACTION_ORDER_NET, fontsize=9)
    axes[0].set_ylabel("Reaction", fontsize=12)

    fig.suptitle("Forward Flux Error", fontsize=16, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = RESULTS_DIR / f"flux_error_forward.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()

    # --- Reverse flux error heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, cond in zip(axes, ["FFE", "NE"]):
        cond_data = df[df["condition"] == cond].sort_values("alpha")
        alphas = cond_data["alpha"].values

        pct_diff = np.full((len(REACTION_ORDER_REV), len(alphas)), np.nan)
        for j, (_, row) in enumerate(cond_data.iterrows()):
            for i, rxn in enumerate(REACTION_ORDER_REV):
                col = f"{rxn}_r"
                if col not in row:
                    continue
                true_val = true_fwd[cond][rxn]["reverse"]
                if true_val != 0:
                    pct_diff[i, j] = (row[col] - true_val) / true_val * 100

        im = plot_heatmap(ax, pct_diff, alphas, REACTION_ORDER_REV, cond)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Reverse flux error (%)", fontsize=10)

    axes[0].set_yticks(range(len(REACTION_ORDER_REV)))
    axes[0].set_yticklabels(REACTION_ORDER_REV, fontsize=9)
    axes[0].set_ylabel("Reaction", fontsize=12)

    fig.suptitle("Reverse Flux Error", fontsize=16, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = RESULTS_DIR / f"flux_error_reverse.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()

    print(f"\nGenerated in: {RESULTS_DIR}")
    print("  - flux_error_forward.png/pdf")
    print("  - flux_error_reverse.png/pdf")


if __name__ == "__main__":
    main()

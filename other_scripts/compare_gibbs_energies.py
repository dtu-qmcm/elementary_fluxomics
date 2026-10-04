#!/usr/bin/env python3
"""Compare reaction Gibbs energies recovered by the lumped vs separated model.

Both fits estimate reaction Gibbs energies from their fitted forward/backward
fluxes via the inverse relation

    v_f / v_b = exp(-dG / (R*T))   =>   dG = -R*T * ln(v_f / v_b)

with v_f = net + reverse and v_b = reverse (the `_r` columns). For the FBA
reaction the two model structures differ:

  * lumped (aggregated):  one FBA step          -> dG estimated directly.
  * separated (mechanistic): FBA1 + FBA2 steps  -> overall dG = dG(FBA1) + dG(FBA2)
    (Gibbs energies add across the sequential decomposition).

The headline metric is the per-reaction error RATIO

    ratio = |true_dG - lumped_dG| / |true_dG - separated_dG|

  ratio > 1  -> the lumped model is worse (separating the step helps recovery)
  ratio < 1  -> the lumped model is better
  ratio ~ 1  -> equivalent

True dG: non-FBA reactions from the thermodynamics TOML; FBA from the additive
total used to generate the data (total_fba_dG, constant across alpha).

Reads:  results/other_scripts/fit_mech_vs_agg/fitted_total_fluxes.csv       (lumped)
        results/other_scripts/fit_mech_vs_agg/fitted_total_fluxes_mech.csv  (separated)
Writes (results/other_scripts/fit_mech_vs_agg/):
  - gibbs_error_ratio.csv             per (condition, alpha, reaction) dGs + ratio
  - gibbs_fba_comparison.png/pdf      FBA: true / lumped / separated dG vs alpha
  - gibbs_error_ratio_fba.png/pdf     FBA error ratio vs alpha (headline)
  - gibbs_error_ratio_heatmap.png/pdf log10(ratio) for all reversible reactions
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
RESULTS_DIR = PROJECT_DIR / "results" / "other_scripts" / "fit_mech_vs_agg"

R = 8.314e-3  # Gas constant in kJ/(mol*K)

LUMPED_FILE = RESULTS_DIR / "fitted_total_fluxes.csv"
SEPARATED_FILE = RESULTS_DIR / "fitted_total_fluxes_mech.csv"

# Reversible reactions in the aggregated model (pathway display order)
REVERSIBLE_RXNS = [
    "GLUT2", "HEX", "GPI", "PFK", "FBA", "TPI",
    "GAPD_PGK", "PGM", "PYK",
    "G6PDH2r_PGL", "RPI", "RPE",
    "X5P_EC2", "F6P_EC2", "S7P_EC2", "F6P_EC3", "S7P_EC3",
]

# Thermodynamic conditions: thermo file + true total (lumped) FBA dG.
CONDITIONS = {
    "FFE": {"thermo_file": "glycolysis_thermodynamics.toml", "total_fba_dG": -5.0},
    "NE": {"thermo_file": "glycolysis_thermodynamics_ne.toml", "total_fba_dG": -0.87},
}


# ============================================================================
# Helper Functions
# ============================================================================


def forward_reverse_to_deltaG(v_f, v_b, temperature):
    """Recover reaction Gibbs energy from forward/backward fluxes.

    dG = -R*T * ln(v_f / v_b). Returns NaN if either flux is non-positive
    (Gibbs energy undefined / unidentifiable).
    """
    v_f = np.asarray(v_f, dtype=float)
    v_b = np.asarray(v_b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dG = -R * temperature * np.log(v_f / v_b)
    return float(np.where((v_f > 0) & (v_b > 0), dG, np.nan))


def est_dG_from_row(row, rxn, temperature):
    """Estimated dG for a single reversible reaction from a fitted-flux row."""
    net = row.get(rxn, np.nan)
    rev = row.get(f"{rxn}_r", np.nan)
    if pd.isna(net) or pd.isna(rev):
        return np.nan
    return forward_reverse_to_deltaG(net + rev, rev, temperature)


def load_true_deltaG(condition):
    """True dG per reversible reaction (kJ/mol) and temperature for a condition."""
    cfg = CONDITIONS[condition]
    thermo_data = toml.load(str(STD_MODEL_DIR / cfg["thermo_file"]))
    temperature = thermo_data.get("conditions", {}).get("temperature", 298.15)
    true_dG = {}
    for rxn in REVERSIBLE_RXNS:
        if rxn == "FBA":
            true_dG[rxn] = cfg["total_fba_dG"]
        else:
            true_dG[rxn] = thermo_data["reactions"][rxn]["deltaG_prime"]
    return true_dG, temperature


# ============================================================================
# Analysis
# ============================================================================


def _sim_modes(df):
    """Simulation modes present in a flux DataFrame (default ss)."""
    if "sim_mode" in df.columns:
        return [m for m in ["ss", "inst"] if (df["sim_mode"] == m).any()]
    return ["ss"]


def build_comparison(df_lumped, df_sep):
    """Merge lumped and separated fits into a per-reaction dG error table.

    Iterates over (sim_mode, condition); for each, joins lumped & separated
    rows on alpha and computes the lumped-vs-separated dG error ratio.
    """
    if "sim_mode" not in df_lumped.columns:
        df_lumped = df_lumped.assign(sim_mode="ss")
    if "sim_mode" not in df_sep.columns:
        df_sep = df_sep.assign(sim_mode="ss")

    rows = []
    for sim_mode in _sim_modes(df_lumped):
        for cond in CONDITIONS:
            true_dG, temperature = load_true_deltaG(cond)
            lump = df_lumped[
                (df_lumped["condition"] == cond)
                & (df_lumped["sim_mode"] == sim_mode)
            ].copy()
            sep = df_sep[
                (df_sep["condition"] == cond) & (df_sep["sim_mode"] == sim_mode)
            ].copy()
            lump["akey"] = lump["alpha"].round(6)
            sep["akey"] = sep["alpha"].round(6)
            sep_by_alpha = {r["akey"]: r for _, r in sep.iterrows()}

            for _, lrow in lump.sort_values("alpha").iterrows():
                srow = sep_by_alpha.get(lrow["akey"])
                for rxn in REVERSIBLE_RXNS:
                    lumped_dG = est_dG_from_row(lrow, rxn, temperature)

                    if srow is None:
                        sep_dG = np.nan
                    elif rxn == "FBA":
                        # Overall FBA dG = dG(FBA1) + dG(FBA2) (additive steps).
                        d1 = est_dG_from_row(srow, "FBA1", temperature)
                        d2 = est_dG_from_row(srow, "FBA2", temperature)
                        sep_dG = d1 + d2
                    else:
                        sep_dG = est_dG_from_row(srow, rxn, temperature)

                    err_lumped = abs(true_dG[rxn] - lumped_dG)
                    err_sep = abs(true_dG[rxn] - sep_dG)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = err_lumped / err_sep
                    rows.append(
                        {
                            "sim_mode": sim_mode,
                            "condition": cond,
                            "alpha": lrow["alpha"],
                            "reaction": rxn,
                            "true_dG": true_dG[rxn],
                            "lumped_dG": lumped_dG,
                            "sep_dG": sep_dG,
                            "abs_err_lumped": err_lumped,
                            "abs_err_sep": err_sep,
                            "ratio": ratio,
                        }
                    )
    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================


def plot_fba_comparison(comp, output_dir):
    """FBA Gibbs energy: true vs lumped vs separated, gridded sim_mode x condition."""
    sim_modes = _sim_modes(comp)
    conds = ["FFE", "NE"]
    c_lump = cmc.batlow(0.25)
    c_sep = cmc.batlow(0.7)

    fig, axes = plt.subplots(
        len(sim_modes), len(conds),
        figsize=(6.5 * len(conds), 4.5 * len(sim_modes)),
        squeeze=False, sharex=True,
    )

    for ri, sim_mode in enumerate(sim_modes):
        for ci, cond in enumerate(conds):
            ax = axes[ri][ci]
            sub = comp[
                (comp["sim_mode"] == sim_mode)
                & (comp["condition"] == cond)
                & (comp["reaction"] == "FBA")
            ].sort_values("alpha")
            if not len(sub):
                ax.axis("off")
                continue
            true_val = sub["true_dG"].iloc[0]
            ax.axhline(
                true_val, color="black", linestyle="--", linewidth=1.5,
                label=f"true ({true_val:g})",
            )
            ax.plot(
                sub["alpha"], sub["lumped_dG"], "o-", color=c_lump,
                markersize=5, label="lumped",
            )
            ax.plot(
                sub["alpha"], sub["sep_dG"], "s-", color=c_sep,
                markersize=5, label="separated (FBA1+FBA2)",
            )
            ax.set_title(f"{sim_mode.upper()} / {cond}  (FBA $\\Delta G$)", fontsize=12)
            ax.set_xlabel(r"$\alpha$", fontsize=11)
            ax.set_ylabel(r"$\Delta G$ [kJ/mol]", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("FBA Gibbs Energy: True vs Lumped vs Separated", fontsize=15)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = output_dir / f"gibbs_fba_comparison.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
        sys.stdout.flush()
    plt.close()


def plot_ratio_fba(comp, output_dir):
    """Headline: FBA error ratio (|true-lumped|/|true-sep|) vs alpha.

    Overlays simulation modes (line style) and conditions (color).
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"FFE": cmc.batlow(0.25), "NE": cmc.batlow(0.7)}
    mode_style = {"ss": {"ls": "-", "marker": "o"}, "inst": {"ls": "--", "marker": "s"}}

    for sim_mode in _sim_modes(comp):
        for cond in ["FFE", "NE"]:
            sub = comp[
                (comp["sim_mode"] == sim_mode)
                & (comp["condition"] == cond)
                & (comp["reaction"] == "FBA")
            ].sort_values("alpha")
            if not len(sub):
                continue
            r = sub["ratio"].replace([np.inf, -np.inf], np.nan)
            st = mode_style.get(sim_mode, {"ls": "-", "marker": "o"})
            ax.plot(
                sub["alpha"], r, linestyle=st["ls"], marker=st["marker"],
                color=colors[cond], linewidth=2, markersize=6,
                label=f"{sim_mode.upper()} / {cond}",
            )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2,
               label="ratio = 1 (equivalent)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$ (fraction of total $\Delta G$ on FBA1)", fontsize=12)
    ax.set_ylabel(
        r"$\frac{|\Delta G_{true} - \Delta G_{lumped}|}"
        r"{|\Delta G_{true} - \Delta G_{sep}|}$",
        fontsize=14,
    )
    ax.set_title("FBA Gibbs-Energy Error Ratio (lumped / separated)", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = output_dir / f"gibbs_error_ratio_fba.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
        sys.stdout.flush()
    plt.close()


def plot_ratio_heatmap(comp, output_dir, sim_mode="ss"):
    """log10(error ratio) for every reversible reaction x alpha, FFE/NE.

    One figure per simulation mode; filename carries the mode suffix.
    """
    comp = comp[comp["sim_mode"] == sim_mode] if "sim_mode" in comp.columns else comp

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    logr_all = np.log10(
        comp["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    )
    logr_all = logr_all[np.isfinite(logr_all)]
    vmax = float(np.nanpercentile(np.abs(logr_all), 98)) if len(logr_all) else 1.0
    if vmax < 0.3:
        vmax = 0.3

    for ax, cond in zip(axes, ["FFE", "NE"]):
        cond_data = comp[comp["condition"] == cond]
        alphas = np.sort(cond_data["alpha"].unique())

        mat = np.full((len(REVERSIBLE_RXNS), len(alphas)), np.nan)
        for i, rxn in enumerate(REVERSIBLE_RXNS):
            for j, a in enumerate(alphas):
                sel = cond_data[
                    (cond_data["reaction"] == rxn) & (cond_data["alpha"] == a)
                ]
                if len(sel):
                    ratio = sel["ratio"].values[0]
                    if np.isfinite(ratio) and ratio > 0:
                        mat[i, j] = np.log10(ratio)
                    elif np.isinf(ratio) or (np.isfinite(ratio) and ratio > 1e9):
                        mat[i, j] = vmax  # lumped vastly worse

        im = ax.imshow(
            mat, aspect="auto", cmap=cmc.vik, vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45, ha="right")
        ax.set_xlabel(r"$\alpha$", fontsize=12)
        ax.set_title(cond, fontsize=14)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                color = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=5, color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(
            r"$\log_{10}\,(|true-lumped| / |true-sep|)$", fontsize=10
        )

    axes[0].set_yticks(range(len(REVERSIBLE_RXNS)))
    axes[0].set_yticklabels(REVERSIBLE_RXNS, fontsize=9)
    axes[0].set_ylabel("Reaction", fontsize=12)

    fig.suptitle(
        f"Gibbs-Energy Error Ratio [{sim_mode.upper()}]  "
        "(red = lumped worse, blue = separated worse)",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = output_dir / f"gibbs_error_ratio_heatmap_{sim_mode}.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
        sys.stdout.flush()
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main():
    """Build the lumped-vs-separated dG error comparison and plot it."""
    print("=" * 80)
    print("Compare Gibbs-Energy Recovery: Lumped vs Separated")
    print("=" * 80)
    sys.stdout.flush()

    if not SEPARATED_FILE.exists():
        sys.exit(
            f"Missing {SEPARATED_FILE}.\n"
            "Run fit_mechanistic_vs_aggregated.py first (it now fits both models)."
        )

    df_lumped = pd.read_csv(LUMPED_FILE)
    df_sep = pd.read_csv(SEPARATED_FILE)
    print(f"Loaded: {LUMPED_FILE.name} ({len(df_lumped)} rows)")
    print(f"Loaded: {SEPARATED_FILE.name} ({len(df_sep)} rows)")

    comp = build_comparison(df_lumped, df_sep)
    csv_path = RESULTS_DIR / "gibbs_error_ratio.csv"
    comp.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    sys.stdout.flush()

    sim_modes = _sim_modes(comp)

    print("Creating FBA dG comparison plot...")
    plot_fba_comparison(comp, RESULTS_DIR)
    print("Creating FBA error-ratio plot...")
    plot_ratio_fba(comp, RESULTS_DIR)
    for sim_mode in sim_modes:
        print(f"Creating error-ratio heatmap ({sim_mode})...")
        plot_ratio_heatmap(comp, RESULTS_DIR, sim_mode=sim_mode)

    # Summary
    print(f"\n{'=' * 80}\nSummary (ratio = |true-lumped| / |true-sep|)\n{'=' * 80}")
    for sim_mode in sim_modes:
        for cond in CONDITIONS:
            sub = comp[(comp["sim_mode"] == sim_mode) & (comp["condition"] == cond)]
            fba = sub[sub["reaction"] == "FBA"].copy()
            fba_r = fba["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            print(f"\n[{sim_mode.upper()}] {cond}:")
            if len(fba_r):
                print(
                    f"  FBA ratio: median={fba_r.median():.2f}, "
                    f"mean={fba_r.mean():.2f}, "
                    f"range={fba_r.min():.2f}..{fba_r.max():.2f}"
                )
                n_lumped_worse = (fba["ratio"] > 1).sum()
                print(
                    f"  FBA: lumped worse than separated in "
                    f"{n_lumped_worse}/{len(fba)} alpha points"
                )
            allr = sub["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(allr):
                frac = (allr > 1).mean() * 100
                print(
                    f"  all reversible reactions: lumped worse in {frac:.0f}% of "
                    f"cells (median ratio {allr.median():.2f})"
                )
    sys.stdout.flush()

    print(f"\nGenerated files in: {RESULTS_DIR}")
    print("  - gibbs_error_ratio.csv (with sim_mode column)")
    print("  - gibbs_fba_comparison.png/pdf")
    print("  - gibbs_error_ratio_fba.png/pdf")
    print("  - gibbs_error_ratio_heatmap_ss.png/pdf, gibbs_error_ratio_heatmap_inst.*")


if __name__ == "__main__":
    main()

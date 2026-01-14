#!/usr/bin/env python3
"""1D Chi-squared Comparison: FBA2 Fixed, FBA1 Variable.

This script compares mechanistic model configurations where FBA2 is held constant
at -2.0 kJ/mol while FBA1 varies from -1.0 to -5.0 kJ/mol.

Reference (fixed): FBA1=-3.0, FBA2=-2.0 kJ/mol
Variable: FBA1 from -1.0 to -5.0 kJ/mol (FBA2 constant at -2.0)

Outputs:
- Aggregated SSR vs FBA1 parameter plot
- Timecourse chi-squared (aggregated and per-metabolite)
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from freeflux import Model

# ============================================================================
# Configuration
# ============================================================================

# Paths - relative to scripts folder
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_mech_gly3p"
RESULTS_DIR = PROJECT_DIR / "results" / "gibbs_comparison_fba2_fixed"

# Gas constant in kJ/(mol*K)
R = 8.314e-3

# Timepoints: 100 points logarithmically spaced from 1e-3s to 1e1s, plus final at 1e3s
TIMEPOINTS = list(np.logspace(-3, 1, 100)) + [1e3]
FINAL_TIMEPOINT = 1e3  # Excluded from timecourse SSR

# Target EMUs to compare
TARGET_EMUS = {
    "g6p": ["123456"],
    "f6p": ["123456"],
    "dhap": ["123"],
    "g3p": ["123"],
}

# Mapping from EMU ID to metabolite name for output
EMU_TO_METABOLITE = {
    "g6p_123456": "g6p",
    "f6p_123456": "f6p",
    "dhap_123": "dhap",
    "g3p_123": "g3p",
}

# Log-normal error scale (standard deviation of log-transformed values)
LOGNORMAL_SCALE = 0.01

# Small epsilon to avoid log(0)
EPSILON = 1e-10

# Reference and fixed values
REFERENCE_FBA1 = -3.0
REFERENCE_FBA2 = -2.0
FIXED_FBA2 = -2.0  # FBA2 stays constant


# ============================================================================
# Helper Functions
# ============================================================================


def create_output_directory():
    """Create output directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def net_to_forward_backward(v_net, deltaG_prime, temperature):
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


def run_simulation(thermo_overrides=None):
    """Run forward simulation and return MDVs."""
    model = Model("glycolysis_mech_gly3p")
    model.read_from_file(str(MODEL_DIR / "reactions.tsv"))

    flux_data = toml.load(str(MODEL_DIR / "glycolysis_fluxes.toml"))
    thermo_data = toml.load(str(MODEL_DIR / "glycolysis_thermodynamics.toml"))
    temperature = thermo_data.get("conditions", {}).get("temperature", 298.15)

    if thermo_overrides:
        for rxn_id, dG in thermo_overrides.items():
            if "reactions" not in thermo_data:
                thermo_data["reactions"] = {}
            if rxn_id not in thermo_data["reactions"]:
                thermo_data["reactions"][rxn_id] = {}
            thermo_data["reactions"][rxn_id]["deltaG_prime"] = dG

    with model.simulator("inst") as isim:
        isim.set_target_EMUs(TARGET_EMUS)
        isim.set_timepoints(TIMEPOINTS)
        isim.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=["110001"],
            percentage=[1.0],
            purity=[1.0],
        )

        reversible_rxns = {
            fid.rsplit("_", 1)[0] for fid in model.totalfluxids if fid.endswith("_f")
        }
        conversion_factor = 1000 / 3600

        for section in ["external_fluxes", "internal_fluxes"]:
            if section in flux_data:
                for rxn_id, value in flux_data[section].items():
                    if rxn_id == "glc_uptake":
                        rxn_id = "glc_input"
                    elif rxn_id == "pyr_secretion":
                        continue

                    v_net = value * conversion_factor
                    deltaG = (
                        thermo_data.get("reactions", {})
                        .get(rxn_id, {})
                        .get("deltaG_prime", -5.0)
                    )

                    if rxn_id in reversible_rxns:
                        v_f, v_b = net_to_forward_backward(v_net, deltaG, temperature)
                        isim.set_flux(f"{rxn_id}_f", v_f)
                        isim.set_flux(f"{rxn_id}_b", v_b)
                    else:
                        isim.set_flux(rxn_id, v_net)

        for metab in model.metabolites:
            isim.set_concentration(metab, 1.0)

        isim.prepare(n_jobs=1)
        res = isim.simulate()

        mdvs = {}
        for emu_id in res.simulated_EMUs:
            mdv_result = res.simulated_MDV(emu_id)
            mdvs[emu_id] = {t: np.array(mdv_result[t]) for t in res.timepoints}

        return mdvs


def calculate_chisquared_at_timepoint(mdvs_ref, mdvs_test, timepoint, scale):
    """Calculate chi-squared at a single timepoint with per-metabolite breakdown.

    Uses log-normal distribution: sum((log(test) - log(ref))^2 / scale^2)
    """
    total_chi_squared = 0.0
    per_metabolite = {}
    variance = scale**2

    for emu_id in mdvs_ref:
        if emu_id not in mdvs_test:
            continue
        if timepoint not in mdvs_ref[emu_id] or timepoint not in mdvs_test[emu_id]:
            continue

        mdv_ref = mdvs_ref[emu_id][timepoint]
        mdv_test = mdvs_test[emu_id][timepoint]

        mdv_ref = np.maximum(mdv_ref, 0.0)
        mdv_test = np.maximum(mdv_test, 0.0)
        if np.sum(mdv_ref) > 0:
            mdv_ref = mdv_ref / np.sum(mdv_ref)
        if np.sum(mdv_test) > 0:
            mdv_test = mdv_test / np.sum(mdv_test)

        # Log-normal error
        log_ref = np.log(mdv_ref + EPSILON)
        log_test = np.log(mdv_test + EPSILON)
        chi_sq = float(np.sum((log_test - log_ref) ** 2 / variance))
        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])
        per_metabolite[met_name] = chi_sq
        total_chi_squared += chi_sq

    return total_chi_squared, per_metabolite


def compute_normalized_ssr(mdvs_ref, mdvs_test, scale, exclude_timepoint=None):
    """Compute normalized SSR between two sets of MDVs using log-normal error.

    Uses log-normal distribution: sum((log(test) - log(ref))^2 / scale^2)
    """
    ssr = 0.0
    variance = scale**2

    for emu_id in mdvs_ref:
        if emu_id not in mdvs_test:
            continue
        for t in mdvs_ref[emu_id]:
            if t not in mdvs_test[emu_id]:
                continue
            # Skip excluded timepoint
            if exclude_timepoint is not None and abs(t - exclude_timepoint) < 1e-6:
                continue
            mdv_ref = mdvs_ref[emu_id][t]
            mdv_test = mdvs_test[emu_id][t]

            mdv_ref = np.maximum(mdv_ref, 0.0)
            mdv_test = np.maximum(mdv_test, 0.0)
            if np.sum(mdv_ref) > 0:
                mdv_ref = mdv_ref / np.sum(mdv_ref)
            if np.sum(mdv_test) > 0:
                mdv_test = mdv_test / np.sum(mdv_test)

            # Log-normal error
            log_ref = np.log(mdv_ref + EPSILON)
            log_test = np.log(mdv_test + EPSILON)
            ssr += np.sum((log_test - log_ref) ** 2 / variance)

    return ssr


def run_1d_scan(dG_values):
    """Run 1D scan over FBA1 Gibbs energy with FBA2 fixed.

    Returns both aggregated SSR results and timecourse data.
    """
    print(f"Running reference model (FBA1={REFERENCE_FBA1}, FBA2={REFERENCE_FBA2})...")
    sys.stdout.flush()

    ref_overrides = {"FBA1": REFERENCE_FBA1, "FBA2": REFERENCE_FBA2}
    mdvs_ref = run_simulation(ref_overrides)

    # Filter timepoints for timecourse analysis
    filtered_timepoints = [t for t in TIMEPOINTS if t > 0 and abs(t - FINAL_TIMEPOINT) > 1e-6]

    aggregated_results = []
    timecourse_results = []
    total = len(dG_values)

    for i, dG in enumerate(dG_values):
        print(f"  [{i+1}/{total}] FBA1={dG:.2f} kJ/mol, FBA2={FIXED_FBA2:.2f} (fixed)")
        sys.stdout.flush()

        thermo_overrides = {"FBA1": dG, "FBA2": FIXED_FBA2}

        try:
            mdvs_test = run_simulation(thermo_overrides)
            ssr = compute_normalized_ssr(
                mdvs_ref, mdvs_test, LOGNORMAL_SCALE, exclude_timepoint=FINAL_TIMEPOINT
            )

            # Calculate timecourse chi-squared
            for t in filtered_timepoints:
                total_chi_sq, per_met = calculate_chisquared_at_timepoint(
                    mdvs_ref, mdvs_test, t, LOGNORMAL_SCALE
                )
                row = {
                    "FBA1_deltaG": dG,
                    "FBA2_deltaG": FIXED_FBA2,
                    "timepoint": t,
                    "chi_squared_total": total_chi_sq,
                }
                for met_id, chi_sq in per_met.items():
                    row[f"chi_squared_{met_id}"] = chi_sq
                timecourse_results.append(row)

        except Exception as e:
            print(f"    Error: {e}")
            sys.stdout.flush()
            ssr = np.nan

        aggregated_results.append(
            {
                "FBA1_deltaG": dG,
                "FBA2_deltaG": FIXED_FBA2,
                "SSR": ssr,
            }
        )

    return pd.DataFrame(aggregated_results), pd.DataFrame(timecourse_results)


def plot_comparison(df, output_path):
    """Plot 1D SSR comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["FBA1_deltaG"], df["SSR"], "o-", linewidth=2, markersize=6, color="steelblue"
    )

    # Mark reference FBA1 value
    ax.axvline(
        x=REFERENCE_FBA1,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Reference FBA1={REFERENCE_FBA1}",
    )

    ax.set_xlabel(f"FBA1 dG (kJ/mol) [FBA2 fixed at {FIXED_FBA2}]", fontsize=12)
    ax.set_ylabel("Normalized SSR", fontsize=12)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_timecourse_aggregated(df, output_path):
    """Plot aggregated chi-squared timecourse for all test conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique FBA1 values
    fba1_values = sorted(df["FBA1_deltaG"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(fba1_values)))

    for idx, fba1 in enumerate(fba1_values):
        condition_data = df[df["FBA1_deltaG"] == fba1].sort_values("timepoint")
        label = f"FBA1={fba1:.2f}"
        if abs(fba1 - REFERENCE_FBA1) < 0.1:
            label += " (ref)"
        ax.plot(
            condition_data["timepoint"],
            condition_data["chi_squared_total"],
            linewidth=2,
            color=colors[idx],
            label=label,
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Chi-Squared (total)", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_timecourse_per_metabolite(df, output_path):
    """Plot per-metabolite chi-squared timecourse in subplots."""
    fba1_values = sorted(df["FBA1_deltaG"].unique())
    n_conditions = len(fba1_values)

    # Determine grid size
    ncols = 4
    nrows = (n_conditions + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten() if n_conditions > 1 else [axes]

    metabolite_colors = {
        "g6p": "C0",
        "f6p": "C1",
        "dhap": "C2",
        "g3p": "C3",
    }

    for idx, fba1 in enumerate(fba1_values):
        ax = axes[idx]
        condition_data = df[df["FBA1_deltaG"] == fba1].sort_values("timepoint")

        for met_id, color in metabolite_colors.items():
            col_name = f"chi_squared_{met_id}"
            if col_name in condition_data.columns:
                ax.plot(
                    condition_data["timepoint"],
                    condition_data[col_name],
                    linewidth=1.5,
                    color=color,
                    label=met_id,
                )

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Chi-Squared", fontsize=10)
        title = f"FBA1={fba1:.2f} kJ/mol"
        if abs(fba1 - REFERENCE_FBA1) < 0.1:
            title += " (ref)"
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Hide unused subplots
    for idx in range(n_conditions, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def main():
    """Run 1D comparison analysis with FBA2 fixed."""
    print("=" * 80)
    print("1D Normalized SSR Comparison: FBA2 Fixed, FBA1 Variable")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()

    dG_values = np.linspace(-5, -1, 20)

    print(f"\nConfiguration:")
    print(f"  Reference: FBA1={REFERENCE_FBA1}, FBA2={REFERENCE_FBA2} kJ/mol")
    print(f"  Fixed: FBA2={FIXED_FBA2} kJ/mol")
    print(f"  Variable: FBA1 from {dG_values.min():.1f} to {dG_values.max():.1f} kJ/mol")
    print(f"  Number of points: {len(dG_values)}")
    sys.stdout.flush()

    # Run scan
    print("\n" + "=" * 80)
    print("Running 1D Scan")
    print("=" * 80)
    sys.stdout.flush()

    df_agg, df_timecourse = run_1d_scan(dG_values)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    sys.stdout.flush()

    csv_path = output_dir / "gibbs_comparison_fba2_fixed.csv"
    df_agg.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    timecourse_csv_path = output_dir / "timecourse_chisquared.csv"
    df_timecourse.to_csv(timecourse_csv_path, index=False)
    print(f"  Saved CSV: {timecourse_csv_path}")
    sys.stdout.flush()

    # Plot aggregated SSR
    print("\n  Creating comparison plot...")
    sys.stdout.flush()
    plot_comparison(df_agg, output_dir / "gibbs_comparison_fba2_fixed.png")

    # Plot timecourse
    print("  Creating timecourse plots...")
    sys.stdout.flush()
    plot_timecourse_aggregated(df_timecourse, output_dir / "timecourse_aggregated.png")
    plot_timecourse_per_metabolite(df_timecourse, output_dir / "timecourse_per_metabolite.png")

    # Summary
    min_idx = df_agg["SSR"].idxmin()
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(
        f"  Min SSR: {df_agg['SSR'].min():.6f} at FBA1={df_agg.loc[min_idx, 'FBA1_deltaG']:.2f} kJ/mol"
    )
    print(f"  Max SSR: {df_agg['SSR'].max():.6f}")

    # Find SSR at reference FBA1
    closest_to_ref = df_agg.iloc[
        (df_agg["FBA1_deltaG"] - REFERENCE_FBA1).abs().argsort()[:1]
    ]
    print(f"  SSR at reference FBA1={REFERENCE_FBA1}: {closest_to_ref['SSR'].values[0]:.6f}")

    print(f"\nGenerated files in: {output_dir}")
    print("  - gibbs_comparison_fba2_fixed.csv")
    print("  - gibbs_comparison_fba2_fixed.png")
    print("  - timecourse_chisquared.csv")
    print("  - timecourse_aggregated.png")
    print("  - timecourse_per_metabolite.png")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

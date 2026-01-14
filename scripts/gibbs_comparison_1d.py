#!/usr/bin/env python3
"""1D Chi-squared Comparison: Mechanistic vs Non-Mechanistic Model.

This script compares the mechanistic glycolysis model (FBA1/FBA2) against
the non-mechanistic model (single FBA) as the FBA dG varies.

Reference (fixed): Mechanistic model with FBA1=-3, FBA2=-2 kJ/mol
Variable: Non-mechanistic model with FBA dG from -9 to -1 kJ/mol
"""

from pathlib import Path
import sys

import cmcrameri.cm as cmc
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
MECH_MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_mech_gly3p"
STD_MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_gly3p"
RESULTS_DIR = PROJECT_DIR / "results" / "gibbs_comparison_1d"

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

# Log-normal error scale (standard deviation of log-transformed values)
LOGNORMAL_SCALE = 0.01

# Small epsilon to avoid log(0)
EPSILON = 1e-10


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


def run_simulation(model_dir, model_name, thermo_overrides=None):
    """Run forward simulation and return MDVs."""
    model = Model(model_name)
    model.read_from_file(str(model_dir / "reactions.tsv"))

    flux_data = toml.load(str(model_dir / "glycolysis_fluxes.toml"))
    thermo_data = toml.load(str(model_dir / "glycolysis_thermodynamics.toml"))
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

            # Log-normal error: use log of values with epsilon to avoid log(0)
            log_ref = np.log(mdv_ref + EPSILON)
            log_test = np.log(mdv_test + EPSILON)
            ssr += np.sum((log_test - log_ref) ** 2 / variance)

    return ssr


def compute_ssr_at_timepoint(mdvs_ref, mdvs_test, scale, target_timepoint):
    """Compute normalized SSR at a specific timepoint using log-normal error."""
    ssr = 0.0
    variance = scale**2

    for emu_id in mdvs_ref:
        if emu_id not in mdvs_test:
            continue
        for t in mdvs_ref[emu_id]:
            if t not in mdvs_test[emu_id]:
                continue
            # Only include target timepoint
            if abs(t - target_timepoint) > 1e-6:
                continue
            mdv_ref = mdvs_ref[emu_id][t]
            mdv_test = mdvs_test[emu_id][t]

            mdv_ref = np.maximum(mdv_ref, 0.0)
            mdv_test = np.maximum(mdv_test, 0.0)
            if np.sum(mdv_ref) > 0:
                mdv_ref = mdv_ref / np.sum(mdv_ref)
            if np.sum(mdv_test) > 0:
                mdv_test = mdv_test / np.sum(mdv_test)

            # Log-normal error: use log of values with epsilon to avoid log(0)
            log_ref = np.log(mdv_ref + EPSILON)
            log_test = np.log(mdv_test + EPSILON)
            ssr += np.sum((log_test - log_ref) ** 2 / variance)

    return ssr


def run_1d_scan(dG_values):
    """Run 1D scan over FBA Gibbs energy of non-mechanistic model."""
    print("Running mechanistic model (FBA1=-3, FBA2=-2)...")
    sys.stdout.flush()

    mech_overrides = {"FBA1": -3.0, "FBA2": -2.0}
    mdvs_mech = run_simulation(MECH_MODEL_DIR, "glycolysis_mech_gly3p", mech_overrides)

    results = []
    total = len(dG_values)

    for i, dG in enumerate(dG_values):
        print(f"  [{i+1}/{total}] Non-mechanistic FBA dG = {dG:.2f} kJ/mol")
        sys.stdout.flush()

        thermo_overrides = {"FBA": dG}

        try:
            mdvs_std = run_simulation(STD_MODEL_DIR, "glycolysis_gly3p", thermo_overrides)
            ssr_timecourse = compute_normalized_ssr(
                mdvs_mech, mdvs_std, LOGNORMAL_SCALE, exclude_timepoint=FINAL_TIMEPOINT
            )
            ssr_final = compute_ssr_at_timepoint(
                mdvs_mech, mdvs_std, LOGNORMAL_SCALE, FINAL_TIMEPOINT
            )
        except Exception as e:
            print(f"    Error: {e}")
            sys.stdout.flush()
            ssr_timecourse = np.nan
            ssr_final = np.nan

        results.append(
            {
                "FBA_deltaG": dG,
                "SSR_timecourse": ssr_timecourse,
                "SSR_final": ssr_final,
            }
        )

    return pd.DataFrame(results)


def plot_comparison(df, output_path):
    """Plot 1D SSR comparison for timecourse data using batlow colormap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use batlow colormap
    color = cmc.batlow(0.3)

    ax.plot(
        df["FBA_deltaG"],
        df["SSR_timecourse"],
        "o-",
        linewidth=2,
        markersize=6,
        color=color,
        label="Timecourse SSR (1e-3s to 10s)",
    )

    # Mark where FBA dG = -5 (sum of FBA1 + FBA2 = -3 + -2)
    ax.axvline(
        x=-5,
        color=cmc.batlow(0.85),
        linestyle="--",
        linewidth=1.5,
        label="dG(FBA) = dG(FBA1) + dG(FBA2) = -5",
    )

    ax.set_xlabel("Non-mechanistic FBA dG (kJ/mol)", fontsize=12)
    ax.set_ylabel("Normalized SSR (Timecourse)", fontsize=12)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_final_timepoint(df, output_path):
    """Plot 1D SSR comparison for final timepoint (steady-state) using batlow colormap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use batlow colormap - different color for final
    color = cmc.batlow(0.6)

    ax.plot(
        df["FBA_deltaG"],
        df["SSR_final"],
        "s-",
        linewidth=2,
        markersize=6,
        color=color,
        label=f"Final timepoint SSR (t={FINAL_TIMEPOINT:.0f}s)",
    )

    # Mark where FBA dG = -5 (sum of FBA1 + FBA2 = -3 + -2)
    ax.axvline(
        x=-5,
        color=cmc.batlow(0.85),
        linestyle="--",
        linewidth=1.5,
        label="dG(FBA) = dG(FBA1) + dG(FBA2) = -5",
    )

    ax.set_xlabel("Non-mechanistic FBA dG (kJ/mol)", fontsize=12)
    ax.set_ylabel("Normalized SSR (Final Timepoint)", fontsize=12)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def main():
    """Run 1D comparison analysis."""
    print("=" * 80)
    print("1D Normalized SSR Comparison")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()

    dG_values = np.linspace(-9, -1, 20)

    print(f"\nConfiguration:")
    print(f"  Mechanistic model: FBA1=-3, FBA2=-2 kJ/mol (fixed)")
    print(f"  Non-mechanistic FBA dG range: {dG_values.min():.1f} to {dG_values.max():.1f} kJ/mol")
    print(f"  Number of points: {len(dG_values)}")
    sys.stdout.flush()

    # Run scan
    print("\n" + "=" * 80)
    print("Running 1D Scan")
    print("=" * 80)
    sys.stdout.flush()

    df = run_1d_scan(dG_values)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    sys.stdout.flush()

    csv_path = output_dir / "gibbs_comparison_1d.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    sys.stdout.flush()

    # Plot timecourse comparison
    print("\n  Creating timecourse comparison plot...")
    sys.stdout.flush()
    plot_comparison(df, output_dir / "gibbs_comparison_1d.png")

    # Plot final timepoint comparison
    print("  Creating final timepoint comparison plot...")
    sys.stdout.flush()
    plot_final_timepoint(df, output_dir / "gibbs_comparison_1d_final.png")

    # Summary
    min_idx_tc = df["SSR_timecourse"].idxmin()
    min_idx_final = df["SSR_final"].idxmin()
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nTimecourse Summary (1e-3s to 10s):")
    print(f"  Min SSR: {df['SSR_timecourse'].min():.6f} at FBA dG = {df.loc[min_idx_tc, 'FBA_deltaG']:.2f} kJ/mol")
    print(f"  Max SSR: {df['SSR_timecourse'].max():.6f}")

    print(f"\nFinal Timepoint Summary (t={FINAL_TIMEPOINT:.0f}s):")
    print(f"  Min SSR: {df['SSR_final'].min():.6f} at FBA dG = {df.loc[min_idx_final, 'FBA_deltaG']:.2f} kJ/mol")
    print(f"  Max SSR: {df['SSR_final'].max():.6f}")

    # Find SSR at dG = -5
    closest_to_5 = df.iloc[(df["FBA_deltaG"] + 5).abs().argsort()[:1]]
    print(f"\nSSR near dG=-5:")
    print(f"  Timecourse: {closest_to_5['SSR_timecourse'].values[0]:.6f}")
    print(f"  Final: {closest_to_5['SSR_final'].values[0]:.6f}")

    print(f"\nGenerated files in: {output_dir}")
    print("  - gibbs_comparison_1d.csv")
    print("  - gibbs_comparison_1d.png (timecourse)")
    print("  - gibbs_comparison_1d_final.png (final timepoint)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

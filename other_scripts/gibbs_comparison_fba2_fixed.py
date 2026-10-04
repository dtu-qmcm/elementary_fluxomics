#!/usr/bin/env python3
"""1D Negative Log-Likelihood Comparison: FBA2 Fixed, FBA1 Variable.

This script compares mechanistic model configurations where FBA2 is held constant
at -2.0 kJ/mol while FBA1 varies from -1.0 to -5.0 kJ/mol.

Reference (fixed): FBA1=-3.0, FBA2=-2.0 kJ/mol
Variable: FBA1 from -1.0 to -5.0 kJ/mol (FBA2 constant at -2.0)

Outputs:
- Aggregated NLL vs FBA1 parameter plot (timecourse and final)
- Timecourse NLL (aggregated and per-metabolite)
"""

from pathlib import Path
import sys

import cmcrameri.cm as cmc
import jax.numpy as jnp
import jax.scipy.stats as jstats
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
RESULTS_DIR = PROJECT_DIR / "results" / "other_scripts" / "gibbs_comparison_fba2_fixed"

# Gas constant in kJ/(mol*K)
R = 8.314e-3

# Timepoints: 100 points logarithmically spaced from 1e-3s to 1e1s (timecourse only)
# Steady-state is computed using freeflux SS mode, not long INST simulation
TIMEPOINTS = list(np.logspace(-3, 1, 100))

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
LOGNORMAL_SCALE = 0.05

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
    """Run INST simulation for timecourse and SS simulation for steady-state.

    Returns:
        Tuple of (mdvs_timecourse, mdvs_ss) where:
        - mdvs_timecourse: dict of {emu_id: {time: mdv}} for timecourse
        - mdvs_ss: dict of {emu_id: mdv} for steady-state
    """
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

    # Build flux dict for reuse in both INST and SS simulations
    reversible_rxns = {
        fid.rsplit("_", 1)[0] for fid in model.totalfluxids if fid.endswith("_f")
    }
    conversion_factor = 1000 / 3600

    fluxes = {}
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
                    fluxes[f"{rxn_id}_f"] = v_f
                    fluxes[f"{rxn_id}_b"] = v_b
                else:
                    fluxes[rxn_id] = v_net

    # 1. INST simulation for timecourse
    with model.simulator("inst") as isim:
        isim.set_target_EMUs(TARGET_EMUS)
        isim.set_timepoints(TIMEPOINTS)
        isim.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=["110000"],
            percentage=[1.0],
            purity=[1.0],
        )

        for flux_id, value in fluxes.items():
            isim.set_flux(flux_id, value)

        for metab in model.metabolites:
            isim.set_concentration(metab, 1.0)

        isim.prepare(n_jobs=1)
        res = isim.simulate()

        mdvs_timecourse = {}
        for emu_id in res.simulated_EMUs:
            mdv_result = res.simulated_MDV(emu_id)
            mdvs_timecourse[emu_id] = {t: np.array(mdv_result[t]) for t in res.timepoints}

    # 2. SS simulation for steady-state (algebraic solution)
    with model.simulator("ss") as ssim:
        ssim.set_target_EMUs(TARGET_EMUS)
        ssim.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=["110000"],
            percentage=[1.0],
            purity=[1.0],
        )

        for flux_id, value in fluxes.items():
            ssim.set_flux(flux_id, value)

        ssim.prepare(n_jobs=1)
        res_ss = ssim.simulate()

        mdvs_ss = {}
        for emu_id in res_ss.simulated_EMUs:
            mdvs_ss[emu_id] = np.array(res_ss.simulated_MDV(emu_id))

    return mdvs_timecourse, mdvs_ss


def compute_nll_at_timepoint(mdvs_ref, mdvs_test, timepoint, scale):
    """Compute negative log-likelihood at a single timepoint with per-metabolite breakdown.

    Uses jax.scipy.stats.norm.logpdf on log-transformed values.
    """
    total_nll = 0.0
    per_metabolite = {}

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

        # Negative log-likelihood using JAX norm.logpdf
        log_ref = jnp.log(jnp.array(mdv_ref) + EPSILON)
        log_test = jnp.log(jnp.array(mdv_test) + EPSILON)
        nll = -float(jnp.sum(jstats.norm.logpdf(log_test, loc=log_ref, scale=scale)))

        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])
        per_metabolite[met_name] = nll
        total_nll += nll

    return total_nll, per_metabolite


def compute_nll(mdvs_ref, mdvs_test, scale):
    """Compute negative log-likelihood between two sets of MDVs using log-normal error.

    Uses jax.scipy.stats.norm.logpdf on log-transformed values.
    """
    total_nll = 0.0

    for emu_id in mdvs_ref:
        if emu_id not in mdvs_test:
            continue
        for t in mdvs_ref[emu_id]:
            if t not in mdvs_test[emu_id]:
                continue
            mdv_ref = mdvs_ref[emu_id][t]
            mdv_test = mdvs_test[emu_id][t]

            mdv_ref = np.maximum(mdv_ref, 0.0)
            mdv_test = np.maximum(mdv_test, 0.0)
            if np.sum(mdv_ref) > 0:
                mdv_ref = mdv_ref / np.sum(mdv_ref)
            if np.sum(mdv_test) > 0:
                mdv_test = mdv_test / np.sum(mdv_test)

            # Negative log-likelihood using JAX norm.logpdf
            log_ref = jnp.log(jnp.array(mdv_ref) + EPSILON)
            log_test = jnp.log(jnp.array(mdv_test) + EPSILON)
            ll = jnp.sum(jstats.norm.logpdf(log_test, loc=log_ref, scale=scale))
            total_nll -= float(ll)

    return total_nll


def compute_nll_ss(mdvs_ref_ss, mdvs_test_ss, scale):
    """Compute negative log-likelihood between two sets of steady-state MDVs.

    Args:
        mdvs_ref_ss: Reference SS MDVs {emu_id: mdv_array}
        mdvs_test_ss: Test SS MDVs {emu_id: mdv_array}
        scale: Log-normal scale parameter

    Returns:
        Tuple of (total_nll, per_metabolite_dict)
    """
    total_nll = 0.0
    per_metabolite = {}

    for emu_id in mdvs_ref_ss:
        if emu_id not in mdvs_test_ss:
            continue

        mdv_ref = mdvs_ref_ss[emu_id]
        mdv_test = mdvs_test_ss[emu_id]

        mdv_ref = np.maximum(mdv_ref, 0.0)
        mdv_test = np.maximum(mdv_test, 0.0)
        if np.sum(mdv_ref) > 0:
            mdv_ref = mdv_ref / np.sum(mdv_ref)
        if np.sum(mdv_test) > 0:
            mdv_test = mdv_test / np.sum(mdv_test)

        # Negative log-likelihood using JAX norm.logpdf
        log_ref = jnp.log(jnp.array(mdv_ref) + EPSILON)
        log_test = jnp.log(jnp.array(mdv_test) + EPSILON)
        nll = -float(jnp.sum(jstats.norm.logpdf(log_test, loc=log_ref, scale=scale)))

        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])
        per_metabolite[met_name] = nll
        total_nll += nll

    return total_nll, per_metabolite


def run_1d_scan(dG_values):
    """Run 1D scan over FBA1 Gibbs energy with FBA2 fixed.

    Returns both aggregated NLL results and timecourse data.
    """
    print(f"Running reference model (FBA1={REFERENCE_FBA1}, FBA2={REFERENCE_FBA2})...")
    sys.stdout.flush()

    ref_overrides = {"FBA1": REFERENCE_FBA1, "FBA2": REFERENCE_FBA2}
    mdvs_ref_tc, mdvs_ref_ss = run_simulation(ref_overrides)

    # Filter timepoints for timecourse analysis
    filtered_timepoints = [t for t in TIMEPOINTS if t > 0]

    aggregated_results = []
    timecourse_results = []
    total = len(dG_values)

    for i, dG in enumerate(dG_values):
        print(f"  [{i+1}/{total}] FBA1={dG:.2f} kJ/mol, FBA2={FIXED_FBA2:.2f} (fixed)")
        sys.stdout.flush()

        thermo_overrides = {"FBA1": dG, "FBA2": FIXED_FBA2}

        try:
            mdvs_test_tc, mdvs_test_ss = run_simulation(thermo_overrides)
            nll_timecourse = compute_nll(mdvs_ref_tc, mdvs_test_tc, LOGNORMAL_SCALE)
            nll_final, _ = compute_nll_ss(mdvs_ref_ss, mdvs_test_ss, LOGNORMAL_SCALE)

            # Calculate per-timepoint, per-metabolite NLL
            for t in filtered_timepoints:
                total_nll, per_met = compute_nll_at_timepoint(
                    mdvs_ref_tc, mdvs_test_tc, t, LOGNORMAL_SCALE
                )
                row = {
                    "FBA1_deltaG": dG,
                    "FBA2_deltaG": FIXED_FBA2,
                    "timepoint": t,
                    "nll_total": total_nll,
                }
                for met_id, nll in per_met.items():
                    row[f"nll_{met_id}"] = nll
                timecourse_results.append(row)

        except Exception as e:
            print(f"    Error: {e}")
            sys.stdout.flush()
            nll_timecourse = np.nan
            nll_final = np.nan

        aggregated_results.append(
            {
                "FBA1_deltaG": dG,
                "FBA2_deltaG": FIXED_FBA2,
                "nll_timecourse": nll_timecourse,
                "nll_final": nll_final,
            }
        )

    return pd.DataFrame(aggregated_results), pd.DataFrame(timecourse_results)


def plot_comparison(df, output_path):
    """Plot 1D NLL comparison for timecourse."""
    fig, ax = plt.subplots(figsize=(10, 6))

    color = cmc.batlow(0.3)
    ax.plot(
        df["FBA1_deltaG"], df["nll_timecourse"], "o-", linewidth=2, markersize=6, color=color
    )

    # Mark reference FBA1 value
    ax.axvline(
        x=REFERENCE_FBA1,
        color=cmc.batlow(0.85),
        linestyle="--",
        linewidth=1.5,
        label=f"Reference FBA1={REFERENCE_FBA1}",
    )

    ax.set_xlabel(f"FBA1 dG (kJ/mol) [FBA2 fixed at {FIXED_FBA2}]", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Timecourse)", fontsize=12)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_final_timepoint(df, output_path):
    """Plot 1D NLL comparison for steady-state."""
    fig, ax = plt.subplots(figsize=(10, 6))

    color = cmc.batlow(0.6)
    ax.plot(
        df["FBA1_deltaG"], df["nll_final"], "s-", linewidth=2, markersize=6, color=color,
        label="Steady-State NLL",
    )

    # Mark reference FBA1 value
    ax.axvline(
        x=REFERENCE_FBA1,
        color=cmc.batlow(0.85),
        linestyle="--",
        linewidth=1.5,
        label=f"Reference FBA1={REFERENCE_FBA1}",
    )

    ax.set_xlabel(f"FBA1 dG (kJ/mol) [FBA2 fixed at {FIXED_FBA2}]", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Steady-State)", fontsize=12)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_timecourse_aggregated(df, output_path):
    """Plot aggregated NLL timecourse for all test conditions."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique FBA1 values
    fba1_values = sorted(df["FBA1_deltaG"].unique())
    colors = [cmc.batlow(i / len(fba1_values)) for i in range(len(fba1_values))]

    for idx, fba1 in enumerate(fba1_values):
        condition_data = df[df["FBA1_deltaG"] == fba1].sort_values("timepoint")
        label = f"FBA1={fba1:.2f}"
        if abs(fba1 - REFERENCE_FBA1) < 0.1:
            label += " (ref)"
        ax.plot(
            condition_data["timepoint"],
            condition_data["nll_total"],
            linewidth=1.5,
            color=colors[idx],
            label=label,
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Total)", fontsize=12)
    ax.set_title("NLL Over Time: FBA2 Fixed, FBA1 Variable", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_timecourse_per_metabolite(df, output_path):
    """Plot per-metabolite NLL timecourse in subplots."""
    metabolites = ["g6p", "f6p", "dhap", "g3p"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fba1_values = sorted(df["FBA1_deltaG"].unique())
    colors = [cmc.batlow(i / len(fba1_values)) for i in range(len(fba1_values))]

    for ax, met_id in zip(axes, metabolites):
        col_name = f"nll_{met_id}"
        if col_name not in df.columns:
            ax.set_visible(False)
            continue

        for idx, fba1 in enumerate(fba1_values):
            condition_data = df[df["FBA1_deltaG"] == fba1].sort_values("timepoint")
            label = f"FBA1={fba1:.1f}"
            ax.plot(
                condition_data["timepoint"],
                condition_data[col_name],
                "-",
                color=colors[idx],
                linewidth=1.5,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Negative Log-Likelihood", fontsize=10)
        ax.set_title(f"{met_id.upper()}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Add legend to figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.02, 0.5))

    plt.suptitle("Per-Metabolite NLL Over Time (FBA2 Fixed)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def main():
    """Run 1D comparison analysis with FBA2 fixed."""
    print("=" * 80)
    print("1D Negative Log-Likelihood Comparison: FBA2 Fixed, FBA1 Variable")
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

    timecourse_csv_path = output_dir / "timecourse_nll.csv"
    df_timecourse.to_csv(timecourse_csv_path, index=False)
    print(f"  Saved CSV: {timecourse_csv_path}")
    sys.stdout.flush()

    # Plot timecourse NLL comparison
    print("\n  Creating timecourse comparison plot...")
    sys.stdout.flush()
    plot_comparison(df_agg, output_dir / "gibbs_comparison_fba2_fixed.png")

    # Plot steady-state NLL comparison
    print("  Creating steady-state comparison plot...")
    sys.stdout.flush()
    plot_final_timepoint(df_agg, output_dir / "gibbs_comparison_fba2_fixed_final.png")

    # Plot timecourse
    print("  Creating timecourse plots...")
    sys.stdout.flush()
    plot_timecourse_aggregated(df_timecourse, output_dir / "timecourse_aggregated.png")
    plot_timecourse_per_metabolite(df_timecourse, output_dir / "timecourse_per_metabolite.png")

    # Summary (lower NLL is better)
    min_idx_tc = df_agg["nll_timecourse"].idxmin()
    min_idx_final = df_agg["nll_final"].idxmin()
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nTimecourse Summary (1e-3s to 10s):")
    print(
        f"  Min NLL: {df_agg['nll_timecourse'].min():.2f} at FBA1={df_agg.loc[min_idx_tc, 'FBA1_deltaG']:.2f} kJ/mol"
    )
    print(f"  Max NLL: {df_agg['nll_timecourse'].max():.2f}")

    print(f"\nSteady-State Summary:")
    print(
        f"  Min NLL: {df_agg['nll_final'].min():.2f} at FBA1={df_agg.loc[min_idx_final, 'FBA1_deltaG']:.2f} kJ/mol"
    )
    print(f"  Max NLL: {df_agg['nll_final'].max():.2f}")

    # Find NLL at reference FBA1
    closest_to_ref = df_agg.iloc[
        (df_agg["FBA1_deltaG"] - REFERENCE_FBA1).abs().argsort()[:1]
    ]
    print(f"\nNLL at reference FBA1={REFERENCE_FBA1}:")
    print(f"  Timecourse: {closest_to_ref['nll_timecourse'].values[0]:.2f}")
    print(f"  Steady-State: {closest_to_ref['nll_final'].values[0]:.2f}")

    print(f"\nGenerated files in: {output_dir}")
    print("  - gibbs_comparison_fba2_fixed.csv")
    print("  - gibbs_comparison_fba2_fixed.png (timecourse)")
    print("  - gibbs_comparison_fba2_fixed_final.png (final timepoint)")
    print("  - timecourse_nll.csv")
    print("  - timecourse_aggregated.png")
    print("  - timecourse_per_metabolite.png")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

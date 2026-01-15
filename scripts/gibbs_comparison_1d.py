#!/usr/bin/env python3
"""1D Negative Log-Likelihood Comparison: Mechanistic vs Non-Mechanistic Model.

This script compares the mechanistic glycolysis model (FBA1/FBA2) against
the non-mechanistic model (single FBA) as the FBA dG varies.

Reference (fixed): Mechanistic model with FBA1=-3, FBA2=-2 kJ/mol
Variable: Non-mechanistic model with FBA dG from -9 to -1 kJ/mol
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

# EMU ID to metabolite name mapping
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


def compute_nll(mdvs_ref, mdvs_test, scale, exclude_timepoint=None):
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

            # Negative log-likelihood using JAX norm.logpdf on log-transformed values
            log_ref = jnp.log(jnp.array(mdv_ref) + EPSILON)
            log_test = jnp.log(jnp.array(mdv_test) + EPSILON)
            ll = jnp.sum(jstats.norm.logpdf(log_test, loc=log_ref, scale=scale))
            total_nll -= float(ll)

    return total_nll


def compute_nll_at_timepoint(mdvs_ref, mdvs_test, scale, target_timepoint):
    """Compute negative log-likelihood at a specific timepoint with per-metabolite breakdown."""
    total_nll = 0.0
    per_metabolite = {}

    for emu_id in mdvs_ref:
        if emu_id not in mdvs_test:
            continue
        if target_timepoint not in mdvs_ref[emu_id] or target_timepoint not in mdvs_test[emu_id]:
            continue

        mdv_ref = mdvs_ref[emu_id][target_timepoint]
        mdv_test = mdvs_test[emu_id][target_timepoint]

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
    """Run 1D scan over FBA Gibbs energy of non-mechanistic model.

    Returns aggregated results, per-metabolite timecourse data, and reference NLLs.
    """
    print("Running mechanistic model (FBA1=-3, FBA2=-2)...")
    sys.stdout.flush()

    mech_overrides = {"FBA1": -3.0, "FBA2": -2.0}
    mdvs_mech = run_simulation(MECH_MODEL_DIR, "glycolysis_mech_gly3p", mech_overrides)

    # Compute reference NLL (mechanistic model compared to itself)
    print("Computing reference NLL (mechanistic vs itself)...")
    sys.stdout.flush()
    ref_nll_timecourse = compute_nll(
        mdvs_mech, mdvs_mech, LOGNORMAL_SCALE, exclude_timepoint=FINAL_TIMEPOINT
    )
    ref_nll_final, _ = compute_nll_at_timepoint(
        mdvs_mech, mdvs_mech, LOGNORMAL_SCALE, FINAL_TIMEPOINT
    )
    ref_nll = {"timecourse": ref_nll_timecourse, "final": ref_nll_final}
    print(f"  Reference NLL - Timecourse: {ref_nll_timecourse:.2f}, Final: {ref_nll_final:.2f}")

    # Filter timepoints for timecourse analysis
    filtered_timepoints = [t for t in TIMEPOINTS if t > 0 and abs(t - FINAL_TIMEPOINT) > 1e-6]

    aggregated_results = []
    timecourse_results = []
    total = len(dG_values)

    for i, dG in enumerate(dG_values):
        print(f"  [{i+1}/{total}] Non-mechanistic FBA dG = {dG:.2f} kJ/mol")
        sys.stdout.flush()

        thermo_overrides = {"FBA": dG}

        try:
            mdvs_std = run_simulation(STD_MODEL_DIR, "glycolysis_gly3p", thermo_overrides)
            nll_timecourse = compute_nll(
                mdvs_mech, mdvs_std, LOGNORMAL_SCALE, exclude_timepoint=FINAL_TIMEPOINT
            )
            nll_final, _ = compute_nll_at_timepoint(
                mdvs_mech, mdvs_std, LOGNORMAL_SCALE, FINAL_TIMEPOINT
            )

            # Calculate per-timepoint, per-metabolite negative log-likelihood
            for t in filtered_timepoints:
                total_nll, per_met = compute_nll_at_timepoint(
                    mdvs_mech, mdvs_std, LOGNORMAL_SCALE, t
                )
                row = {
                    "FBA_deltaG": dG,
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
                "FBA_deltaG": dG,
                "nll_timecourse": nll_timecourse,
                "nll_final": nll_final,
            }
        )

    return pd.DataFrame(aggregated_results), pd.DataFrame(timecourse_results), ref_nll


def plot_comparison(df, output_path, ref_nll=None):
    """Plot 1D negative log-likelihood comparison for timecourse data."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use batlow colormap
    color = cmc.batlow(0.3)

    ax.plot(
        df["FBA_deltaG"],
        df["nll_timecourse"],
        "o-",
        linewidth=2,
        markersize=6,
        color=color,
        label="Timecourse NLL (1e-3s to 10s)",
    )

    # Mark where FBA dG = -5 (sum of FBA1 + FBA2 = -3 + -2)
    ax.axvline(
        x=-5,
        color=cmc.batlow(0.85),
        linestyle="--",
        linewidth=1.5,
        label="dG(FBA) = dG(FBA1) + dG(FBA2) = -5",
    )

    # Add horizontal line for reference NLL
    if ref_nll is not None:
        ax.axhline(
            y=ref_nll["timecourse"],
            color=cmc.batlow(0.5),
            linestyle=":",
            linewidth=2,
            label=f"Reference NLL (mechanistic) = {ref_nll['timecourse']:.1f}",
        )

    ax.set_xlabel("Non-mechanistic FBA dG (kJ/mol)", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Timecourse)", fontsize=12)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_final_timepoint(df, output_path, ref_nll=None):
    """Plot 1D negative log-likelihood comparison for final timepoint."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use batlow colormap - different color for final
    color = cmc.batlow(0.6)

    ax.plot(
        df["FBA_deltaG"],
        df["nll_final"],
        "s-",
        linewidth=2,
        markersize=6,
        color=color,
        label=f"Final timepoint NLL (t={FINAL_TIMEPOINT:.0f}s)",
    )

    # Mark where FBA dG = -5 (sum of FBA1 + FBA2 = -3 + -2)
    ax.axvline(
        x=-5,
        color=cmc.batlow(0.85),
        linestyle="--",
        linewidth=1.5,
        label="dG(FBA) = dG(FBA1) + dG(FBA2) = -5",
    )

    # Add horizontal line for reference NLL
    if ref_nll is not None:
        ax.axhline(
            y=ref_nll["final"],
            color=cmc.batlow(0.5),
            linestyle=":",
            linewidth=2,
            label=f"Reference NLL (mechanistic) = {ref_nll['final']:.1f}",
        )

    ax.set_xlabel("Non-mechanistic FBA dG (kJ/mol)", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Final Timepoint)", fontsize=12)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


def plot_timecourse_aggregated(df_timecourse, output_path):
    """Plot aggregated negative log-likelihood timecourse for all test conditions."""
    fig, ax = plt.subplots(figsize=(12, 7))

    dG_values = sorted(df_timecourse["FBA_deltaG"].unique())
    colors = [cmc.batlow(i / len(dG_values)) for i in range(len(dG_values))]

    for dG, color in zip(dG_values, colors):
        condition_data = df_timecourse[df_timecourse["FBA_deltaG"] == dG].sort_values("timepoint")
        ax.plot(
            condition_data["timepoint"],
            condition_data["nll_total"],
            "-",
            color=color,
            linewidth=1.5,
            label=f"FBA dG={dG:.1f}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Total)", fontsize=12)
    ax.set_title("NLL Over Time: Non-mechanistic vs Mechanistic Reference", fontsize=14)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()
    plt.close()


def plot_timecourse_per_metabolite(df_timecourse, output_path):
    """Plot per-metabolite negative log-likelihood timecourse in subplots."""
    metabolites = ["g6p", "f6p", "dhap", "g3p"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    dG_values = sorted(df_timecourse["FBA_deltaG"].unique())
    colors = [cmc.batlow(i / len(dG_values)) for i in range(len(dG_values))]

    for ax, met_id in zip(axes, metabolites):
        col_name = f"nll_{met_id}"
        if col_name not in df_timecourse.columns:
            ax.set_visible(False)
            continue

        for dG, color in zip(dG_values, colors):
            condition_data = df_timecourse[df_timecourse["FBA_deltaG"] == dG].sort_values("timepoint")
            ax.plot(
                condition_data["timepoint"],
                condition_data[col_name],
                "-",
                color=color,
                linewidth=1.5,
                label=f"FBA dG={dG:.1f}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Negative Log-Likelihood", fontsize=10)
        ax.set_title(f"{met_id.upper()}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Add legend to last subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.02, 0.5))

    plt.suptitle("Per-Metabolite Negative Log-Likelihood Over Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()
    plt.close()


def main():
    """Run 1D comparison analysis."""
    print("=" * 80)
    print("1D Negative Log-Likelihood Comparison")
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

    df, df_timecourse, ref_nll = run_1d_scan(dG_values)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    sys.stdout.flush()

    csv_path = output_dir / "gibbs_comparison_1d.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    timecourse_csv_path = output_dir / "timecourse_nll.csv"
    df_timecourse.to_csv(timecourse_csv_path, index=False)
    print(f"  Saved CSV: {timecourse_csv_path}")
    sys.stdout.flush()

    # Plot timecourse comparison
    print("\n  Creating timecourse comparison plot...")
    sys.stdout.flush()
    plot_comparison(df, output_dir / "gibbs_comparison_1d.png", ref_nll=ref_nll)

    # Plot final timepoint comparison
    print("  Creating final timepoint comparison plot...")
    sys.stdout.flush()
    plot_final_timepoint(df, output_dir / "gibbs_comparison_1d_final.png", ref_nll=ref_nll)

    # Plot timecourse aggregated
    print("  Creating aggregated timecourse plot...")
    sys.stdout.flush()
    plot_timecourse_aggregated(df_timecourse, output_dir / "timecourse_aggregated.png")

    # Plot per-metabolite timecourse
    print("  Creating per-metabolite timecourse plot...")
    sys.stdout.flush()
    plot_timecourse_per_metabolite(df_timecourse, output_dir / "timecourse_per_metabolite.png")

    # Summary (lower NLL is better)
    min_idx_tc = df["nll_timecourse"].idxmin()
    min_idx_final = df["nll_final"].idxmin()
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    print(f"\nReference NLL (mechanistic model vs itself):")
    print(f"  Timecourse: {ref_nll['timecourse']:.2f}")
    print(f"  Final: {ref_nll['final']:.2f}")

    print(f"\nTimecourse Summary (1e-3s to 10s):")
    print(f"  Min NLL: {df['nll_timecourse'].min():.2f} at FBA dG = {df.loc[min_idx_tc, 'FBA_deltaG']:.2f} kJ/mol")
    print(f"  Max NLL: {df['nll_timecourse'].max():.2f}")

    print(f"\nFinal Timepoint Summary (t={FINAL_TIMEPOINT:.0f}s):")
    print(f"  Min NLL: {df['nll_final'].min():.2f} at FBA dG = {df.loc[min_idx_final, 'FBA_deltaG']:.2f} kJ/mol")
    print(f"  Max NLL: {df['nll_final'].max():.2f}")

    # Find NLL at dG = -5
    closest_to_5 = df.iloc[(df["FBA_deltaG"] + 5).abs().argsort()[:1]]
    print(f"\nNegative Log-Likelihood near dG=-5:")
    print(f"  Timecourse: {closest_to_5['nll_timecourse'].values[0]:.2f}")
    print(f"  Final: {closest_to_5['nll_final'].values[0]:.2f}")

    print(f"\nGenerated files in: {output_dir}")
    print("  - gibbs_comparison_1d.csv")
    print("  - timecourse_nll.csv")
    print("  - gibbs_comparison_1d.png (timecourse)")
    print("  - gibbs_comparison_1d_final.png (final timepoint)")
    print("  - timecourse_aggregated.png")
    print("  - timecourse_per_metabolite.png")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

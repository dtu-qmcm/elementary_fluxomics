#!/usr/bin/env python3
"""FBA Gibbs Energy Split Comparison: Mechanistic vs Aggregated Model.

This script varies how the total FBA Gibbs energy (-5.09 kJ/mol) is split between
FBA1 and FBA2 in the mechanistic model, and compares each split against the
aggregated (non-mechanistic) model.

Reference (fixed): Aggregated model with FBA dG = -5.09 kJ/mol
Variable: Mechanistic model with FBA1 = f * (-5.09), FBA2 = (1-f) * (-5.09)
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
RESULTS_DIR = PROJECT_DIR / "results" / "gibbs_fba_split"

# Total FBA Gibbs energy from aggregated model (kJ/mol)
TOTAL_FBA_DG = -5.09

# Fraction range: 0.1 to 0.9, step 0.05 (17 values)
FRACTIONS = np.arange(0.1, 0.91, 0.05)

# Gas constant in kJ/(mol*K)
R = 8.314e-3

# Enzyme-bound intermediates (FBA_dhap, and the transketolase/transaldolase
# EC2/EC3 carriers) are given a small pool rather than the bulk default.
ENZYME_BOUND_METABOLITES = ("FBA_dhap", "EC2", "EC3")

# Timepoints: 100 points logarithmically spaced from 1e-3s to 1e1s
TIMEPOINTS = list(np.logspace(-3, 1, 100))

# Target EMUs to compare
TARGET_EMUS = {
    "g6p": ["123456"],
    "f6p": ["123456"],
    "dhap": ["123"],
    "g3p": ["123"],
    "r5p": ["12345"],
}

# EMU ID to metabolite name mapping
EMU_TO_METABOLITE = {
    "g6p_123456": "g6p",
    "f6p_123456": "f6p",
    "dhap_123": "dhap",
    "g3p_123": "g3p",
    "r5p_12345": "r5p",
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
    """Run INST simulation for timecourse and SS simulation for steady-state.

    Returns:
        Tuple of (mdvs_timecourse, mdvs_ss) where:
        - mdvs_timecourse: dict of {emu_id: {time: mdv}} for timecourse
        - mdvs_ss: dict of {emu_id: mdv} for steady-state
    """
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
            if metab in ENZYME_BOUND_METABOLITES:
                isim.set_concentration(metab, 0.01)
            else:
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


def compute_nll(mdvs_ref, mdvs_test, scale):
    """Compute negative log-likelihood between two sets of MDVs using log-normal error."""
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

        log_ref = jnp.log(jnp.array(mdv_ref) + EPSILON)
        log_test = jnp.log(jnp.array(mdv_test) + EPSILON)
        nll = -float(jnp.sum(jstats.norm.logpdf(log_test, loc=log_ref, scale=scale)))

        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])
        per_metabolite[met_name] = nll
        total_nll += nll

    return total_nll, per_metabolite


def compute_nll_ss(mdvs_ref_ss, mdvs_test_ss, scale):
    """Compute negative log-likelihood between two sets of steady-state MDVs."""
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

        log_ref = jnp.log(jnp.array(mdv_ref) + EPSILON)
        log_test = jnp.log(jnp.array(mdv_test) + EPSILON)
        nll = -float(jnp.sum(jstats.norm.logpdf(log_test, loc=log_ref, scale=scale)))

        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])
        per_metabolite[met_name] = nll
        total_nll += nll

    return total_nll, per_metabolite


def compute_max_isotope_diff(mdvs_ref, mdvs_test):
    """Compute maximum absolute difference in isotope fractions.

    Returns dict of {metabolite: max_diff} where max_diff is the
    largest |ref[i] - test[i]| across all isotopomers and timepoints.
    """
    max_diffs = {}

    for emu_id in mdvs_ref:
        if emu_id not in mdvs_test:
            continue

        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])
        max_diff = 0.0

        for t in mdvs_ref[emu_id]:
            if t not in mdvs_test[emu_id]:
                continue

            mdv_ref = mdvs_ref[emu_id][t]
            mdv_test = mdvs_test[emu_id][t]

            # Normalize to ensure they sum to 1
            mdv_ref = np.maximum(mdv_ref, 0.0)
            mdv_test = np.maximum(mdv_test, 0.0)
            if np.sum(mdv_ref) > 0:
                mdv_ref = mdv_ref / np.sum(mdv_ref)
            if np.sum(mdv_test) > 0:
                mdv_test = mdv_test / np.sum(mdv_test)

            # Compute max absolute difference across isotopomers
            diff = np.max(np.abs(mdv_ref - mdv_test))
            max_diff = max(max_diff, diff)

        max_diffs[met_name] = max_diff

    return max_diffs


def compute_max_isotope_diff_ss(mdvs_ref_ss, mdvs_test_ss):
    """Compute maximum absolute difference in steady-state isotope fractions."""
    max_diffs = {}

    for emu_id in mdvs_ref_ss:
        if emu_id not in mdvs_test_ss:
            continue

        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])

        mdv_ref = mdvs_ref_ss[emu_id]
        mdv_test = mdvs_test_ss[emu_id]

        # Normalize
        mdv_ref = np.maximum(mdv_ref, 0.0)
        mdv_test = np.maximum(mdv_test, 0.0)
        if np.sum(mdv_ref) > 0:
            mdv_ref = mdv_ref / np.sum(mdv_ref)
        if np.sum(mdv_test) > 0:
            mdv_test = mdv_test / np.sum(mdv_test)

        max_diffs[met_name] = np.max(np.abs(mdv_ref - mdv_test))

    return max_diffs


def run_fba_split_scan(fractions):
    """Run scan over FBA1/FBA2 energy split fractions.

    Returns aggregated results, per-metabolite timecourse data, and max isotope diff data.
    """
    # Run aggregated (non-mechanistic) model once as reference
    print("Running aggregated (non-mechanistic) model as reference...")
    print(f"  FBA dG = {TOTAL_FBA_DG:.2f} kJ/mol")
    sys.stdout.flush()

    mdvs_ref_tc, mdvs_ref_ss = run_simulation(
        STD_MODEL_DIR, "glycolysis_gly3p", {"FBA": TOTAL_FBA_DG}
    )

    # Filter timepoints for timecourse analysis
    filtered_timepoints = [t for t in TIMEPOINTS if t > 0]

    aggregated_results = []
    timecourse_results = []
    isotope_diff_results = []
    total = len(fractions)

    for i, frac in enumerate(fractions):
        fba1_dG = frac * TOTAL_FBA_DG
        fba2_dG = (1 - frac) * TOTAL_FBA_DG

        print(f"  [{i+1}/{total}] Fraction={frac:.2f}: FBA1={fba1_dG:.3f}, FBA2={fba2_dG:.3f} kJ/mol")
        sys.stdout.flush()

        thermo_overrides = {"FBA1": fba1_dG, "FBA2": fba2_dG}

        try:
            mdvs_mech_tc, mdvs_mech_ss = run_simulation(
                MECH_MODEL_DIR, "glycolysis_mech_gly3p", thermo_overrides
            )

            # Compute NLL (mechanistic vs aggregated reference)
            nll_timecourse = compute_nll(mdvs_ref_tc, mdvs_mech_tc, LOGNORMAL_SCALE)
            nll_ss, _ = compute_nll_ss(mdvs_ref_ss, mdvs_mech_ss, LOGNORMAL_SCALE)

            # Compute max isotope difference
            max_diff_tc = compute_max_isotope_diff(mdvs_ref_tc, mdvs_mech_tc)
            max_diff_ss = compute_max_isotope_diff_ss(mdvs_ref_ss, mdvs_mech_ss)

            # Calculate per-timepoint, per-metabolite NLL
            for t in filtered_timepoints:
                total_nll, per_met = compute_nll_at_timepoint(
                    mdvs_ref_tc, mdvs_mech_tc, LOGNORMAL_SCALE, t
                )
                row = {
                    "fraction": frac,
                    "FBA1_deltaG": fba1_dG,
                    "FBA2_deltaG": fba2_dG,
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
            nll_ss = np.nan
            max_diff_tc = {m: np.nan for m in ["g6p", "f6p", "dhap", "g3p", "r5p"]}
            max_diff_ss = {m: np.nan for m in ["g6p", "f6p", "dhap", "g3p", "r5p"]}

        # Aggregated results
        aggregated_results.append({
            "fraction": frac,
            "FBA1_deltaG": fba1_dG,
            "FBA2_deltaG": fba2_dG,
            "nll_timecourse": nll_timecourse,
            "nll_ss": nll_ss,
        })

        # Isotope diff results
        isotope_row = {
            "fraction": frac,
            "FBA1_deltaG": fba1_dG,
            "FBA2_deltaG": fba2_dG,
        }
        max_overall_tc = 0.0
        max_overall_ss = 0.0
        for met in ["g6p", "f6p", "dhap", "g3p", "r5p"]:
            isotope_row[f"max_diff_tc_{met}"] = max_diff_tc.get(met, np.nan)
            isotope_row[f"max_diff_ss_{met}"] = max_diff_ss.get(met, np.nan)
            if not np.isnan(max_diff_tc.get(met, np.nan)):
                max_overall_tc = max(max_overall_tc, max_diff_tc[met])
            if not np.isnan(max_diff_ss.get(met, np.nan)):
                max_overall_ss = max(max_overall_ss, max_diff_ss[met])
        isotope_row["max_diff_tc_overall"] = max_overall_tc
        isotope_row["max_diff_ss_overall"] = max_overall_ss
        isotope_diff_results.append(isotope_row)

    return (
        pd.DataFrame(aggregated_results),
        pd.DataFrame(timecourse_results),
        pd.DataFrame(isotope_diff_results),
    )


def plot_nll_timecourse(df, output_path):
    """Plot timecourse NLL vs fraction."""
    fig, ax = plt.subplots(figsize=(10, 6))

    color = cmc.batlow(0.3)

    ax.plot(
        df["fraction"],
        df["nll_timecourse"],
        "o-",
        linewidth=2,
        markersize=6,
        color=color,
        label="Timecourse NLL (1e-3s to 10s)",
    )

    ax.set_xlabel("FBA1 Fraction of Total dG", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Timecourse)", fontsize=12)
    ax.set_title(f"Timecourse NLL: Mechanistic vs Aggregated (Total FBA dG = {TOTAL_FBA_DG:.2f} kJ/mol)", fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()
    plt.close()


def plot_nll_ss(df, output_path):
    """Plot steady-state NLL vs fraction."""
    fig, ax = plt.subplots(figsize=(10, 6))

    color = cmc.batlow(0.7)

    ax.plot(
        df["fraction"],
        df["nll_ss"],
        "s-",
        linewidth=2,
        markersize=6,
        color=color,
        label="Steady-State NLL",
    )

    ax.set_xlabel("FBA1 Fraction of Total dG", fontsize=12)
    ax.set_ylabel("Negative Log-Likelihood (Steady-State)", fontsize=12)
    ax.set_title(f"Steady-State NLL: Mechanistic vs Aggregated (Total FBA dG = {TOTAL_FBA_DG:.2f} kJ/mol)", fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()
    plt.close()


def plot_max_isotope_diff(df, output_path):
    """Plot maximum isotope difference vs fraction per metabolite."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metabolites = ["g6p", "f6p", "dhap", "g3p", "r5p"]
    colors = [cmc.batlow(i / len(metabolites)) for i in range(len(metabolites))]

    for met, color in zip(metabolites, colors):
        col_name = f"max_diff_tc_{met}"
        if col_name in df.columns:
            ax.plot(
                df["fraction"],
                df[col_name],
                "o-",
                linewidth=2,
                markersize=5,
                color=color,
                label=met.upper(),
            )

    ax.set_xlabel("FBA1 Fraction of Total dG", fontsize=12)
    ax.set_ylabel("Max Absolute Isotope Fraction Difference", fontsize=12)
    ax.set_title("Maximum Isotope Difference: Mechanistic vs Aggregated (Timecourse)", fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()
    plt.close()


def plot_timecourse_per_metabolite(df_timecourse, output_path):
    """Plot per-metabolite NLL timecourse in subplots."""
    metabolites = ["g6p", "f6p", "dhap", "g3p", "r5p"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()

    fractions = sorted(df_timecourse["fraction"].unique())
    colors = [cmc.batlow(i / len(fractions)) for i in range(len(fractions))]

    for ax, met_id in zip(axes, metabolites):
        col_name = f"nll_{met_id}"
        if col_name not in df_timecourse.columns:
            ax.set_visible(False)
            continue

        for frac, color in zip(fractions, colors):
            condition_data = df_timecourse[df_timecourse["fraction"] == frac].sort_values("timepoint")
            ax.plot(
                condition_data["timepoint"],
                condition_data[col_name],
                "-",
                color=color,
                linewidth=1.5,
                label=f"f={frac:.2f}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Negative Log-Likelihood", fontsize=10)
        ax.set_title(f"{met_id.upper()}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(metabolites):]:
        ax.set_visible(False)

    # Add legend to last subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, bbox_to_anchor=(1.02, 0.5))

    plt.suptitle("Per-Metabolite NLL Over Time (Mechanistic vs Aggregated)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()
    plt.close()


def main():
    """Run FBA split comparison analysis."""
    print("=" * 80)
    print("FBA Gibbs Energy Split Comparison")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()

    print(f"\nConfiguration:")
    print(f"  Total FBA dG: {TOTAL_FBA_DG:.2f} kJ/mol")
    print(f"  Fraction range: {FRACTIONS.min():.2f} to {FRACTIONS.max():.2f}")
    print(f"  Number of splits: {len(FRACTIONS)}")
    sys.stdout.flush()

    # Run scan
    print("\n" + "=" * 80)
    print("Running FBA Split Scan")
    print("=" * 80)
    sys.stdout.flush()

    df, df_timecourse, df_isotope = run_fba_split_scan(FRACTIONS)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    sys.stdout.flush()

    csv_path = output_dir / "gibbs_fba_split.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    timecourse_csv_path = output_dir / "timecourse_nll.csv"
    df_timecourse.to_csv(timecourse_csv_path, index=False)
    print(f"  Saved CSV: {timecourse_csv_path}")

    isotope_csv_path = output_dir / "max_isotope_diff_summary.csv"
    df_isotope.to_csv(isotope_csv_path, index=False)
    print(f"  Saved CSV: {isotope_csv_path}")
    sys.stdout.flush()

    # Create plots
    print("\n  Creating timecourse NLL vs fraction plot...")
    sys.stdout.flush()
    plot_nll_timecourse(df, output_dir / "gibbs_fba_split_timecourse.png")

    print("  Creating steady-state NLL vs fraction plot...")
    sys.stdout.flush()
    plot_nll_ss(df, output_dir / "gibbs_fba_split_ss.png")

    print("  Creating max isotope diff plot...")
    sys.stdout.flush()
    plot_max_isotope_diff(df_isotope, output_dir / "max_isotope_diff.png")

    print("  Creating per-metabolite timecourse plot...")
    sys.stdout.flush()
    plot_timecourse_per_metabolite(df_timecourse, output_dir / "timecourse_per_metabolite.png")

    # Summary
    min_idx_tc = df["nll_timecourse"].idxmin()
    min_idx_ss = df["nll_ss"].idxmin()
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    print(f"\nTimecourse Summary:")
    print(f"  Min NLL: {df['nll_timecourse'].min():.2f} at fraction = {df.loc[min_idx_tc, 'fraction']:.2f}")
    print(f"  (FBA1 = {df.loc[min_idx_tc, 'FBA1_deltaG']:.3f}, FBA2 = {df.loc[min_idx_tc, 'FBA2_deltaG']:.3f} kJ/mol)")

    print(f"\nSteady-State Summary:")
    print(f"  Min NLL: {df['nll_ss'].min():.2f} at fraction = {df.loc[min_idx_ss, 'fraction']:.2f}")
    print(f"  (FBA1 = {df.loc[min_idx_ss, 'FBA1_deltaG']:.3f}, FBA2 = {df.loc[min_idx_ss, 'FBA2_deltaG']:.3f} kJ/mol)")

    # Find max isotope diff summary
    print(f"\nMaximum Isotope Differences (timecourse):")
    for met in ["g6p", "f6p", "dhap", "g3p", "r5p"]:
        col = f"max_diff_tc_{met}"
        max_val = df_isotope[col].max()
        max_idx = df_isotope[col].idxmax()
        print(f"  {met.upper()}: {max_val:.4f} at fraction = {df_isotope.loc[max_idx, 'fraction']:.2f}")

    print(f"\nOverall max isotope diff (timecourse): {df_isotope['max_diff_tc_overall'].max():.4f}")

    print(f"\nGenerated files in: {output_dir}")
    print("  - gibbs_fba_split.csv")
    print("  - timecourse_nll.csv")
    print("  - max_isotope_diff_summary.csv")
    print("  - gibbs_fba_split_timecourse.png")
    print("  - gibbs_fba_split_ss.png")
    print("  - max_isotope_diff.png")
    print("  - timecourse_per_metabolite.png")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

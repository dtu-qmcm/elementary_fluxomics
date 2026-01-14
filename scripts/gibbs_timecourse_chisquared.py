#!/usr/bin/env python3
"""Gibbs Energy Timecourse Chi-Squared Analysis using freeflux.

This script evaluates how chi-squared goodness-of-fit evolves over time for
different Gibbs energy parameters where FBA1 + FBA2 = -5 kJ/mol.

Analysis:
- Reference: FBA1=-3.0, FBA2=-2.0 kJ/mol
- Test conditions: 5 parameter sets where FBA1 ranges from -1 to -4 kJ/mol, FBA2 = -5 - FBA1
- Chi-squared calculated at each timepoint individually (not cumulative)
- Two visualizations:
  1. Aggregated chi-squared timecourse (all metabolites combined)
  2. Per-metabolite chi-squared (separate subplots for each condition)
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
REACTIONS_FILE = MODEL_DIR / "reactions.tsv"
FLUXES_FILE = MODEL_DIR / "glycolysis_fluxes.toml"
THERMO_FILE = MODEL_DIR / "glycolysis_thermodynamics.toml"
RESULTS_DIR = PROJECT_DIR / "results" / "gibbs_timecourse"

# Gas constant in kJ/(mol*K)
R = 8.314e-3

# Timepoints: 100 points logarithmically spaced from 1e-3s to 1e1s, plus final at 1e3s
TIMEPOINTS = list(np.logspace(np.log10(1e-3), np.log10(1e1), 100)) + [1e3]
FINAL_TIMEPOINT = 1e3  # Excluded from timecourse plots

# Target EMUs for measurement (freeflux format: metabolite -> [carbon_string])
TARGET_EMUS = {
    "g6p": ["123456"],  # Full 6-carbon
    "f6p": ["123456"],  # Full 6-carbon
    "dhap": ["123"],  # Full 3-carbon
    "g3p": ["123"],  # Full 3-carbon
    "pg3": ["123"],  # Full 3-carbon (3pg in original -> pg3 in freeflux)
    "pyr": ["123"],  # Full 3-carbon
}

# Mapping from EMU ID to metabolite name for output
EMU_TO_METABOLITE = {
    "g6p_123456": "g6p",
    "f6p_123456": "f6p",
    "dhap_123": "dhap",
    "g3p_123": "g3p",
    "pg3_123": "3pg",
    "pyr_123": "pyr",
}

# Reference condition
REFERENCE_FBA1 = -3.0
REFERENCE_FBA2 = -2.0

# Test conditions: FBA1 varies from -1 to -4 kJ/mol, FBA2 = -5 - FBA1
# This ensures FBA1 + FBA2 = -5 kJ/mol while FBA1 is in [-1, -4] range
TEST_CONDITIONS = [
    {"FBA1": -1.0, "FBA2": -4.0},
    {"FBA1": -1.75, "FBA2": -3.25},
    {"FBA1": -2.5, "FBA2": -2.5},
    {"FBA1": -3.25, "FBA2": -1.75},
    {"FBA1": -4.0, "FBA2": -1.0},
]

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


def load_thermo_data(filepath, overrides=None):
    """Load thermodynamic data from TOML file with optional overrides."""
    data = toml.load(str(filepath))
    temperature = data.get("conditions", {}).get("temperature", 298.15)

    thermo = {}
    for rxn_id, rxn_data in data.get("reactions", {}).items():
        thermo[rxn_id] = rxn_data.get("deltaG_prime", -5.0)

    if overrides:
        thermo.update(overrides)

    return thermo, temperature


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


def load_fluxes_from_toml(flux_filepath):
    """Load net fluxes from TOML file and convert units."""
    data = toml.load(str(flux_filepath))
    fluxes = {}
    conversion_factor = 1000 / 3600

    for section in ["external_fluxes", "internal_fluxes"]:
        if section in data:
            for rxn_id, value in data[section].items():
                if rxn_id == "glc_uptake":
                    rxn_id = "glc_input"
                elif rxn_id == "pyr_secretion":
                    continue

                fluxes[rxn_id] = value * conversion_factor

    return fluxes


def run_simulation(gibbs_overrides, timepoints):
    """Run freeflux INST simulation with specified Gibbs energies."""
    model = Model("glycolysis_mech_gly3p")
    model.read_from_file(str(REACTIONS_FILE))

    net_fluxes = load_fluxes_from_toml(FLUXES_FILE)
    thermo, temperature = load_thermo_data(THERMO_FILE, overrides=gibbs_overrides)

    reversible_rxns = {
        fid.rsplit("_", 1)[0] for fid in model.totalfluxids if fid.endswith("_f")
    }

    with model.simulator("inst") as isim:
        isim.set_target_EMUs(TARGET_EMUS)
        isim.set_timepoints(timepoints)
        isim.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=["110001"],
            percentage=[1.0],
            purity=[1.0],
        )

        for rxn_id, v_net in net_fluxes.items():
            deltaG = thermo.get(rxn_id, -5.0)

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
            mdvs[emu_id] = {}
            for t in res.timepoints:
                if t > 0:
                    mdvs[emu_id][t] = np.array(mdv_result[t])

        return mdvs


def extract_mdvs_at_timepoints(simulation_mdvs, timepoints):
    """Restructure MDVs from {emu: {time: mdv}} to {time: {emu: mdv}}."""
    mdvs_by_time = {}

    for t in timepoints:
        mdvs_by_time[t] = {}
        for emu_id, emu_mdvs in simulation_mdvs.items():
            available_times = list(emu_mdvs.keys())
            if not available_times:
                continue
            closest_t = min(available_times, key=lambda x: abs(x - t))

            mdv = emu_mdvs[closest_t]
            mdv = np.maximum(mdv, 0.0)
            mdv_sum = np.sum(mdv)
            if mdv_sum > 0:
                mdv = mdv / mdv_sum
            mdvs_by_time[t][emu_id] = mdv

    return mdvs_by_time


def calculate_chisquared_at_timepoint(test_mdvs, reference_mdvs, scale):
    """Calculate chi-squared at a single timepoint using log-normal error.

    Uses log-normal distribution: sum((log(test) - log(ref))^2 / scale^2)
    """
    total_chi_squared = 0.0
    per_metabolite_chi_squared = {}
    variance = scale**2

    for emu_id in test_mdvs:
        if emu_id not in reference_mdvs:
            continue

        test_mdv = test_mdvs[emu_id]
        ref_mdv = reference_mdvs[emu_id]

        # Log-normal error
        log_ref = np.log(ref_mdv + EPSILON)
        log_test = np.log(test_mdv + EPSILON)
        metabolite_chi_squared = float(np.sum((log_test - log_ref) ** 2 / variance))

        met_name = EMU_TO_METABOLITE.get(emu_id, emu_id.split("_")[0])
        per_metabolite_chi_squared[met_name] = metabolite_chi_squared
        total_chi_squared += metabolite_chi_squared

    return total_chi_squared, per_metabolite_chi_squared


# ============================================================================
# Main Analysis
# ============================================================================


def main():
    """Run Gibbs energy timecourse chi-squared analysis."""
    print("=" * 80)
    print("Gibbs Energy Timecourse Chi-Squared Analysis (freeflux)")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()

    # Exclude final timepoint (1e3) from timecourse analysis
    filtered_timepoints = [t for t in TIMEPOINTS if t > 0 and abs(t - FINAL_TIMEPOINT) > 1e-6]
    print(
        f"\nTimepoints: {len(filtered_timepoints)} points from {filtered_timepoints[0]:.4f}s to {filtered_timepoints[-1]:.4f}s"
    )
    print(f"Measured metabolites: {list(TARGET_EMUS.keys())}")
    sys.stdout.flush()

    # Step 1: Generate reference observations
    print("\n" + "=" * 80)
    print("Generating Reference Observations")
    print("=" * 80)
    print(f"  FBA1={REFERENCE_FBA1:.2f}, FBA2={REFERENCE_FBA2:.2f} kJ/mol")
    sys.stdout.flush()

    reference_sim = run_simulation(
        gibbs_overrides={"FBA1": REFERENCE_FBA1, "FBA2": REFERENCE_FBA2},
        timepoints=filtered_timepoints,
    )

    reference_mdvs = extract_mdvs_at_timepoints(reference_sim, filtered_timepoints)
    print(f"  Extracted MDVs for {len(filtered_timepoints)} timepoints")
    sys.stdout.flush()

    # Step 2: Simulate test conditions and calculate chi-squared
    print("\n" + "=" * 80)
    print("Simulating Test Conditions")
    print("=" * 80)
    print(
        f"  {len(TEST_CONDITIONS)} conditions (FBA1 in [-4, -1] kJ/mol, FBA1 + FBA2 = -5.0 kJ/mol)"
    )
    sys.stdout.flush()

    results_data = []

    for idx, condition in enumerate(TEST_CONDITIONS, 1):
        fba1 = condition["FBA1"]
        fba2 = condition["FBA2"]

        print(f"\n  [{idx}/{len(TEST_CONDITIONS)}] FBA1={fba1:.2f}, FBA2={fba2:.2f} kJ/mol")
        sys.stdout.flush()

        test_sim = run_simulation(
            gibbs_overrides={"FBA1": fba1, "FBA2": fba2},
            timepoints=filtered_timepoints,
        )

        test_mdvs = extract_mdvs_at_timepoints(test_sim, filtered_timepoints)

        for timepoint in filtered_timepoints:
            total_chi_squared, per_metabolite = calculate_chisquared_at_timepoint(
                test_mdvs[timepoint],
                reference_mdvs[timepoint],
                LOGNORMAL_SCALE,
            )

            result_row = {
                "FBA1": fba1,
                "FBA2": fba2,
                "timepoint": timepoint,
                "chi_squared_total": total_chi_squared,
            }

            for met_id, chi_sq in per_metabolite.items():
                result_row[f"chi_squared_{met_id}"] = chi_sq

            results_data.append(result_row)

        print(f"    Calculated chi-squared at {len(filtered_timepoints)} timepoints")
        sys.stdout.flush()

    print("\n  All simulations complete")
    sys.stdout.flush()

    # Step 3: Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    sys.stdout.flush()

    df = pd.DataFrame(results_data)
    csv_path = output_dir / "timecourse_chisquared.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    sys.stdout.flush()

    # Step 4: Create aggregated chi-squared timecourse plot
    print("\n  Creating aggregated chi-squared plot...")
    sys.stdout.flush()

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_conditions = [
        c
        for c in TEST_CONDITIONS
        if not (c["FBA1"] == REFERENCE_FBA1 and c["FBA2"] == REFERENCE_FBA2)
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_conditions)))

    for idx, condition in enumerate(plot_conditions):
        fba1 = condition["FBA1"]
        fba2 = condition["FBA2"]

        condition_data = df[(df["FBA1"] == fba1) & (df["FBA2"] == fba2)]

        ax.plot(
            condition_data["timepoint"],
            condition_data["chi_squared_total"],
            linewidth=2,
            color=colors[idx],
            label=f"FBA1={fba1:.2f}, FBA2={fba2:.2f}",
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Chi-Squared", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()

    fig_path = output_dir / "timecourse_chisquared_aggregated.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"  Saved figure: {fig_path}")
    sys.stdout.flush()

    plt.close()

    # Step 5: Create per-metabolite chi-squared subplots
    print("\n  Creating per-metabolite chi-squared plots...")
    sys.stdout.flush()

    n_conditions = len(plot_conditions)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metabolite_colors = {
        "g6p": "C0",
        "f6p": "C1",
        "dhap": "C2",
        "g3p": "C3",
        "3pg": "C4",
        "pyr": "C5",
    }

    for idx, condition in enumerate(plot_conditions):
        ax = axes[idx]
        fba1 = condition["FBA1"]
        fba2 = condition["FBA2"]

        condition_data = df[(df["FBA1"] == fba1) & (df["FBA2"] == fba2)]

        for met_id in metabolite_colors:
            col_name = f"chi_squared_{met_id}"
            if col_name in condition_data.columns:
                ax.plot(
                    condition_data["timepoint"],
                    condition_data[col_name],
                    linewidth=1.5,
                    color=metabolite_colors[met_id],
                    label=met_id,
                )

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Chi-Squared", fontsize=10)
        ax.set_title(f"FBA1={fba1:.2f}, FBA2={fba2:.2f} kJ/mol", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for idx in range(n_conditions, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    fig_path = output_dir / "timecourse_chisquared_per_metabolite.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"  Saved figure: {fig_path}")
    sys.stdout.flush()

    plt.close()

    # Summary
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}")
    print("  - timecourse_chisquared.csv")
    print("  - timecourse_chisquared_aggregated.png")
    print("  - timecourse_chisquared_per_metabolite.png")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

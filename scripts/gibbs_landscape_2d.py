#!/usr/bin/env python3
"""2D Gibbs Energy Landscape Analysis - Negative Log-Likelihood.

This script scans a 2D grid of Gibbs energies for FBA1 and FBA2 reactions,
calculating negative log-likelihood against a reference condition.

Supports two modes:
- FFE (far-from-equilibrium): Total FBA dG = -5.09 kJ/mol (from glycolysis_thermodynamics.toml)
- NE (near-equilibrium): Total FBA dG = -0.87 kJ/mol (from glycolysis_thermodynamics_ne.toml)

Usage:
    uv run python scripts/gibbs_landscape_2d.py --mode ffe
    uv run python scripts/gibbs_landscape_2d.py --mode ne

Outputs:
- NLL over the full timecourse (integrated across all timepoints)
- NLL at the final timepoint only
"""

import argparse
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
REACTIONS_FILE = MODEL_DIR / "reactions.tsv"
FLUXES_FILE = MODEL_DIR / "glycolysis_fluxes.toml"

# Mode-specific configurations
MODE_CONFIGS = {
    "ffe": {
        "thermo_file": MODEL_DIR / "glycolysis_thermodynamics.toml",
        "results_dir": PROJECT_DIR / "results" / "gibbs_landscape_2d_ffe",
        "total_fba_dg": -5.09,  # FBA1 (-3.0) + FBA2 (-2.0) + margin = -5.09 approx
        "description": "Far-from-equilibrium (FFE)",
    },
    "ne": {
        "thermo_file": MODEL_DIR / "glycolysis_thermodynamics_ne.toml",
        "results_dir": PROJECT_DIR / "results" / "gibbs_landscape_2d_ne",
        "total_fba_dg": -0.87,
        "description": "Near-equilibrium (NE)",
    },
}

# Grid configuration: 10x10 grid with ±50% range around total FBA dG
N_GRID_POINTS = 20

# Gas constant in kJ/(mol*K)
R = 8.314e-3

# Timepoints: 100 points logarithmically spaced from 1e-3s to 1e1s (timecourse only)
# Steady-state is computed using freeflux SS mode, not long INST simulation
TIMEPOINTS = list(np.logspace(np.log10(1e-3), np.log10(1e1), 100))

# Target EMUs to compare
TARGET_EMUS = {
    "g6p": ["123456"],
    "f6p": ["123456"],
    "dhap": ["123"],
    "g3p": ["123"],
    "pg3": ["123"],
    "pyr": ["123"],
    "r5p": ["12345"],
}

# Log-normal error scale (standard deviation of log-transformed values)
LOGNORMAL_SCALE = 0.05

# Small epsilon to avoid log(0)
EPSILON = 1e-10


def get_grid_range(total_dg):
    """Compute FBA1/FBA2 grid range centered around the reference (total_dg/2).

    Args:
        total_dg: Total FBA Gibbs energy (negative value)

    Returns:
        Tuple of (min_dg, max_dg) centered at total_dg/2 with ±50% margin
    """
    # Reference is at total_dg/2, center grid around it with ±50% of reference
    ref = total_dg / 2
    margin = abs(ref) * 0.5
    dg_min = ref + margin  # less negative (closer to equilibrium)
    dg_max = ref - margin  # more negative (further from equilibrium)
    return dg_min, dg_max


def get_reference_condition(total_dg):
    """Get reference condition with equal split of total dG between FBA1 and FBA2.

    Args:
        total_dg: Total FBA Gibbs energy (negative value)

    Returns:
        Tuple of (FBA1_dg, FBA2_dg) each equal to total_dg / 2
    """
    # Equal split: each reaction gets half of total dG
    half_dg = total_dg / 2
    return half_dg, half_dg


# ============================================================================
# Helper Functions
# ============================================================================


def create_output_directory(results_dir):
    """Create output directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {results_dir}")
    sys.stdout.flush()
    return results_dir


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


def run_simulation(gibbs_overrides, timepoints, thermo_file):
    """Run INST simulation for timecourse and SS simulation for steady-state.

    Args:
        gibbs_overrides: Dict of reaction_id -> deltaG overrides
        timepoints: List of timepoints for INST simulation
        thermo_file: Path to thermodynamics TOML file

    Returns:
        Tuple of (mdvs_timecourse, mdvs_ss) where:
        - mdvs_timecourse: dict of {emu_id: {time: mdv}} for timecourse
        - mdvs_ss: dict of {emu_id: mdv} for steady-state
    """
    model = Model("glycolysis_mech_gly3p")
    model.read_from_file(str(REACTIONS_FILE))

    net_fluxes = load_fluxes_from_toml(FLUXES_FILE)
    thermo, temperature = load_thermo_data(thermo_file, overrides=gibbs_overrides)

    reversible_rxns = {
        fid.rsplit("_", 1)[0] for fid in model.totalfluxids if fid.endswith("_f")
    }

    # Build flux dict for reuse in both INST and SS simulations
    fluxes = {}
    for rxn_id, v_net in net_fluxes.items():
        deltaG = thermo.get(rxn_id, -5.0)

        if rxn_id in reversible_rxns:
            v_f, v_b = net_to_forward_backward(v_net, deltaG, temperature)
            fluxes[f"{rxn_id}_f"] = v_f
            fluxes[f"{rxn_id}_b"] = v_b
        else:
            fluxes[rxn_id] = v_net

    # 1. INST simulation for timecourse
    with model.simulator("inst") as isim:
        isim.set_target_EMUs(TARGET_EMUS)
        isim.set_timepoints(timepoints)
        isim.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=["110000"],
            percentage=[1.0],
            purity=[1.0],
        )

        for flux_id, value in fluxes.items():
            isim.set_flux(flux_id, value)

        for metab in model.metabolites:
            if metab == "FBA_dhap":
                isim.set_concentration(metab, 0.01)
            else:
                isim.set_concentration(metab, 1.0)

        isim.prepare(n_jobs=1)
        res = isim.simulate()

        mdvs_timecourse = {}
        for emu_id in res.simulated_EMUs:
            mdv_result = res.simulated_MDV(emu_id)
            mdvs_timecourse[emu_id] = {t: np.array(mdv_result[t]) for t in res.timepoints if t > 0}

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
    """Compute negative log-likelihood between two sets of MDVs using log-normal error.

    Uses jax.scipy.stats.norm.logpdf on log-transformed values.

    Args:
        mdvs_ref: Reference MDV values {emu_id: {timepoint: mdv_array}}
        mdvs_test: Test MDV values {emu_id: {timepoint: mdv_array}}
        scale: Log-normal scale parameter

    Returns:
        Total negative log-likelihood
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

            # Normalize MDVs
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
        Total negative log-likelihood at steady-state
    """
    total_nll = 0.0

    for emu_id in mdvs_ref_ss:
        if emu_id not in mdvs_test_ss:
            continue

        mdv_ref = mdvs_ref_ss[emu_id]
        mdv_test = mdvs_test_ss[emu_id]

        # Normalize MDVs
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


def run_2d_scan(dG1_values, dG2_values, timepoints, thermo_file, ref_fba1, ref_fba2):
    """Run 2D scan over FBA1 and FBA2 Gibbs energies.

    Compares each point against the reference condition.

    Args:
        dG1_values: Array of FBA1 Gibbs energy values
        dG2_values: Array of FBA2 Gibbs energy values
        timepoints: List of timepoints for simulation
        thermo_file: Path to thermodynamics TOML file
        ref_fba1: Reference FBA1 Gibbs energy
        ref_fba2: Reference FBA2 Gibbs energy

    Returns:
        DataFrame with columns: FBA1_deltaG, FBA2_deltaG, nll_timecourse, nll_final
        and dict with reference NLL values
    """
    print(f"Running reference simulation (FBA1={ref_fba1:.3f}, FBA2={ref_fba2:.3f})...")
    sys.stdout.flush()

    ref_overrides = {"FBA1": ref_fba1, "FBA2": ref_fba2}
    mdvs_ref_tc, mdvs_ref_ss = run_simulation(ref_overrides, timepoints, thermo_file)

    # Compute reference NLL (reference vs itself)
    print("Computing reference NLL (reference vs itself)...")
    sys.stdout.flush()
    ref_nll_timecourse = compute_nll(mdvs_ref_tc, mdvs_ref_tc, LOGNORMAL_SCALE)
    ref_nll_final = compute_nll_ss(mdvs_ref_ss, mdvs_ref_ss, LOGNORMAL_SCALE)
    ref_nll = {"timecourse": ref_nll_timecourse, "final": ref_nll_final}
    print(f"  Reference NLL - Timecourse: {ref_nll_timecourse:.2f}, SS: {ref_nll_final:.2f}")

    results = []
    total = len(dG1_values) * len(dG2_values)
    count = 0

    for dG1 in dG1_values:
        for dG2 in dG2_values:
            count += 1
            print(f"  [{count}/{total}] FBA1={dG1:.2f}, FBA2={dG2:.2f} kJ/mol")
            sys.stdout.flush()

            thermo_overrides = {"FBA1": dG1, "FBA2": dG2}

            try:
                mdvs_test_tc, mdvs_test_ss = run_simulation(thermo_overrides, timepoints, thermo_file)

                # Calculate NLL over full timecourse
                nll_timecourse = compute_nll(mdvs_ref_tc, mdvs_test_tc, LOGNORMAL_SCALE)

                # Calculate NLL at steady-state
                nll_final = compute_nll_ss(mdvs_ref_ss, mdvs_test_ss, LOGNORMAL_SCALE)

            except Exception as e:
                print(f"    Error: {e}")
                sys.stdout.flush()
                nll_timecourse = np.nan
                nll_final = np.nan

            results.append(
                {
                    "FBA1_deltaG": dG1,
                    "FBA2_deltaG": dG2,
                    "nll_timecourse": nll_timecourse,
                    "nll_final": nll_final,
                }
            )

    return pd.DataFrame(results), ref_nll


def plot_landscape(df, value_col, title_suffix, output_path, ref_fba1, ref_fba2):
    """Plot 2D NLL landscape as heatmap.

    Args:
        df: DataFrame with FBA1_deltaG, FBA2_deltaG, and value columns
        value_col: Column name for values to plot
        title_suffix: Suffix for plot title
        output_path: Path to save the plot
        ref_fba1: Reference FBA1 Gibbs energy
        ref_fba2: Reference FBA2 Gibbs energy
    """
    # Pivot for heatmap (FBA2 on x-axis, FBA1 on y-axis)
    pivot = df.pivot(index="FBA1_deltaG", columns="FBA2_deltaG", values=value_col)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
        cmap=cmc.batlow,
    )

    # Mark the reference point
    ax.plot(
        ref_fba2,
        ref_fba1,
        "o",
        color="red",
        markersize=10,
        label=f"Reference (FBA1={ref_fba1:.3f}, FBA2={ref_fba2:.3f})",
    )

    ax.set_xlabel("FBA2 dG (kJ/mol)", fontsize=12)
    ax.set_ylabel("FBA1 dG (kJ/mol)", fontsize=12)

    # Invert axes so more negative is at top-right
    ax.invert_xaxis()
    ax.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax, label="Negative Log-Likelihood")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    sys.stdout.flush()

    plt.close()


# ============================================================================
# Main Analysis
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="2D Gibbs Energy Landscape Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/gibbs_landscape_2d.py --mode ffe
  uv run python scripts/gibbs_landscape_2d.py --mode ne
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["ffe", "ne"],
        required=True,
        help="Analysis mode: ffe (far-from-equilibrium) or ne (near-equilibrium)",
    )
    return parser.parse_args()


def main():
    """Run 2D Gibbs energy landscape analysis."""
    args = parse_args()
    mode = args.mode

    # Get mode-specific configuration
    config = MODE_CONFIGS[mode]
    thermo_file = config["thermo_file"]
    results_dir = config["results_dir"]
    total_fba_dg = config["total_fba_dg"]
    description = config["description"]

    # Compute grid range (±50% of total FBA dG)
    dg_min, dg_max = get_grid_range(total_fba_dg)
    dG1_values = np.linspace(dg_min, dg_max, N_GRID_POINTS)
    dG2_values = np.linspace(dg_min, dg_max, N_GRID_POINTS)

    # Reference condition: equal split of total dG between FBA1 and FBA2
    ref_fba1, ref_fba2 = get_reference_condition(total_fba_dg)

    print("=" * 80)
    print(f"2D Gibbs Energy Landscape - {description}")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory(results_dir)

    # Filter timepoints
    filtered_timepoints = [t for t in TIMEPOINTS if t > 0]

    print(f"\nMode: {mode.upper()} ({description})")
    print(f"Thermodynamics file: {thermo_file}")
    print(f"\nGrid configuration:")
    print(f"  Total FBA dG: {total_fba_dg:.2f} kJ/mol")
    print(f"  Grid range: {dg_min:.3f} to {dg_max:.3f} kJ/mol (±50% of total)")
    print(f"  FBA1 range: {dG1_values.min():.3f} to {dG1_values.max():.3f} kJ/mol ({len(dG1_values)} points)")
    print(f"  FBA2 range: {dG2_values.min():.3f} to {dG2_values.max():.3f} kJ/mol ({len(dG2_values)} points)")
    print(f"  Reference: FBA1={ref_fba1:.3f}, FBA2={ref_fba2:.3f} kJ/mol (equal split)")
    print(f"  Total grid points: {len(dG1_values) * len(dG2_values)}")
    print(f"  Timepoints: {len(filtered_timepoints)} (from {filtered_timepoints[0]:.4f}s to {filtered_timepoints[-1]:.4f}s)")
    sys.stdout.flush()

    # Run 2D scan
    print("\n" + "=" * 80)
    print("Running 2D Scan")
    print("=" * 80)
    sys.stdout.flush()

    df, ref_nll = run_2d_scan(dG1_values, dG2_values, filtered_timepoints, thermo_file, ref_fba1, ref_fba2)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    sys.stdout.flush()

    csv_path = output_dir / "gibbs_landscape_2d.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    sys.stdout.flush()

    # Plot timecourse NLL landscape
    print("\n  Creating timecourse NLL landscape plot...")
    sys.stdout.flush()
    plot_landscape(
        df,
        "nll_timecourse",
        "(Full Timecourse)",
        output_dir / "gibbs_landscape_2d_timecourse.png",
        ref_fba1,
        ref_fba2,
    )

    # Plot steady-state NLL landscape
    print("  Creating steady-state NLL landscape plot...")
    sys.stdout.flush()
    plot_landscape(
        df,
        "nll_final",
        "(Steady-State)",
        output_dir / "gibbs_landscape_2d_final.png",
        ref_fba1,
        ref_fba2,
    )

    # Summary
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    print(f"\nMode: {mode.upper()} ({description})")
    print(f"Total FBA dG: {total_fba_dg:.2f} kJ/mol")
    print(f"Reference: FBA1={ref_fba1:.3f}, FBA2={ref_fba2:.3f} kJ/mol")

    print(f"\nReference NLL (reference vs itself):")
    print(f"  Timecourse: {ref_nll['timecourse']:.2f}")
    print(f"  Steady-State: {ref_nll['final']:.2f}")

    print("\nTimecourse NLL Summary:")
    min_idx = df["nll_timecourse"].idxmin()
    print(
        f"  Min: {df['nll_timecourse'].min():.2f} at "
        f"FBA1={df.loc[min_idx, 'FBA1_deltaG']:.3f}, FBA2={df.loc[min_idx, 'FBA2_deltaG']:.3f}"
    )
    print(f"  Max: {df['nll_timecourse'].max():.2f}")

    print("\nSteady-State NLL Summary:")
    min_idx = df["nll_final"].idxmin()
    print(
        f"  Min: {df['nll_final'].min():.2f} at "
        f"FBA1={df.loc[min_idx, 'FBA1_deltaG']:.3f}, FBA2={df.loc[min_idx, 'FBA2_deltaG']:.3f}"
    )
    print(f"  Max: {df['nll_final'].max():.2f}")

    print(f"\nGenerated files in: {output_dir}")
    print("  - gibbs_landscape_2d.csv")
    print("  - gibbs_landscape_2d_timecourse.png")
    print("  - gibbs_landscape_2d_final.png")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

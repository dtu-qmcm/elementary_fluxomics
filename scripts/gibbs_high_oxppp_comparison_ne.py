#!/usr/bin/env python3
"""High-oxPPP Flux Map: Mechanistic vs Aggregated Model (NE values).

Simulates the aggregated (single FBA) and mechanistic (FBA1 + FBA2) glycolysis
models on the high-oxPPP flux map (glycolysis_fluxes_high_oxppp.toml: PFK and
the oxPPP split 50/50, PFK = G6PDH2r_PGL = 74.5% of glucose uptake) with the NE
thermodynamics, and plots the MID differences between the two models for INST
and SS labeling with 1,2-13C glucose.

Reference: Aggregated model with FBA dG = -0.87 kJ/mol
Test:      Mechanistic model with a 20/80 split, FBA1 = -0.174, FBA2 = -0.696 kJ/mol

The NE deltaG values were set for the baseline flux map. As a safeguard for
flux maps that reverse a reaction, any reversible reaction whose net flux
direction disagrees with the sign of its deltaG has its deltaG sign flipped so
that deltaG = -sign(v_net) * |deltaG| (none do for the current map).
"""

from pathlib import Path
import sys

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from freeflux import Model

from gibbs_fba_split_comparison_ne import net_to_forward_backward
from style import apply_style, DOUBLE_COL, METABOLITE_COLORS

apply_style()

# ============================================================================
# Configuration
# ============================================================================

# Paths - relative to scripts folder
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MECH_MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_mech_gly3p"
AGG_MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_gly3p"
RESULTS_DIR = PROJECT_DIR / "results" / "gibbs_high_oxppp_ne"

FLUX_FILE = "glycolysis_fluxes_high_oxppp.toml"
THERMO_FILE = "glycolysis_thermodynamics_ne.toml"

# Total FBA Gibbs energy from aggregated model NE thermodynamics (kJ/mol)
TOTAL_FBA_DG = -0.87

# Fraction of the total FBA dG assigned to FBA1 (FBA2 takes the rest)
FBA1_FRACTION = 0.2

# mmol/gDCW/h -> umol/gDCW/s
CONVERSION_FACTOR = 1000 / 3600

# Enzyme-bound intermediates (FBA_dhap, and the transketolase/transaldolase
# EC2/EC3 carriers) are given a small pool rather than the bulk default.
ENZYME_BOUND_METABOLITES = ("FBA_dhap", "EC2", "EC3")

# Tracer: 1,2-13C glucose (carbons 1 and 2 labeled)
LABELING_PATTERN = "110000"

# Timepoints: 100 points logarithmically spaced from 1e-3s to 1e1s
TIMEPOINTS = list(np.logspace(-3, 1, 100))

# Target EMUs to compare
TARGET_EMUS = {
    "g6p": ["123456"],
    "f6p": ["123456"],
    "fdp": ["123456"],
    "dhap": ["123"],
    "g3p": ["123"],
    "r5p": ["12345"],
}

# EMU ID to metabolite name mapping
EMU_TO_METABOLITE = {
    "g6p_123456": "g6p",
    "f6p_123456": "f6p",
    "fdp_123456": "fdp",
    "dhap_123": "dhap",
    "g3p_123": "g3p",
    "r5p_12345": "r5p",
}

# Measured metabolites in pathway order (plot and summary order)
METABOLITES = list(EMU_TO_METABOLITE.values())

# Isotopologues of the largest target EMU (g6p/f6p/fdp: M+0..M+6)
MAX_ISOTOPOLOGUES = 7


# ============================================================================
# Helper Functions
# ============================================================================


def create_output_directory():
    """Create output directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def pool_size(metab):
    """Pool size used for a metabolite in INST simulations."""
    return 0.01 if metab in ENZYME_BOUND_METABOLITES else 1.0


def build_fluxes(model, model_dir, thermo_overrides=None, flux_file=FLUX_FILE):
    """Build the total (forward/backward) flux dict from flux_file and THERMO_FILE.

    Returns:
        Tuple of (fluxes, flipped) where:
        - fluxes: dict of {flux_id: value} in umol/gCDW/s, with reversible
          reactions split into _f/_b via the Gibbs energy
        - flipped: reactions whose deltaG sign was flipped to match the net flux
    """
    flux_data = toml.load(str(model_dir / flux_file))
    thermo_data = toml.load(str(model_dir / THERMO_FILE))
    temperature = thermo_data.get("conditions", {}).get("temperature", 298.15)

    if thermo_overrides:
        for rxn_id, dG in thermo_overrides.items():
            thermo_data.setdefault("reactions", {}).setdefault(rxn_id, {})
            thermo_data["reactions"][rxn_id]["deltaG_prime"] = dG

    reversible_rxns = {
        fid.rsplit("_", 1)[0] for fid in model.totalfluxids if fid.endswith("_f")
    }

    fluxes = {}
    flipped = []
    for section in ["external_fluxes", "internal_fluxes"]:
        for rxn_id, value in flux_data.get(section, {}).items():
            if rxn_id == "glc_uptake":
                rxn_id = "glc_input"
            elif rxn_id == "pyr_secretion":
                continue

            v_net = value * CONVERSION_FACTOR
            deltaG = (
                thermo_data.get("reactions", {})
                .get(rxn_id, {})
                .get("deltaG_prime", -5.0)
            )

            if rxn_id in reversible_rxns:
                # Thermodynamic consistency: deltaG must oppose the net flux
                if v_net != 0 and np.sign(deltaG) == np.sign(v_net):
                    deltaG = -deltaG
                    flipped.append(rxn_id)
                v_f, v_b = net_to_forward_backward(v_net, deltaG, temperature)
                fluxes[f"{rxn_id}_f"] = v_f
                fluxes[f"{rxn_id}_b"] = v_b
            else:
                fluxes[rxn_id] = v_net

    return fluxes, flipped


def run_simulation(
    model_dir, model_name, thermo_overrides=None, timepoints=TIMEPOINTS, flux_file=FLUX_FILE
):
    """Run INST simulation for timecourse and SS simulation for steady-state.

    Returns:
        Tuple of (mdvs_timecourse, mdvs_ss, flipped) where:
        - mdvs_timecourse: dict of {emu_id: {time: mdv}} for timecourse
        - mdvs_ss: dict of {emu_id: mdv} for steady-state
        - flipped: reactions whose deltaG sign was flipped to match the net flux
    """
    model = Model(model_name)
    model.read_from_file(str(model_dir / "reactions.tsv"))

    fluxes, flipped = build_fluxes(model, model_dir, thermo_overrides, flux_file)

    # 1. INST simulation for timecourse
    with model.simulator("inst") as isim:
        isim.set_target_EMUs(TARGET_EMUS)
        isim.set_timepoints(list(timepoints))
        isim.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=[LABELING_PATTERN],
            percentage=[1.0],
            purity=[1.0],
        )

        for flux_id, value in fluxes.items():
            isim.set_flux(flux_id, value)

        for metab in model.metabolites:
            isim.set_concentration(metab, pool_size(metab))

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
            labeling_pattern=[LABELING_PATTERN],
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

    return mdvs_timecourse, mdvs_ss, flipped


def normalize_mdv(mdv):
    """Clip negative values and renormalize an MDV to sum to 1."""
    mdv = np.maximum(mdv, 0.0)
    total = np.sum(mdv)
    return mdv / total if total > 0 else mdv


def compute_mid_differences(mdvs_agg_tc, mdvs_mech_tc, mdvs_agg_ss, mdvs_mech_ss):
    """Signed per-isotopologue differences, MID_mech - MID_agg.

    Returns:
        Tuple of (df_ss, df_inst) long-format DataFrames with columns
        metabolite, [timepoint], isotopologue, mid_agg, mid_mech, diff.
    """
    rows_ss = []
    rows_inst = []

    for emu_id, met in EMU_TO_METABOLITE.items():
        mid_agg = normalize_mdv(mdvs_agg_ss[emu_id])
        mid_mech = normalize_mdv(mdvs_mech_ss[emu_id])
        for k, (a, m) in enumerate(zip(mid_agg, mid_mech)):
            rows_ss.append({
                "metabolite": met,
                "isotopologue": k,
                "mid_agg": a,
                "mid_mech": m,
                "diff": m - a,
            })

        for t in sorted(mdvs_agg_tc[emu_id]):
            if t <= 0 or t not in mdvs_mech_tc[emu_id]:
                continue
            mid_agg = normalize_mdv(mdvs_agg_tc[emu_id][t])
            mid_mech = normalize_mdv(mdvs_mech_tc[emu_id][t])
            for k, (a, m) in enumerate(zip(mid_agg, mid_mech)):
                rows_inst.append({
                    "metabolite": met,
                    "timepoint": t,
                    "isotopologue": k,
                    "mid_agg": a,
                    "mid_mech": m,
                    "diff": m - a,
                })

    return pd.DataFrame(rows_ss), pd.DataFrame(rows_inst)


def max_abs_diff_over_time(df_inst):
    """Max |MID_mech - MID_agg| across isotopologues, per metabolite and timepoint."""
    return (
        df_inst.assign(abs_diff=df_inst["diff"].abs())
        .groupby(["metabolite", "timepoint"], as_index=False)["abs_diff"]
        .max()
    )


def plot_differences(df_ss, df_inst, output_stem):
    """Two panels: SS signed differences per isotopologue, INST max |diff| vs time."""
    fig, (ax_ss, ax_inst) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))

    # (a) Steady state: grouped bars, one bar per metabolite at each isotopologue
    width = 0.8 / len(METABOLITES)
    for i, met in enumerate(METABOLITES):
        data = df_ss[df_ss["metabolite"] == met]
        x = data["isotopologue"] + (i - (len(METABOLITES) - 1) / 2) * width
        ax_ss.bar(x, data["diff"], width=width, color=METABOLITE_COLORS[met], label=met)

    ax_ss.axhline(0.0, color="black", linewidth=0.6)
    ax_ss.set_xticks(range(MAX_ISOTOPOLOGUES))
    ax_ss.set_xticklabels([f"M+{k}" for k in range(MAX_ISOTOPOLOGUES)])
    ax_ss.set_xlabel("Isotopologue")
    ax_ss.set_ylabel(r"$MID_\mathrm{mech} - MID_\mathrm{agg}$")
    ax_ss.set_title("Steady state")

    # (b) Instationary: largest isotopologue difference over time. fdp's teal
    # sits close to g6p/f6p, so it is also dashed.
    df_max = max_abs_diff_over_time(df_inst)
    for met in METABOLITES:
        data = df_max[df_max["metabolite"] == met]
        ax_inst.plot(
            data["timepoint"],
            data["abs_diff"],
            color=METABOLITE_COLORS[met],
            linestyle="--" if met == "fdp" else "-",
            label=met,
        )

    ax_inst.set_xscale("log")
    ax_inst.set_xlabel("Time (s)")
    ax_inst.set_ylabel(r"max $|MID_\mathrm{mech} - MID_\mathrm{agg}|$")
    ax_inst.set_title("Instationary")
    ax_inst.legend(loc="best")

    for ax in (ax_ss, ax_inst):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_stem.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    sys.stdout.flush()
    plt.close(fig)


def plot_inst_isotopologues(df_inst, output_stem):
    """One panel per metabolite: signed difference of each isotopologue over time."""
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.6), sharex=True)
    axes = axes.flatten()

    colors = [cmc.batlow(k / (MAX_ISOTOPOLOGUES - 1)) for k in range(MAX_ISOTOPOLOGUES)]

    for ax, met in zip(axes, METABOLITES):
        data = df_inst[df_inst["metabolite"] == met]
        for k in sorted(data["isotopologue"].unique()):
            iso = data[data["isotopologue"] == k]
            ax.plot(iso["timepoint"], iso["diff"], color=colors[k], label=f"M+{k}")

        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xscale("log")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_title(met)

    for ax in axes[3:]:
        ax.set_xlabel("Time (s)")
    for ax in axes[::3]:
        ax.set_ylabel(r"$MID_\mathrm{mech} - MID_\mathrm{agg}$")

    # The g6p panel carries every isotopologue, so its handles cover all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=MAX_ISOTOPOLOGUES)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        path = output_stem.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    sys.stdout.flush()
    plt.close(fig)


def main():
    """Run the high-oxPPP mechanistic vs aggregated comparison (NE thermodynamics)."""
    print("=" * 80)
    print("High-oxPPP Flux Map: Mechanistic vs Aggregated (NE Thermodynamics)")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()

    fba1_dG = FBA1_FRACTION * TOTAL_FBA_DG
    fba2_dG = (1 - FBA1_FRACTION) * TOTAL_FBA_DG

    print(f"\nConfiguration:")
    print(f"  Flux map: {FLUX_FILE}")
    print(f"  Thermodynamics: {THERMO_FILE}")
    print(f"  Aggregated:  FBA = {TOTAL_FBA_DG:.3f} kJ/mol")
    print(f"  Mechanistic: FBA1 = {fba1_dG:.3f}, FBA2 = {fba2_dG:.3f} kJ/mol")
    sys.stdout.flush()

    print("\nRunning aggregated (non-mechanistic) model...")
    sys.stdout.flush()
    mdvs_agg_tc, mdvs_agg_ss, flipped_agg = run_simulation(
        AGG_MODEL_DIR, "glycolysis_gly3p", {"FBA": TOTAL_FBA_DG}
    )
    print(f"  deltaG sign flipped to match net flux: {flipped_agg}")

    print("Running mechanistic model...")
    sys.stdout.flush()
    mdvs_mech_tc, mdvs_mech_ss, flipped_mech = run_simulation(
        MECH_MODEL_DIR, "glycolysis_mech_gly3p", {"FBA1": fba1_dG, "FBA2": fba2_dG}
    )
    print(f"  deltaG sign flipped to match net flux: {flipped_mech}")
    sys.stdout.flush()

    df_ss, df_inst = compute_mid_differences(
        mdvs_agg_tc, mdvs_mech_tc, mdvs_agg_ss, mdvs_mech_ss
    )

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    ss_csv_path = output_dir / "mid_differences_ss.csv"
    df_ss.to_csv(ss_csv_path, index=False)
    print(f"  Saved CSV: {ss_csv_path}")

    inst_csv_path = output_dir / "mid_differences_inst.csv"
    df_inst.to_csv(inst_csv_path, index=False)
    print(f"  Saved CSV: {inst_csv_path}")
    sys.stdout.flush()

    print("\n  Creating SS/INST difference plot...")
    plot_differences(df_ss, df_inst, output_dir / "mech_vs_agg_differences")

    print("  Creating per-isotopologue INST difference plot...")
    plot_inst_isotopologues(df_inst, output_dir / "inst_isotopologue_differences")

    # Summary
    df_max = max_abs_diff_over_time(df_inst)
    t_final = df_max["timepoint"].max()
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nMax |MID_mech - MID_agg| per metabolite:")
    print(f"  {'metabolite':<12}{'SS':>10}{'INST (max)':>14}{'t at max (s)':>14}{f'INST t={t_final:g}s':>16}")
    for met in METABOLITES:
        ss = df_ss.loc[df_ss["metabolite"] == met, "diff"].abs().max()
        inst = df_max[df_max["metabolite"] == met]
        i_max = inst["abs_diff"].idxmax()
        final = inst.loc[inst["timepoint"] == t_final, "abs_diff"].iloc[0]
        print(
            f"  {met:<12}{ss:>10.2e}{inst.loc[i_max, 'abs_diff']:>14.2e}"
            f"{inst.loc[i_max, 'timepoint']:>14.3g}{final:>16.2e}"
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()

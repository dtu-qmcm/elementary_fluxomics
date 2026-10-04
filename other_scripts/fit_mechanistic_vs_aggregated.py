#!/usr/bin/env python3
"""Fit aggregated and mechanistic models to mechanistic-model steady-state MIDs.

Generate synthetic steady-state MDVs from the mechanistic model (FBA1/FBA2),
then use freeflux's SS fitter to estimate fluxes with BOTH the aggregated model
(single lumped FBA) and the mechanistic model itself (self-fit). Tests whether
each model recovers fluxes when data comes from the mechanistic model, across
different FBA energy partitions and thermodynamic conditions (FFE = far from
equilibrium, NE = near equilibrium). Results carry a `model` column; the
aggregated flux CSVs keep their legacy names, the mechanistic ones get a
`_mech` suffix.
"""

import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from freeflux import Model

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MECH_MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_mech_gly3p"
STD_MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_gly3p"
RESULTS_DIR = PROJECT_DIR / "results" / "other_scripts" / "fit_mech_vs_agg"

R = 8.314e-3  # Gas constant in kJ/(mol*K)
CONVERSION_FACTOR = 1000 / 3600  # mmol/gDCW/h -> umol/gCDW/s

# Target EMUs to measure (common to both models)
TARGET_EMUS = {
    "g6p": ["123456"],
    "f6p": ["123456"],
    "dhap": ["123"],
    "g3p": ["123"],
    "pyr": ["123"],
    "lac": ["123"],
    "pgc6": ["123456"],
    "r5p": ["12345"],
}

# Boundary fluxes from mechanistic model (mmol/gDCW/h)
BOUNDARY_FLUXES = {
    "glc_input": 100.0043,
    "pyr_drain": 87.5243,
    "GLY3PDH": 5.0,
    "LDH": 100.0,
    "r5p_drain": 1.0,
}

# Thermodynamic conditions
CONDITIONS = {
    "FFE": {
        "thermo_file": "glycolysis_thermodynamics.toml",
        "total_fba_dG": -5.0,
    },
    "NE": {
        "thermo_file": "glycolysis_thermodynamics_ne.toml",
        "total_fba_dG": -0.87,
    },
}

# Alpha values: fraction of total dG assigned to FBA1
ALPHA_VALUES = np.linspace(0.1, 0.9, 7)

# Number of random restarts for the fitter (per simulation mode)
N_STARTS = 10  # steady-state fits
N_STARTS_INST = 10  # instationary fits

# Maximum solver iterations per restart (INST needs more: harder convergence)
MAX_ITERS_SS = 400
MAX_ITERS_INST = 3000

# Measurement standard deviation for MDVs and boundary fluxes
MEAS_SD = 0.01

# Instationary configuration: 5 log-spaced timepoints (seconds) over 1e-3..1e1 s.
INST_TIMEPOINTS = np.logspace(-3, 1, 5)

# Metabolite pool size for INST: data is generated at conc = 1.0 for all
# metabolites; the INST fit fixes concentrations there via a tight bracket
# (freeflux requires lower < upper) so only fluxes are estimated.
CONC_VALUE = 1.0
CONC_BOUNDS = [CONC_VALUE - 1e-3, CONC_VALUE + 1e-3]

# Initial flux guess for the aggregated model (net fluxes in mmol/gDCW/h,
# adapted from mechanistic model with TPI = FBA - GLY3PDH = 88.52)
# Reaction display order (glycolysis then PPP)
REACTION_ORDER = [
    "glc_input",
    "GLUT2",
    "HEX",
    "GPI",
    "PFK",
    "FBA",
    "GLY3PDH",
    "TPI",
    "GAPD_PGK",
    "PGM",
    "PYK",
    "pyr_drain",
    "LDH",
    "G6PDH2r_PGL",
    "GND",
    "RPI",
    "RPE",
    "r5p_drain",
    "X5P_EC2",
    "F6P_EC2",
    "S7P_EC2",
    "F6P_EC3",
    "S7P_EC3",
]

# True net fluxes for the aggregated model (mmol/gDCW/h). Must stay in sync with
# models/glycolysis_gly3p/glycolysis_fluxes.toml, where every metabolite balance
# closes exactly. Adapted from mechanistic model: FBA1=FBA2=FBA,
# TPI = FBA - GLY3PDH = 88.52.
# Net fluxes are constant across alpha (alpha only changes forward/backward split).
INITIAL_FLUXES_MMOL = {
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

# Half reactions whose net flux is negative as written, so their net flux bounds
# must allow negative values (freeflux applies set_flux_bounds to NET fluxes).
NEGATIVE_NET_RXNS = ("F6P_EC2", "S7P_EC2", "F6P_EC3")

TRUE_FLUXES = {k: v * CONVERSION_FACTOR for k, v in INITIAL_FLUXES_MMOL.items()}

# Initial net fluxes for the mechanistic model: identical to the aggregated
# model except the lumped FBA is split into FBA1 and FBA2 (both carry the same
# net flux as FBA, since FBA1 -> FBA2 are sequential with no branch between).
INITIAL_FLUXES_MMOL_MECH = {
    **{k: v for k, v in INITIAL_FLUXES_MMOL.items() if k != "FBA"},
    "FBA1": INITIAL_FLUXES_MMOL["FBA"],
    "FBA2": INITIAL_FLUXES_MMOL["FBA"],
}

# Model registry: display name -> (freeflux model id, model dir, initial fluxes)
FIT_MODELS = {
    "aggregated": ("glycolysis_gly3p", STD_MODEL_DIR, INITIAL_FLUXES_MMOL),
    "mechanistic": (
        "glycolysis_mech_gly3p",
        MECH_MODEL_DIR,
        INITIAL_FLUXES_MMOL_MECH,
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================


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


def _build_mech_model_and_fluxes(thermo_file, thermo_overrides=None):
    """Build the mechanistic model and its forward/backward flux dict.

    Shared by the SS and INST simulators. Reads net fluxes and thermodynamics
    from the mechanistic model TOMLs (applying any deltaG overrides) and converts
    reversible reactions to forward/backward fluxes via the Gibbs energy.

    Args:
        thermo_file: Name of thermodynamics TOML file (in MECH_MODEL_DIR).
        thermo_overrides: Dict of {reaction_id: deltaG} overrides.

    Returns:
        (model, fluxes) where fluxes is {flux_id: value} in umol/gCDW/s.
    """
    model = Model("glycolysis_mech_gly3p")
    model.read_from_file(str(MECH_MODEL_DIR / "reactions.tsv"))

    flux_data = toml.load(str(MECH_MODEL_DIR / "glycolysis_fluxes.toml"))
    thermo_data = toml.load(str(MECH_MODEL_DIR / thermo_file))
    temperature = thermo_data.get("conditions", {}).get("temperature", 298.15)

    if thermo_overrides:
        for rxn_id, dG in thermo_overrides.items():
            if "reactions" not in thermo_data:
                thermo_data["reactions"] = {}
            if rxn_id not in thermo_data["reactions"]:
                thermo_data["reactions"][rxn_id] = {}
            thermo_data["reactions"][rxn_id]["deltaG_prime"] = dG

    reversible_rxns = {
        fid.rsplit("_", 1)[0] for fid in model.totalfluxids if fid.endswith("_f")
    }

    fluxes = {}
    for section in ["external_fluxes", "internal_fluxes"]:
        if section in flux_data:
            for rxn_id, value in flux_data[section].items():
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
                    v_f, v_b = net_to_forward_backward(v_net, deltaG, temperature)
                    fluxes[f"{rxn_id}_f"] = v_f
                    fluxes[f"{rxn_id}_b"] = v_b
                else:
                    fluxes[rxn_id] = v_net

    return model, fluxes


def run_ss_simulation(thermo_file, thermo_overrides=None):
    """Run a steady-state simulation with the mechanistic model.

    Returns:
        dict of {emu_id: np.array(mdv)} for steady-state MDVs.
    """
    model, fluxes = _build_mech_model_and_fluxes(thermo_file, thermo_overrides)

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
        res = ssim.simulate()

        mdvs = {}
        for emu_id in res.simulated_EMUs:
            mdvs[emu_id] = np.array(res.simulated_MDV(emu_id))

    return mdvs


def run_inst_simulation(thermo_file, thermo_overrides=None):
    """Run an instationary simulation with the mechanistic model.

    Uses INST_TIMEPOINTS and fixes all metabolite concentrations at CONC_VALUE.

    Returns:
        dict of {emu_id: {timepoint: np.array(mdv)}} for t > 0.
    """
    model, fluxes = _build_mech_model_and_fluxes(thermo_file, thermo_overrides)

    with model.simulator("inst") as isim:
        isim.set_target_EMUs(TARGET_EMUS)
        isim.set_timepoints(list(INST_TIMEPOINTS))
        isim.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=["110000"],
            percentage=[1.0],
            purity=[1.0],
        )

        for flux_id, value in fluxes.items():
            isim.set_flux(flux_id, value)

        for metab in model.metabolites:
            isim.set_concentration(metab, CONC_VALUE)

        isim.prepare(n_jobs=1)
        res = isim.simulate()

        mdvs = {}
        for emu_id in res.simulated_EMUs:
            mdv_tcourse = res.simulated_MDV(emu_id)
            mdvs[emu_id] = {
                t: np.array(mdv_tcourse[t]) for t in res.timepoints if t > 0
            }

    return mdvs


def _write_ini_fluxes_tsv(flux_dict_mmol):
    """Write initial flux guesses to a temp TSV file (freeflux ini_fluxes format).

    Args:
        flux_dict_mmol: dict of {reaction_id: net_flux_mmol_per_gDCW_per_h}.

    Returns:
        Path to the temp TSV file (caller must delete after use).
    """
    fd, path = tempfile.mkstemp(suffix=".tsv")
    with os.fdopen(fd, "w") as f:
        f.write("#reaction_ID\tflux\n")
        for rxn_id, value in flux_dict_mmol.items():
            f.write(f"{rxn_id}\t{value * CONVERSION_FACTOR}\n")
    return path


def _write_ini_concs_tsv(conc_dict):
    """Write initial concentration guesses to a temp TSV (freeflux ini_concs).

    freeflux's read_initial_values calls os.path.isfile first, so a Series is
    rejected; a file path is required.

    Args:
        conc_dict: dict of {metabolite_id: concentration_umol_per_gCDW}.

    Returns:
        Path to the temp TSV file (caller must delete after use).
    """
    fd, path = tempfile.mkstemp(suffix=".tsv")
    with os.fdopen(fd, "w") as f:
        f.write("#metabolite_ID\tvalue\n")
        for metab_id, value in conc_dict.items():
            f.write(f"{metab_id}\t{value}\n")
    return path


def fit_model(model_name, model_dir, initial_fluxes_mmol, measured_data, sim_mode,
              n_starts, max_iters=400):
    """Fit a model to measured MDVs using freeflux's SS or INST fitter.

    Works for either the aggregated (single FBA) or mechanistic (FBA1/FBA2)
    model, in steady-state ("ss") or instationary ("inst") mode. Uses multiple
    random restarts to avoid local minima: the first start uses the model's
    nominal initial guess, subsequent starts perturb it.

    For INST, metabolite concentrations are fixed at CONC_VALUE via tight
    bounds (CONC_BOUNDS) and ini_concs, so only fluxes are estimated.

    Args:
        model_name: freeflux model id (e.g. "glycolysis_gly3p").
        model_dir: directory containing the model's reactions.tsv.
        initial_fluxes_mmol: dict of {reaction_id: net_flux_mmol_per_gDCW_per_h}.
        measured_data: SS -> {emu_id: np.array(mdv)};
                       INST -> {emu_id: {timepoint: np.array(mdv)}}.
        sim_mode: "ss" or "inst".
        n_starts: number of random restarts.

    Returns:
        dict with SSR, chi2_pass, optimization_successful, fitted_fluxes.
    """
    model = Model(model_name)
    model.read_from_file(str(model_dir / "reactions.tsv"))

    best_ssr = np.inf
    best_data = None

    fitter_kind = "inst" if sim_mode == "inst" else "ss"
    with model.fitter(fitter_kind) as fitter:
        fitter.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=["110000"],
            percentage=[1.0],
            purity=[1.0],
        )

        # Set measured MDVs from mechanistic simulation
        if sim_mode == "inst":
            for emu_id, tcourse in measured_data.items():
                tps = sorted(tcourse.keys())
                means = [list(tcourse[t]) for t in tps]
                sds = [
                    list(np.full_like(np.asarray(tcourse[t], dtype=float), MEAS_SD))
                    for t in tps
                ]
                fitter.set_measured_MDVs(emu_id, tps, means, sds)
            fitter.set_timepoints(list(INST_TIMEPOINTS))
            # Pin concentrations at CONC_VALUE via tight bounds + ini_concs (set
            # below). Do NOT call set_concentration here: the optimizer rebuilds
            # model.concentrations from concids, and a longer preset Series would
            # break its `concentrations[:] = c` assignment.
            fitter.set_concentration_bounds("all", CONC_BOUNDS)
        else:
            for emu_id, mdv in measured_data.items():
                sd = np.full_like(np.asarray(mdv, dtype=float), MEAS_SD)
                fitter.set_measured_MDV(emu_id, list(mdv), list(sd))

        # Set measured boundary fluxes (converted to umol/gCDW/s)
        for flux_id, value_mmol in BOUNDARY_FLUXES.items():
            value = value_mmol * CONVERSION_FACTOR
            fitter.set_measured_flux(flux_id, value, MEAS_SD)

        fitter.set_flux_bounds("all", [0, 200])
        # The EC2/EC3 half reactions listed here run in reverse as written,
        # so [0, 200] would make the fit infeasible. Later calls override
        # earlier ones, so this only relaxes these three reactions.
        for _rxn_id in NEGATIVE_NET_RXNS:
            fitter.set_flux_bounds(_rxn_id, [-200, 0])

        fitter.prepare(n_jobs=1)

        # Fixed concentration initial guess for INST (concentrations pinned at
        # CONC_VALUE for every metabolite; written to a temp TSV file).
        ini_concs_path = None
        if sim_mode == "inst":
            ini_concs_path = _write_ini_concs_tsv(
                {m: CONC_VALUE for m in model.concids}
            )

        for start in range(n_starts):
            if start == 0:
                ini_dict = initial_fluxes_mmol
            else:
                ini_dict = {
                    k: v * np.random.uniform(0.5, 1.5)
                    for k, v in initial_fluxes_mmol.items()
                }

            ini_path = _write_ini_fluxes_tsv(ini_dict)
            try:
                solve_kwargs = dict(
                    solver="slsqp",
                    ini_fluxes=ini_path,
                    tol=1e-6,
                    max_iters=max_iters,
                    show_progress=False,
                )
                if sim_mode == "inst":
                    solve_kwargs["ini_concs"] = ini_concs_path
                with contextlib.redirect_stdout(io.StringIO()):
                    results = fitter.solve(**solve_kwargs)
            except Exception:
                continue
            finally:
                os.unlink(ini_path)

            if results.opt_objective < best_ssr:
                best_ssr = results.opt_objective
                try:
                    fitted_fluxes = {
                        k: float(v) for k, v in results.opt_net_fluxes.items()
                    }
                except Exception:
                    fitted_fluxes = {}
                try:
                    total = results.opt_total_fluxes
                    fitted_reverse_fluxes = {
                        k.removesuffix("_b"): float(total[k])
                        for k in total.keys()
                        if k.endswith("_b")
                    }
                except Exception:
                    fitted_reverse_fluxes = {}
                best_data = {
                    "SSR": float(results.opt_objective),
                    "optimization_successful": bool(results.optimization_successful),
                    "fitted_fluxes": fitted_fluxes,
                    "fitted_reverse_fluxes": fitted_reverse_fluxes,
                }

        if ini_concs_path is not None:
            os.unlink(ini_concs_path)

        # Chi2 test on best result (suppress stdout)
        if best_data is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    chi2_pass = results.chi2_test()
                best_data["chi2_pass"] = chi2_pass
            except Exception:
                best_data["chi2_pass"] = None

    if best_data is None:
        best_data = {
            "SSR": np.nan,
            "chi2_pass": None,
            "optimization_successful": False,
            "fitted_fluxes": {},
            "fitted_reverse_fluxes": {},
        }

    return best_data


# ============================================================================
# Plotting
# ============================================================================


def plot_ssr_vs_partition(df, output_dir):
    """Plot SSR vs alpha, faceted by simulation mode (SS | INST).

    Within each panel, overlay both conditions (color) and both models
    (solid/circles = aggregated, dashed/squares = mechanistic).
    """
    if "model" not in df.columns:
        df = df.assign(model="aggregated")
    if "sim_mode" not in df.columns:
        df = df.assign(sim_mode="ss")

    sim_modes = [m for m in ["ss", "inst"] if (df["sim_mode"] == m).any()]
    if not sim_modes:
        sim_modes = ["ss"]

    fig, axes = plt.subplots(
        1, len(sim_modes), figsize=(7 * len(sim_modes), 6), squeeze=False
    )
    axes = axes[0]

    colors = {"FFE": cmc.batlow(0.3), "NE": cmc.batlow(0.7)}
    # Solid + circles for aggregated, dashed + squares for mechanistic.
    styles = {
        "aggregated": {"ls": "-", "marker": "o"},
        "mechanistic": {"ls": "--", "marker": "s"},
    }

    for ax, sim_mode in zip(axes, sim_modes):
        for cond_name in ["FFE", "NE"]:
            for model_label in ["aggregated", "mechanistic"]:
                cond_data = df[
                    (df["condition"] == cond_name)
                    & (df["model"] == model_label)
                    & (df["sim_mode"] == sim_mode)
                ].sort_values("alpha")
                if not len(cond_data):
                    continue
                style = styles.get(model_label, {"ls": "-", "marker": "o"})
                ax.plot(
                    cond_data["alpha"],
                    cond_data["SSR"],
                    linestyle=style["ls"],
                    marker=style["marker"],
                    color=colors[cond_name],
                    linewidth=2,
                    markersize=6,
                    label=f"{cond_name} ({model_label})",
                )

        ax.axvline(
            x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5,
            label=r"$\alpha$ = 0.5",
        )
        ax.set_xlabel(
            r"$\alpha$ (fraction of total $\Delta G$ assigned to FBA1)", fontsize=12
        )
        ax.set_ylabel("Optimal SSR", fontsize=12)
        ax.set_title(sim_mode.upper(), fontsize=14)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = output_dir / f"fit_ssr_vs_partition.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
        sys.stdout.flush()
    plt.close()


def plot_flux_error_heatmaps(df_fluxes, output_dir, sim_mode="ss"):
    """Plot heatmaps of % flux error (fitted vs true) for FFE and NE conditions.

    X-axis: alpha, Y-axis: reaction, color: % difference. Filtered to one
    simulation mode; output filename carries the mode suffix.
    """
    if "sim_mode" in df_fluxes.columns:
        df_fluxes = df_fluxes[df_fluxes["sim_mode"] == sim_mode]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, cond_name in zip(axes, ["FFE", "NE"]):
        cond_data = df_fluxes[df_fluxes["condition"] == cond_name].sort_values("alpha")
        alphas = cond_data["alpha"].values

        # Build % difference matrix (reactions x alphas)
        pct_diff = np.full((len(REACTION_ORDER), len(alphas)), np.nan)
        for j, (_, row) in enumerate(cond_data.iterrows()):
            for i, rxn in enumerate(REACTION_ORDER):
                if rxn in row and rxn in TRUE_FLUXES and TRUE_FLUXES[rxn] != 0:
                    pct_diff[i, j] = (
                        (row[rxn] - TRUE_FLUXES[rxn]) / TRUE_FLUXES[rxn] * 100
                    )

        # Symmetric color limits
        vmax = np.nanmax(np.abs(pct_diff))
        if vmax < 0.1:
            vmax = 0.1  # floor for near-zero errors

        im = ax.imshow(
            pct_diff,
            aspect="auto",
            cmap=cmc.vik,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )

        # Axis labels
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45, ha="right")
        ax.set_xlabel(r"$\alpha$", fontsize=12)
        ax.set_title(cond_name, fontsize=14)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Flux error (%)", fontsize=10)

        # Annotate cells with values
        for i in range(pct_diff.shape[0]):
            for j in range(pct_diff.shape[1]):
                val = pct_diff[i, j]
                if np.isfinite(val):
                    color = "white" if abs(val) > 0.6 * vmax else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color=color,
                    )

    axes[0].set_yticks(range(len(REACTION_ORDER)))
    axes[0].set_yticklabels(REACTION_ORDER, fontsize=9)
    axes[0].set_ylabel("Reaction", fontsize=12)

    fig.suptitle(f"Net Flux Error ({sim_mode.upper()})", fontsize=16, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = output_dir / f"flux_error_heatmap_{sim_mode}.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
        sys.stdout.flush()
    plt.close()


# ============================================================================
# Main
# ============================================================================


# Output filename suffix per fitted model (aggregated keeps the legacy names).
MODEL_SUFFIX = {"aggregated": "", "mechanistic": "_mech"}


# Simulation modes: label -> (simulation fn, restart count, max solver iters)
SIM_MODES = {
    "ss": (run_ss_simulation, N_STARTS, MAX_ITERS_SS),
    "inst": (run_inst_simulation, N_STARTS_INST, MAX_ITERS_INST),
}


def _merge_and_save(new_df, path, sim_modes_to_run, sort_cols=None):
    """Save new rows to a CSV, preserving rows for modes not being rerun.

    If the CSV exists and only a subset of simulation modes was run, rows for
    the rerun modes are dropped from the old file and replaced by `new_df`;
    rows for other modes are kept. A full run overwrites the file.

    Returns the saved DataFrame.
    """
    if (
        path.exists()
        and set(sim_modes_to_run) != set(SIM_MODES)
        and len(new_df)
    ):
        old = pd.read_csv(path)
        if "sim_mode" in old.columns:
            old = old[~old["sim_mode"].isin(sim_modes_to_run)]
            new_df = pd.concat([old, new_df], ignore_index=True, sort=False)

    if sort_cols and len(new_df):
        present = [c for c in sort_cols if c in new_df.columns]
        if present:
            new_df = new_df.sort_values(present).reset_index(drop=True)

    new_df.to_csv(path, index=False)
    return new_df


def run_fitting(sim_modes_to_run=None):
    """Fit the aggregated and mechanistic models to the mechanistic MIDs.

    For each (condition, alpha) the mechanistic model generates synthetic MDVs
    in the selected simulation modes, then both models are fit to each. Results
    carry `sim_mode` and `model` columns. When only a subset of modes is run,
    results are merged into the existing CSVs (rows for the rerun modes are
    replaced; other modes preserved).

    Args:
        sim_modes_to_run: list of mode labels to run (default: all of SIM_MODES).
    """
    if sim_modes_to_run is None:
        sim_modes_to_run = list(SIM_MODES)

    all_results = []
    flux_rows = {m: [] for m in FIT_MODELS}
    total_flux_rows = {m: [] for m in FIT_MODELS}

    for cond_name, cond_config in CONDITIONS.items():
        thermo_file = cond_config["thermo_file"]
        total_fba_dG = cond_config["total_fba_dG"]
        total = len(ALPHA_VALUES)

        print(f"\n{'=' * 80}")
        print(f"Condition: {cond_name} (total FBA dG = {total_fba_dG} kJ/mol)")
        print(f"{'=' * 80}")
        sys.stdout.flush()

        for i, alpha in enumerate(ALPHA_VALUES):
            fba1_dG = alpha * total_fba_dG
            fba2_dG = (1 - alpha) * total_fba_dG
            base = {
                "condition": cond_name,
                "alpha": alpha,
                "FBA1_dG": fba1_dG,
                "FBA2_dG": fba2_dG,
            }

            print(
                f"  [{i + 1}/{total}] alpha={alpha:.2f}: "
                f"FBA1 dG={fba1_dG:.3f}, FBA2 dG={fba2_dG:.3f} kJ/mol"
            )
            sys.stdout.flush()

            for sim_mode in sim_modes_to_run:
                sim_fn, n_starts, max_iters = SIM_MODES[sim_mode]
                # Forward simulation with mechanistic model (shared by both fits)
                try:
                    overrides = {"FBA1": fba1_dG, "FBA2": fba2_dG}
                    measured = sim_fn(thermo_file, overrides)
                except Exception as e:
                    print(f"    [{sim_mode}] simulation error: {e}")
                    sys.stdout.flush()
                    for model_label in FIT_MODELS:
                        all_results.append(
                            {
                                **base,
                                "sim_mode": sim_mode,
                                "model": model_label,
                                "SSR": np.nan,
                                "chi2_pass": None,
                                "optimization_successful": False,
                            }
                        )
                    continue

                # Fit each model to the same MDVs
                for model_label, (model_id, model_dir, init_fluxes) in (
                    FIT_MODELS.items()
                ):
                    try:
                        fit_result = fit_model(
                            model_id, model_dir, init_fluxes, measured,
                            sim_mode, n_starts, max_iters,
                        )
                    except Exception as e:
                        print(f"    [{sim_mode}/{model_label}] fit error: {e}")
                        sys.stdout.flush()
                        all_results.append(
                            {
                                **base,
                                "sim_mode": sim_mode,
                                "model": model_label,
                                "SSR": np.nan,
                                "chi2_pass": None,
                                "optimization_successful": False,
                            }
                        )
                        continue

                    print(
                        f"    [{sim_mode}/{model_label}] "
                        f"SSR={fit_result['SSR']:.6f}, "
                        f"chi2_pass={fit_result['chi2_pass']}, "
                        f"success={fit_result['optimization_successful']}"
                    )
                    sys.stdout.flush()

                    all_results.append(
                        {
                            **base,
                            "sim_mode": sim_mode,
                            "model": model_label,
                            "SSR": fit_result["SSR"],
                            "chi2_pass": fit_result["chi2_pass"],
                            "optimization_successful": fit_result[
                                "optimization_successful"
                            ],
                        }
                    )

                    flux_row = {**base, "sim_mode": sim_mode}
                    for flux_id, value in fit_result["fitted_fluxes"].items():
                        flux_row[flux_id] = float(value)
                    flux_rows[model_label].append(flux_row)

                    total_flux_row = {**base, "sim_mode": sim_mode}
                    for flux_id, value in fit_result["fitted_fluxes"].items():
                        total_flux_row[flux_id] = float(value)
                    for flux_id, value in fit_result["fitted_reverse_fluxes"].items():
                        total_flux_row[f"{flux_id}_r"] = float(value)
                    total_flux_rows[model_label].append(total_flux_row)

    # Save results
    print(f"\n{'=' * 80}")
    print("Saving Results")
    print(f"{'=' * 80}")
    sys.stdout.flush()

    df_results = _merge_and_save(
        pd.DataFrame(all_results),
        RESULTS_DIR / "fit_results.csv",
        sim_modes_to_run,
        sort_cols=["sim_mode", "condition", "alpha", "model"],
    )
    print(f"  Saved: {RESULTS_DIR / 'fit_results.csv'}")
    sys.stdout.flush()

    # Per-model flux CSVs (aggregated keeps legacy names; mechanistic gets _mech)
    for model_label in FIT_MODELS:
        suf = MODEL_SUFFIX[model_label]
        for stem, rows in [
            (f"fitted_fluxes{suf}", flux_rows[model_label]),
            (f"fitted_total_fluxes{suf}", total_flux_rows[model_label]),
        ]:
            path = RESULTS_DIR / f"{stem}.csv"
            _merge_and_save(
                pd.DataFrame(rows), path, sim_modes_to_run,
                sort_cols=["sim_mode", "condition", "alpha"],
            )
            print(f"  Saved: {path}")
            sys.stdout.flush()

    # Save true fluxes for reference
    true_path = RESULTS_DIR / "true_fluxes.csv"
    pd.Series(TRUE_FLUXES, name="flux_umol_per_gCDW_per_s").to_csv(true_path)
    print(f"  Saved: {true_path}")
    sys.stdout.flush()

    # Summary (over all modes now present in the merged results)
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    present_modes = [m for m in SIM_MODES if (df_results["sim_mode"] == m).any()]
    for cond_name in ["FFE", "NE"]:
        print(f"\n{cond_name}:")
        for sim_mode in present_modes:
            for model_label in FIT_MODELS:
                cond_data = df_results[
                    (df_results["condition"] == cond_name)
                    & (df_results["sim_mode"] == sim_mode)
                    & (df_results["model"] == model_label)
                ]
                n_success = cond_data["optimization_successful"].sum()
                print(
                    f"  [{sim_mode}/{model_label}] successful optimizations: "
                    f"{n_success}/{len(cond_data)}"
                )
                if n_success > 0:
                    valid = cond_data[cond_data["optimization_successful"]]
                    min_idx = valid["SSR"].idxmin()
                    print(
                        f"    SSR range: {valid['SSR'].min():.6f} - "
                        f"{valid['SSR'].max():.6f}; "
                        f"min at alpha={valid.loc[min_idx, 'alpha']:.2f}"
                    )
    sys.stdout.flush()

    return df_results


def make_plots():
    """Generate plots from saved CSVs (no fitting required)."""
    df_results = pd.read_csv(RESULTS_DIR / "fit_results.csv")
    df_fluxes = pd.read_csv(RESULTS_DIR / "fitted_fluxes.csv")

    print("Creating SSR vs partition plot...")
    sys.stdout.flush()
    plot_ssr_vs_partition(df_results, RESULTS_DIR)

    sim_modes = (
        sorted(df_fluxes["sim_mode"].unique())
        if "sim_mode" in df_fluxes.columns
        else ["ss"]
    )
    for sim_mode in sim_modes:
        print(f"Creating flux error heatmaps ({sim_mode})...")
        sys.stdout.flush()
        plot_flux_error_heatmaps(df_fluxes, RESULTS_DIR, sim_mode=sim_mode)


def main():
    """Run fitting analysis and generate plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from saved CSVs without re-running fitting.",
    )
    parser.add_argument(
        "--sim-modes",
        default=",".join(SIM_MODES),
        help=(
            "Comma-separated simulation modes to run (default: all). "
            "Running a subset (e.g. 'inst') merges into existing CSVs, "
            "preserving the other modes' rows."
        ),
    )
    args = parser.parse_args()

    sim_modes_to_run = [m.strip() for m in args.sim_modes.split(",") if m.strip()]
    unknown = [m for m in sim_modes_to_run if m not in SIM_MODES]
    if unknown:
        parser.error(f"unknown sim mode(s): {unknown}; choose from {list(SIM_MODES)}")

    print("=" * 80)
    print("Fit Aggregated and Mechanistic Models to Mechanistic Model MIDs")
    print("=" * 80)
    sys.stdout.flush()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()

    if args.plot_only:
        print("  (--plot-only: loading results from saved CSVs)")
        sys.stdout.flush()
        make_plots()
    else:
        print(f"  Simulation modes to run: {sim_modes_to_run}")
        sys.stdout.flush()
        run_fitting(sim_modes_to_run)
        make_plots()

    print(f"\nGenerated files in: {RESULTS_DIR}")
    print("  - fit_results.csv (both models/modes, `sim_mode` + `model` columns)")
    print("  - fitted_fluxes.csv / fitted_total_fluxes.csv (aggregated, `sim_mode`)")
    print("  - fitted_fluxes_mech.csv / fitted_total_fluxes_mech.csv (mechanistic)")
    print("  - true_fluxes.csv")
    print("  - fit_ssr_vs_partition.png/pdf")
    print("  - flux_error_heatmap_ss.png/pdf, flux_error_heatmap_inst.png/pdf")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

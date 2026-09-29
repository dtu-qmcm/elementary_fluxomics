#!/usr/bin/env python3
"""Fit Mechanistic MIDs with the Mechanistic and Aggregated Models.

Generates noise-free MIDs from the mechanistic model with the setup of
gibbs_high_oxppp_comparison_ne.py (NE thermodynamics, FBA dG split, tracer and
measured EMUs) on the standard flux map (glycolysis_fluxes.toml), and fits them
with freeflux's SS and INST fitters using both the mechanistic model (self-fit,
the control) and the aggregated model (single lumped FBA).

Measurements:
  - MIDs: sd = 0.002 (0.2 mol%) on every isotopologue; INST MIDs at 5
    log-spaced timepoints from 1e-3 to 10 s
  - Fluxes: glc_input (sd 1%), r5p_drain and GLY3PDH, the dhap drain (sd 5%)

Every start is seeded with explicit total (forward/backward) fluxes: freeflux
only accepts initial net fluxes and maps them to free fluxes by a minimum-norm
projection, which leaves the near-equilibrium exchange fluxes far from the
truth. No start sits at the reference point: every start scales each true net
and exchange flux by U(0.5, 1.5). The starts of each fit are split into small
tasks that run in parallel worker processes. INST pool sizes are fixed at the
values used to simulate the data.

LDH and pyr_drain both export pyruvate with identical atom mapping and neither
is measured, so only their sum is identifiable.
"""

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import io
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from freeflux import Model
from freeflux.solver.nlpsolver import InstMFAModel, MFAModel
from scipy.linalg import pinv
from scipy.stats import chi2

from gibbs_high_oxppp_comparison_ne import (
    AGG_MODEL_DIR,
    CONVERSION_FACTOR,
    FBA1_FRACTION,
    LABELING_PATTERN,
    MECH_MODEL_DIR,
    TARGET_EMUS,
    TOTAL_FBA_DG,
    build_fluxes,
    pool_size,
    run_simulation,
)
from style import apply_style, DOUBLE_COL

apply_style()

# ============================================================================
# Configuration
# ============================================================================

# Paths - relative to scripts folder
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results" / "gibbs_standard_fit_ne"

# Flux map (file in each model directory) that generates and seeds the fits
FLUX_FILE = "glycolysis_fluxes.toml"

# INST measurement times: 5 points logarithmically spaced from 1e-3s to 1e1s
INST_TIMEPOINTS = list(np.logspace(-3, 1, 5))

# Measurement uncertainty on every isotopologue fraction (0.2 mol%)
MDV_SD = 0.002

# Measured fluxes and their sd as a fraction of the true value
MEASURED_FLUX_SD_FRAC = {
    "glc_input": 0.01,
    "r5p_drain": 0.05,
    "GLY3PDH": 0.05,  # the dhap drain
}

# Net flux bounds (umol/gCDW/s); reactions with a negative true net flux get
# the mirrored range [-200, 0]
NET_FLUX_BOUNDS = [0.0, 200.0]

# INST pool sizes are pinned to within +-0.1% of their simulated values
# (freeflux requires lower < upper)
POOL_TOL = 1e-3

# Optimizer settings per fitting mode
N_STARTS = 500

# Starts per worker task: INST starts take minutes each, SS starts seconds
STARTS_PER_TASK = {"ss": 10, "inst": 2}
MAX_ITERS = {"ss": 1000, "inst": 2000}
SLSQP_TOL = 1e-8

# Every start begins at the true fluxes with every net flux and every exchange
# flux, min(v_f, v_b), scaled by U(1 - PERTURB, 1 + PERTURB)
PERTURB = 0.5
RNG_SEED = 12345  # combined with the start index, so starts do not depend on chunking

# Two-sided confidence level of the chi-squared acceptance range
CHI2_CONFIDENCE = 0.95

# Gas constant in kJ/(mol*K) and temperature (K) of the NE thermodynamics
R = 8.314e-3
TEMPERATURE = 298.15

# Unmeasured exports with identical atom mapping: only their sum is identifiable
NON_IDENTIFIABLE = ("LDH", "pyr_drain")

# Only near-equilibrium reactions (|true deltaG| below this, kJ/mol) are plotted:
# far from equilibrium the backward flux is too small to affect labeling, so
# the deltaG is not identifiable
GIBBS_PLOT_MAX_ABS_DG = 2.0

# deltaG % errors beyond +-this are drawn clipped, labelled with their value
GIBBS_PLOT_CLIP_PCT = 200.0

FBA1_DG = FBA1_FRACTION * TOTAL_FBA_DG
FBA2_DG = (1 - FBA1_FRACTION) * TOTAL_FBA_DG

# Fitted models: label -> (model directory, deltaG overrides defining its truth)
FIT_MODELS = {
    "aggregated": (AGG_MODEL_DIR, {"FBA": TOTAL_FBA_DG}),
    "mechanistic": (MECH_MODEL_DIR, {"FBA1": FBA1_DG, "FBA2": FBA2_DG}),
}

MODES = ("ss", "inst")
MODE_TITLES = {"ss": "Steady state", "inst": "Instationary"}

MODEL_COLORS = {
    "aggregated": cmc.batlow(0.15),
    "mechanistic": cmc.batlow(0.70),
}


# ============================================================================
# Helper Functions
# ============================================================================


def simulate_data():
    """Noise-free mechanistic MDVs: {"ss": {emu: mdv}, "inst": {emu: {t: mdv}}}."""
    model_dir, overrides = FIT_MODELS["mechanistic"]
    mdvs_tc, mdvs_ss, _ = run_simulation(
        model_dir,
        "glycolysis_mech_gly3p",
        overrides,
        timepoints=INST_TIMEPOINTS,
        flux_file=FLUX_FILE,
    )
    mdvs_tc = {
        emu_id: {t: mdv for t, mdv in by_t.items() if t > 0}
        for emu_id, by_t in mdvs_tc.items()
    }
    return {"ss": mdvs_ss, "inst": mdvs_tc}


def net_from_total(total_fluxes):
    """Net fluxes {rxn_id: v} from total fluxes with _f/_b reversible pairs."""
    net = {}
    for flux_id, value in total_fluxes.items():
        if flux_id.endswith("_f"):
            rxn_id = flux_id[:-2]
            net[rxn_id] = value - total_fluxes[f"{rxn_id}_b"]
        elif not flux_id.endswith("_b"):
            net[flux_id] = value
    return net


def gibbs_from_total(total_fluxes):
    """deltaG = -RT ln(v_f / v_b) for every reversible reaction (kJ/mol)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            flux_id[:-2]: float(
                -R * TEMPERATURE
                * np.log(total_fluxes[flux_id] / total_fluxes[f"{flux_id[:-2]}_b"])
            )
            for flux_id in total_fluxes
            if flux_id.endswith("_f")
        }


def write_tsv(values, path):
    """Write {id: value} as a freeflux initial-values TSV."""
    with open(path, "w") as f:
        f.write("#id\tvalue\n")
        for key, value in values.items():
            f.write(f"{key}\t{value}\n")


def perturb_fluxes(true_total, N, T, totalfluxids, netfluxids, rng):
    """Balanced total fluxes near the truth for a perturbed start.

    Every net flux and every exchange flux, min(v_f, v_b), is scaled by
    U(1 - PERTURB, 1 + PERTURB). The net fluxes are then projected back onto
    the steady-state space; exchange fluxes do not enter the balances.

    Args:
        true_total: {flux_id: value} true total fluxes.
        N: null space mapping free fluxes to total fluxes.
        T: transform matrix mapping total fluxes to net fluxes.
        totalfluxids, netfluxids: freeflux flux ID orders of N's rows and T's rows.
        rng: numpy random Generator.
    """
    v_true = pd.Series(true_total)[totalfluxids].to_numpy()
    TN = T @ N
    net = (T @ v_true) * rng.uniform(1 - PERTURB, 1 + PERTURB, len(netfluxids))
    net = TN @ pinv(TN) @ net

    perturbed = {}
    for rxn_id, v_net in zip(netfluxids, net):
        if f"{rxn_id}_f" in true_total:
            v_xch = min(true_total[f"{rxn_id}_f"], true_total[f"{rxn_id}_b"])
            v_xch *= rng.uniform(1 - PERTURB, 1 + PERTURB)
            perturbed[f"{rxn_id}_f"] = max(v_net, 0.0) + v_xch
            perturbed[f"{rxn_id}_b"] = max(-v_net, 0.0) + v_xch
        else:
            perturbed[rxn_id] = v_net
    return perturbed


@contextlib.contextmanager
def seeded_start(u0):
    """Start freeflux's optimizer from the free fluxes u0 instead of its own guess.

    freeflux builds the initial point from net fluxes alone; this replaces the
    free-flux block of that point after it is built.
    """
    build_ss = MFAModel.build_initial_flux_values
    build_inst = InstMFAModel.build_initial_flux_and_conc_values

    def seeded_ss(self, ini_netfluxes=None):
        build_ss(self, ini_netfluxes)
        self.x0 = np.array(u0, dtype=float)

    def seeded_inst(self, ini_netfluxes=None, ini_concs=None):
        build_inst(self, ini_netfluxes, ini_concs)
        self.x0[: self.nfreefluxes] = u0

    MFAModel.build_initial_flux_values = seeded_ss
    InstMFAModel.build_initial_flux_and_conc_values = seeded_inst
    try:
        yield
    finally:
        MFAModel.build_initial_flux_values = build_ss
        InstMFAModel.build_initial_flux_and_conc_values = build_inst


def extract_residuals(result, mode):
    """Rows of measured vs fitted MDVs at the optimum."""
    rows = []
    if mode == "ss":
        for emu_id, (mean, sd) in result.measured_MDVs.items():
            fitted = np.asarray(result.simulated_MDVs[emu_id])
            for k, (m, s, f) in enumerate(zip(np.asarray(mean), np.asarray(sd), fitted)):
                rows.append((emu_id, np.nan, k, m, f, s))
    else:
        for emu_id, by_t in result.measured_inst_MDVs.items():
            for t, (mean, sd) in by_t.items():
                fitted = np.asarray(result.simulated_inst_MDVs[emu_id][t])
                for k, (m, s, f) in enumerate(zip(np.asarray(mean), np.asarray(sd), fitted)):
                    rows.append((emu_id, t, k, m, f, s))
    return [
        {
            "emu": emu_id,
            "timepoint": t,
            "isotopologue": k,
            "measured": float(m),
            "fitted": float(f),
            "sd": float(s),
            "norm_residual": float((f - m) / s),
        }
        for emu_id, t, k, m, f, s in rows
    ]


def fit_model(job):
    """Fit one model to one data set from a chunk of starts (worker process).

    Args:
        job: Tuple of (mode, model_label, data, start_ids) with data in the
            format of simulate_data()[mode].

    Returns:
        dict with the true and fitted fluxes, fit statistics, per-start
        objectives and MDV residuals of the chunk's best start (fit fields are
        None if every start in the chunk failed).
    """
    mode, model_label, data, start_ids = job
    model_dir, overrides = FIT_MODELS[model_label]
    tag = f"[{mode}/{model_label}]"

    model = Model(f"{model_label}_{mode}")
    model.read_from_file(str(model_dir / "reactions.tsv"))
    true_total, _ = build_fluxes(model, model_dir, overrides, FLUX_FILE)
    true_net = net_from_total(true_total)

    with model.fitter(mode) as fitter, tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fitter.set_labeling_strategy(
            "Glc_ext",
            labeling_pattern=[LABELING_PATTERN],
            percentage=[1.0],
            purity=[1.0],
        )

        if mode == "ss":
            for emu_id, mdv in data.items():
                fitter.set_measured_MDV(emu_id, mean=mdv, sd=np.full(len(mdv), MDV_SD))
        else:
            for emu_id, by_t in data.items():
                tps = sorted(by_t)
                fitter.set_measured_MDVs(
                    emu_id,
                    tps,
                    [by_t[t] for t in tps],
                    [np.full(len(by_t[t]), MDV_SD) for t in tps],
                )
            fitter.set_timepoints(INST_TIMEPOINTS)
            for metab in model.metabolites:
                c = pool_size(metab)
                fitter.set_concentration_bounds(metab, [c * (1 - POOL_TOL), c * (1 + POOL_TOL)])

        # The measured reactions are irreversible, so total = net
        for rxn_id, sd_frac in MEASURED_FLUX_SD_FRAC.items():
            v = true_total[rxn_id]
            fitter.set_measured_flux(rxn_id, v, sd_frac * abs(v))

        fitter.set_flux_bounds("all", NET_FLUX_BOUNDS)
        for rxn_id, v in true_net.items():
            if v < 0:
                fitter.set_flux_bounds(rxn_id, [-NET_FLUX_BOUNDS[1], 0.0])

        fitter.prepare(n_jobs=1)

        # Free fluxes that reproduce the true total fluxes exactly
        N = fitter.model.null_space
        T = fitter.model.transform_matrix
        totalfluxids = fitter.model.totalfluxids
        pinv_N = pinv(N)
        v_true = pd.Series(true_total)[totalfluxids].to_numpy()
        seed_residual = float(np.max(np.abs(N @ pinv_N @ v_true - v_true)))
        if 0 in start_ids:
            print(f"  {tag} seed reconstruction residual {seed_residual:.1e}")
            sys.stdout.flush()

        solve_kwargs = dict(
            fit_measured_fluxes=True,
            solver="slsqp",
            tol=SLSQP_TOL,
            max_iters=MAX_ITERS[mode],
            show_progress=False,
        )
        if mode == "inst":
            ini_concs_path = tmpdir / "ini_concs.tsv"
            write_tsv({m: pool_size(m) for m in fitter.model.concids}, ini_concs_path)
            solve_kwargs["ini_concs"] = str(ini_concs_path)

        best, best_start = None, None
        starts = []
        for start in start_ids:
            rng = np.random.default_rng([RNG_SEED, start])
            start_total = perturb_fluxes(
                true_total, N, T, totalfluxids, fitter.model.netfluxids, rng
            )
            u_start = pinv_N @ pd.Series(start_total)[totalfluxids].to_numpy()
            ini_path = tmpdir / f"ini_fluxes_{start}.tsv"
            write_tsv(net_from_total(start_total), ini_path)

            t0 = time.time()
            try:
                with seeded_start(u_start), contextlib.redirect_stdout(io.StringIO()):
                    result = fitter.solve(ini_fluxes=str(ini_path), **solve_kwargs)
            except Exception as e:
                print(f"  {tag} start {start}: failed ({type(e).__name__}: {e})")
                sys.stdout.flush()
                starts.append({"start": start, "SSR": np.nan, "success": False})
                continue

            starts.append({
                "start": start,
                "SSR": float(result.opt_objective),
                "success": bool(result.optimization_successful),
            })
            print(
                f"  {tag} start {start}: SSR = {result.opt_objective:.4g} "
                f"({time.time() - t0:.0f} s)"
            )
            sys.stdout.flush()
            if np.isfinite(result.opt_objective) and (
                best is None or result.opt_objective < best.opt_objective
            ):
                best, best_start = result, start

    fit = {
        "mode": mode,
        "model": model_label,
        "true_total": true_total,
        "seed_residual": seed_residual,
        "starts": starts,
        "best_start": best_start,
        "SSR": np.nan,
        "fitted_total": None,
        "n_meas": None,
        "n_params": None,
        "dof": None,
        "success": False,
        "residuals": None,
    }
    if best is not None:
        fit.update({
            "SSR": float(best.opt_objective),
            "fitted_total": {k: float(v) for k, v in best.opt_total_fluxes.items()},
            "n_meas": int(best.n_meas),
            "n_params": int(best.n_params),
            "dof": int(best.dof),
            "success": bool(best.optimization_successful),
            "residuals": extract_residuals(best, mode),
        })
    return fit


def merge_chunks(chunks):
    """Combine chunk results into one fit per (mode, model): the best chunk's
    fit, with the per-start objectives of every chunk."""
    fits = []
    for mode in MODES:
        for model in FIT_MODELS:
            group = [c for c in chunks if c["mode"] == mode and c["model"] == model]
            finite = [c for c in group if np.isfinite(c["SSR"])]
            if not finite:
                raise RuntimeError(f"[{mode}/{model}] all starts failed")
            best = min(finite, key=lambda c: c["SSR"])
            starts = sorted((s for c in group for s in c["starts"]), key=lambda s: s["start"])
            fits.append({**best, "starts": starts})
    return fits


# ============================================================================
# Tables
# ============================================================================


def summary_row(fit):
    """Fit statistics with the chi-squared acceptance range."""
    lo, hi = chi2.ppf([(1 - CHI2_CONFIDENCE) / 2, (1 + CHI2_CONFIDENCE) / 2], fit["dof"])
    return {
        "sim_mode": fit["mode"],
        "model": fit["model"],
        "SSR": fit["SSR"],
        "n_meas": fit["n_meas"],
        "n_params": fit["n_params"],
        "dof": fit["dof"],
        "chi2_lower": lo,
        "chi2_upper": hi,
        # SSR below the lower bound is expected for noise-free data
        "accepted": fit["SSR"] <= hi,
        "optimization_successful": fit["success"],
        "best_start": fit["best_start"],
        "seed_residual": fit["seed_residual"],
    }


def flux_rows(fit):
    """Per-reaction true vs fitted net flux (mmol/gDCW/h) and deltaG (kJ/mol)."""
    true_net = net_from_total(fit["true_total"])
    fitted_net = net_from_total(fit["fitted_total"])
    true_dg = gibbs_from_total(fit["true_total"])
    fitted_dg = gibbs_from_total(fit["fitted_total"])

    rows = []
    for rxn_id, v_true in true_net.items():
        v_fit = fitted_net[rxn_id]
        rows.append({
            "sim_mode": fit["mode"],
            "model": fit["model"],
            "rxn_id": rxn_id,
            "reversible": rxn_id in true_dg,
            "identifiable": rxn_id not in NON_IDENTIFIABLE,
            "true_net": v_true / CONVERSION_FACTOR,
            "fitted_net": v_fit / CONVERSION_FACTOR,
            "pct_error": (v_fit - v_true) / v_true * 100,
            "true_dG": true_dg.get(rxn_id, np.nan),
            "fitted_dG": fitted_dg.get(rxn_id, np.nan),
        })
    return rows


def save_results(fits):
    """Write the fit CSVs and return (summary, fluxes, residuals, starts)."""
    df_summary = pd.DataFrame([summary_row(fit) for fit in fits])
    df_fluxes = pd.DataFrame([row for fit in fits for row in flux_rows(fit)])
    df_residuals = pd.DataFrame([
        {"sim_mode": fit["mode"], "model": fit["model"], **row}
        for fit in fits
        for row in fit["residuals"]
    ])
    df_starts = pd.DataFrame([
        {"sim_mode": fit["mode"], "model": fit["model"], **row}
        for fit in fits
        for row in fit["starts"]
    ])

    for name, df in [
        ("fit_summary", df_summary),
        ("fitted_fluxes", df_fluxes),
        ("mdv_residuals", df_residuals),
        ("start_objectives", df_starts),
    ]:
        path = RESULTS_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved CSV: {path}")
    sys.stdout.flush()
    return df_summary, df_fluxes, df_residuals, df_starts


def load_results():
    """Read the fit CSVs written by save_results()."""
    return tuple(
        pd.read_csv(RESULTS_DIR / f"{name}.csv")
        for name in ("fit_summary", "fitted_fluxes", "mdv_residuals", "start_objectives")
    )


# ============================================================================
# Plotting
# ============================================================================


def save_figure(fig, stem):
    """Save a figure as PNG and PDF in RESULTS_DIR."""
    for ext in ("png", "pdf"):
        path = RESULTS_DIR / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"  Saved: {path}")
    sys.stdout.flush()
    plt.close(fig)


def flux_error_table(df_fluxes, mode, model):
    """% net flux error per reaction, with LDH and pyr_drain summed.

    FBA1 stands in for the mechanistic FBA row (FBA1 and FBA2 are in series
    and carry the same net flux).
    """
    data = df_fluxes[(df_fluxes["sim_mode"] == mode) & (df_fluxes["model"] == model)]
    data = data[data["rxn_id"] != "FBA2"].replace({"rxn_id": {"FBA1": "FBA"}})
    errors = dict(zip(data["rxn_id"], data["pct_error"]))

    pooled = data[data["rxn_id"].isin(NON_IDENTIFIABLE)]
    errors["LDH + pyr_drain"] = (
        (pooled["fitted_net"].sum() - pooled["true_net"].sum()) / pooled["true_net"].sum() * 100
    )
    for rxn_id in NON_IDENTIFIABLE:
        errors.pop(rxn_id)
    return errors


def plot_panels(stem, row_values, xlabel, title_suffix=None, rows=None, clip=None):
    """Grouped horizontal bars (one bar per model) in one panel per mode.

    Each bar also gets a dot at its value, so that zero errors stay visible.

    Args:
        stem: Output file stem.
        row_values: {mode: {model: {row_label: value}}}.
        xlabel: x-axis label.
        title_suffix: optional {mode: str} appended to each panel title.
        rows: row labels in plotting order (default: order of first appearance).
        clip: optional bound; bars beyond +-clip are drawn to it and labelled
            with their actual value.
    """
    if rows is None:
        rows = []
        for by_model in row_values.values():
            for values in by_model.values():
                rows.extend(r for r in values if r not in rows)

    fig, axes = plt.subplots(
        1, len(MODES), figsize=(DOUBLE_COL, 0.16 * len(rows) + 1.4), sharey=True
    )
    height = 0.8 / len(FIT_MODELS)
    y = np.arange(len(rows))

    for ax, mode in zip(axes, MODES):
        for i, model in enumerate(FIT_MODELS):
            values = np.array([row_values[mode][model].get(r, np.nan) for r in rows])
            shown = np.clip(values, -clip, clip) if clip else values
            y_model = y + (i - (len(FIT_MODELS) - 1) / 2) * height
            ax.barh(y_model, shown, height=height, color=MODEL_COLORS[model], label=model)
            ax.plot(shown, y_model, "o", markersize=2.5, color=MODEL_COLORS[model])
            if clip:
                for v, x, y_bar in zip(values, shown, y_model):
                    if abs(v) > clip:
                        ax.text(
                            x * 0.97, y_bar, f"{v:,.0f}", ha="right" if v > 0 else "left",
                            va="center", fontsize=6, color="white", fontweight="bold",
                        )
        ax.axvline(0.0, color="black", linewidth=0.6)
        ax.set_xlabel(xlabel)
        title = MODE_TITLES[mode]
        if title_suffix:
            title += f"\n{title_suffix[mode]}"
        ax.set_title(title, fontsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows)
    axes[0].invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(FIT_MODELS))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, stem)


def ssr_labels(df_summary):
    """Per-mode panel subtitle with each model's SSR and chi2 verdict."""
    labels = {}
    for mode in MODES:
        parts = []
        for model in FIT_MODELS:
            row = df_summary[(df_summary["sim_mode"] == mode) & (df_summary["model"] == model)].iloc[0]
            verdict = "accepted" if row["accepted"] else "rejected"
            parts.append(f"{model}: SSR {row['SSR']:.3g}, {verdict}")
        labels[mode] = "\n".join(parts)
    return labels


def plot_flux_errors(df_summary, df_fluxes):
    """% error of the fitted net fluxes."""
    values = {
        mode: {model: flux_error_table(df_fluxes, mode, model) for model in FIT_MODELS}
        for mode in MODES
    }
    plot_panels("fit_flux_errors", values, "Net flux error (%)", ssr_labels(df_summary))


def plot_gibbs_errors(df_summary, df_fluxes):
    """% error of the fitted Gibbs energies of the near-equilibrium reactions.

    Error = (fitted - true) / true * 100, so a positive error means the fit puts
    the reaction further from equilibrium than it is.
    """
    near_eq = df_fluxes[
        df_fluxes["reversible"] & (df_fluxes["true_dG"].abs() < GIBBS_PLOT_MAX_ABS_DG)
    ]
    values = {}
    for mode in MODES:
        values[mode] = {}
        for model in FIT_MODELS:
            data = near_eq[(near_eq["sim_mode"] == mode) & (near_eq["model"] == model)]
            pct_error = (data["fitted_dG"] - data["true_dG"]) / data["true_dG"] * 100
            values[mode][model] = dict(zip(data["rxn_id"], pct_error))

    # Aggregated reaction order, with the mechanistic FBA1/FBA2 right after FBA
    rows = []
    for rxn_id in dict.fromkeys(near_eq.loc[near_eq["model"] == "aggregated", "rxn_id"]):
        rows.extend(["FBA", "FBA1", "FBA2"] if rxn_id == "FBA" else [rxn_id])
    plot_panels(
        "fit_gibbs_errors",
        values,
        r"$\Delta_r G$ error (%)",
        ssr_labels(df_summary),
        rows=rows,
        clip=GIBBS_PLOT_CLIP_PCT,
    )


def plot_mdv_residuals(df_residuals):
    """Normalized MDV residuals at the optimum, grouped by EMU."""
    emus = [f"{met}_{atoms[0]}" for met, atoms in TARGET_EMUS.items()]
    fig, axes = plt.subplots(
        1, len(MODES), figsize=(DOUBLE_COL, DOUBLE_COL * 0.38), sharey=True
    )
    offset = 0.35 / len(FIT_MODELS)

    for ax, mode in zip(axes, MODES):
        for i, model in enumerate(FIT_MODELS):
            data = df_residuals[
                (df_residuals["sim_mode"] == mode) & (df_residuals["model"] == model)
            ]
            x = data["emu"].map({e: j for j, e in enumerate(emus)})
            # Spread the points of one EMU so the isotopologues stay visible
            spread = (data["isotopologue"] / 6.0 - 0.5) * 0.12
            ax.scatter(
                x + (i - (len(FIT_MODELS) - 1) / 2) * 2 * offset + spread,
                data["norm_residual"],
                s=6,
                color=MODEL_COLORS[model],
                label=model,
            )
        for level in (-2, 2):
            ax.axhline(level, color="gray", linewidth=0.6, linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xticks(range(len(emus)))
        ax.set_xticklabels([e.split("_")[0] for e in emus])
        ax.set_title(MODE_TITLES[mode])

    axes[0].set_ylabel("(fitted $-$ measured) / sd")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    save_figure(fig, "fit_mdv_residuals")


def make_plots(df_summary, df_fluxes, df_residuals):
    """Create all figures from the result tables."""
    print("\n  Creating flux error plot...")
    plot_flux_errors(df_summary, df_fluxes)
    print("  Creating Gibbs energy error plot...")
    plot_gibbs_errors(df_summary, df_fluxes)
    print("  Creating MDV residual plot...")
    plot_mdv_residuals(df_residuals)


def print_summary(df_summary, df_fluxes):
    """Console summary of goodness of fit and the FBA Gibbs energies."""
    print(f"\nGoodness of fit ({CHI2_CONFIDENCE:.0%} chi-squared range):")
    print(f"  {'mode':<6}{'model':<13}{'SSR':>10}{'dof':>6}{'chi2 range':>20}  verdict")
    for _, row in df_summary.iterrows():
        rng = f"[{row['chi2_lower']:.1f}, {row['chi2_upper']:.1f}]"
        verdict = "accepted" if row["accepted"] else "REJECTED"
        print(
            f"  {row['sim_mode']:<6}{row['model']:<13}{row['SSR']:>10.3g}"
            f"{row['dof']:>6}{rng:>20}  {verdict}"
        )

    print("\nFBA Gibbs energies (kJ/mol):")
    fba = df_fluxes[df_fluxes["rxn_id"].isin(["FBA", "FBA1", "FBA2"])]
    for _, row in fba.iterrows():
        print(
            f"  {row['sim_mode']:<6}{row['model']:<13}{row['rxn_id']:<6}"
            f"true {row['true_dG']:>7.3f}  fitted {row['fitted_dG']:>7.3f}"
        )
    sys.stdout.flush()


def main():
    """Fit both models to the mechanistic MIDs in SS and INST mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from the saved CSVs without refitting.",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=N_STARTS,
        help=f"Randomised starts per (mode, model) fit (default: {N_STARTS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 2),
        help="Worker processes (default: all CPUs but two).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Fit (standard fluxes): Mechanistic vs Aggregated Model (NE Thermodynamics)")
    print("=" * 80)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()

    if args.plot_only:
        df_summary, df_fluxes, df_residuals, _ = load_results()
    else:
        print(f"\nConfiguration:")
        print(f"  Flux map: {FLUX_FILE}")
        print(f"  Data: mechanistic model, FBA1 = {FBA1_DG:.3f}, FBA2 = {FBA2_DG:.3f} kJ/mol")
        print(f"  MDV sd: {MDV_SD} on every isotopologue")
        print(f"  INST timepoints: {[f'{t:g}' for t in INST_TIMEPOINTS]} s")
        print(f"  Measured fluxes (sd fraction): {MEASURED_FLUX_SD_FRAC}")
        print(
            f"  Starts: {args.starts} per fit, each at the true net and exchange fluxes "
            f"scaled by U({1 - PERTURB:g}, {1 + PERTURB:g})"
        )
        sys.stdout.flush()

        print("\nSimulating mechanistic data...")
        sys.stdout.flush()
        data = simulate_data()

        jobs = []
        for mode in ("inst", "ss"):  # INST tasks first: they are the slow ones
            chunk = STARTS_PER_TASK[mode]
            for model in FIT_MODELS:
                for i in range(0, args.starts, chunk):
                    start_ids = list(range(i, min(i + chunk, args.starts)))
                    jobs.append((mode, model, data[mode], start_ids))
        print(f"Fitting {len(MODES) * len(FIT_MODELS)} (mode, model) combinations "
              f"as {len(jobs)} tasks on {args.workers} workers...")
        sys.stdout.flush()
        if args.workers > 1:
            # One BLAS thread per worker so the processes do not oversubscribe the CPUs
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
                chunks = list(pool.map(fit_model, jobs))
        else:
            chunks = [fit_model(job) for job in jobs]
        fits = merge_chunks(chunks)

        print("\n" + "=" * 80)
        print("Saving Results")
        print("=" * 80)
        df_summary, df_fluxes, df_residuals, _ = save_results(fits)

    make_plots(df_summary, df_fluxes, df_residuals)
    print_summary(df_summary, df_fluxes)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fisher Information Matrix analysis for enzyme simulation.

Computes the FIM for three measurement scenarios using a single
experimental condition (default parameters):

1. No labeling:   fluxes + concentrations (4 observables)
2. Steady-state:  fluxes + concentrations + SS MIDs (11 observables)
3. Instationary:  fluxes + concentrations + time-course MIDs (39 observables)

The FIM is computed as J^T @ Sigma^{-1} @ J where J = d(predictions)/d(log_params)
is obtained via jax.jacrev (reverse-mode automatic differentiation).

Noise levels match the MCMC scripts:
  - Flux:           normal,    sigma = 0.1
  - Concentrations: lognormal, sigma = 0.01 (on log scale)
  - MIDs:           lognormal, sigma = 0.01 (on log scale)
"""

from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

try:
    from scripts.enzyme_utils import (
        DEFAULT_PARAMS,
        DEFAULT_INITIAL,
        MCMC_PARAM_NAMES,
        FORMATION_ENERGIES,
        TEMPERATURE,
        compute_wegscheider_constraints,
        compute_wegscheider_constraints_jax,
        simulate_enzyme_to_steady_state_jax,
        params_dict_to_array,
    )
    from scripts.emu_jax import (
        load_emu_network_structure,
        precompute_matrix_structure,
        simulate_ss_mdvs_pure_jax,
        simulate_inst_mdvs_pure_jax,
    )
    from scripts.style import apply_style, COLORS, SINGLE_COL, DOUBLE_COL
except ModuleNotFoundError:
    from enzyme_utils import (
        DEFAULT_PARAMS,
        DEFAULT_INITIAL,
        MCMC_PARAM_NAMES,
        FORMATION_ENERGIES,
        TEMPERATURE,
        compute_wegscheider_constraints,
        compute_wegscheider_constraints_jax,
        simulate_enzyme_to_steady_state_jax,
        params_dict_to_array,
    )
    from emu_jax import (
        load_emu_network_structure,
        precompute_matrix_structure,
        simulate_ss_mdvs_pure_jax,
        simulate_inst_mdvs_pure_jax,
    )
    from style import apply_style, COLORS, SINGLE_COL, DOUBLE_COL


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "fisher_information"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Noise standard deviations (matching MCMC scripts)
FLUX_STD_REL = 0.001  # flux_std = 0.001 * |ref_flux| (relative)
CONC_STD = 0.001      # lognormal on concentrations
MID_STD = 0.001       # lognormal on MIDs

# Observation times for instationary (matching enzyme_simulation_instationary.py)
OBSERVATION_TIMES = jnp.logspace(-1, 2, 5)
N_TIMEPOINTS = len(OBSERVATION_TIMES)
N_MIDS = 7  # A_12(3) + Q_1(2) + P_1(2)

FORMATION_ENERGIES_ARRAY = jnp.array([
    FORMATION_ENERGIES["A"],
    FORMATION_ENERGIES["P"],
    FORMATION_ENERGIES["Q"],
])


# =========================================================================
# EMU network
# =========================================================================

def get_emu_network():
    """Load EMU network structure and precompute matrix arrays."""
    network_path = PROJECT_ROOT / "models" / "simple_network" / "emu_network.json"
    network = load_emu_network_structure(network_path)
    precomputed = precompute_matrix_structure(network)
    return network, precomputed


# =========================================================================
# Helpers: flux/conc arrays for EMU simulation
# =========================================================================

def compute_fluxes_array(concs, params, network):
    """Build flux array in network order from concentrations and parameters."""
    A, Q, P = concs["A"], concs["Q"], concs["P"]
    EA, EPQ, EP, E = concs["EA"], concs["EPQ"], concs["EP"], concs["E"]
    fluxes_dict = {
        "v1": params["k_A_ex"] * params["A_e"],
        "v1_out": params["k_A_ex"] * A,
        "v2_f": A * E * params["k2f"],
        "v2_b": EA * params["k2r"],
        "v3_f": EA * params["k3f"],
        "v3_b": EPQ * params["k3r"],
        "v4_f": EPQ * params["k4f"],
        "v4_b": Q * EP * params["k4r"],
        "v5_f": EP * params["k5f"],
        "v5_b": E * P * params["k5r"],
        "vq_out": params["k_Q_ex"] * Q,
        "vq_in": params["k_Q_ex"] * params["Q_e"],
        "vp_out": params["k_P_ex"] * P,
        "vp_in": params["k_P_ex"] * params["P_e"],
        "vpq_f": P * params["k_pq_f"],
        "vpq_b": Q * params["k_pq_r"],
    }
    return jnp.array([fluxes_dict[fid] for fid in network["flux_ids_all"]])


def compute_concs_array(concs, network):
    """Build concentration array in network order."""
    return jnp.array([concs.get(mid, 1.0) for mid in network["metab_ids_all"]])


# =========================================================================
# Core: enzyme ODE to steady state
# =========================================================================

def run_enzyme_ode(log_params, A_e, E_T, Q_e, P_e, initial_array):
    """Run enzyme ODE to steady state from log-transformed MCMC parameters.

    Returns (flux, concs_dict, params).
    """
    param_array = jnp.exp(log_params)
    params = {name: param_array[i] for i, name in enumerate(MCMC_PARAM_NAMES)}
    params["A_e"] = A_e
    params["E_T"] = E_T
    params["Q_e"] = Q_e
    params["P_e"] = P_e

    k2f, k_pq_r = compute_wegscheider_constraints_jax(
        params, FORMATION_ENERGIES_ARRAY, TEMPERATURE
    )
    params["k2f"] = k2f
    params["k_pq_r"] = k_pq_r

    sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys

    A, Q, P = y_ss[0], y_ss[1], y_ss[2]
    EA, EPQ, EP = y_ss[3], y_ss[4], y_ss[5]
    E = E_T - EA - EPQ - EP

    flux = params["k_A_ex"] * (A_e - A)
    concs = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}
    return flux, concs, params


# =========================================================================
# Prediction functions  (log_params -> observable vector)
# =========================================================================

def predict_no_labeling(log_params, A_e, E_T, Q_e, P_e, initial_array):
    """Predict [flux, log(A), log(Q), log(P)]."""
    flux, concs, _ = run_enzyme_ode(log_params, A_e, E_T, Q_e, P_e, initial_array)
    return jnp.array([
        flux, jnp.log(concs["A"]), jnp.log(concs["Q"]), jnp.log(concs["P"]),
    ])


def predict_stationary(log_params, A_e, E_T, Q_e, P_e, initial_array,
                        network, precomputed):
    """Predict [flux, log(A), log(Q), log(P), log(7 MIDs)]."""
    flux, concs, params = run_enzyme_ode(
        log_params, A_e, E_T, Q_e, P_e, initial_array
    )
    fluxes_arr = compute_fluxes_array(concs, params, network)
    ss_mdvs = simulate_ss_mdvs_pure_jax(fluxes_arr, network, precomputed)
    mids = jnp.concatenate([ss_mdvs["A_12"], ss_mdvs["Q_1"], ss_mdvs["P_1"]])
    return jnp.concatenate([
        jnp.array([
            flux, jnp.log(concs["A"]), jnp.log(concs["Q"]), jnp.log(concs["P"]),
        ]),
        jnp.log(mids),
    ])


def predict_instationary(log_params, A_e, E_T, Q_e, P_e, initial_array,
                          network, precomputed):
    """Predict [flux, log(A), log(Q), log(P), log(7 MIDs x 5 timepoints)]."""
    flux, concs, params = run_enzyme_ode(
        log_params, A_e, E_T, Q_e, P_e, initial_array
    )
    fluxes_arr = compute_fluxes_array(concs, params, network)
    concs_arr = compute_concs_array(concs, network)

    parts = [jnp.array([
        flux, jnp.log(concs["A"]), jnp.log(concs["Q"]), jnp.log(concs["P"]),
    ])]
    for t in OBSERVATION_TIMES:
        timepoints = jnp.array([0.0, t])
        mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_arr, concs_arr, timepoints, network, precomputed
        )
        mids = jnp.concatenate([mdvs["A_12"], mdvs["Q_1"], mdvs["P_1"]])
        parts.append(jnp.log(mids))
    return jnp.concatenate(parts)


# =========================================================================
# Sigma (inverse-variance diagonal)
# =========================================================================

def build_sigma_inv_sq(n_mids_total, flux_std):
    """Build 1/sigma^2 vector: [flux, 3 log-concs, n_mids_total log-MIDs]."""
    parts = [
        jnp.array([1.0 / flux_std**2]),
        jnp.ones(3) / CONC_STD**2,
    ]
    if n_mids_total > 0:
        parts.append(jnp.ones(n_mids_total) / MID_STD**2)
    return jnp.concatenate(parts)


# =========================================================================
# FIM computation and analysis
# =========================================================================

def compute_fim(predict_fn, log_params, sigma_inv_sq, *args):
    """Compute FIM = J^T @ diag(1/sigma^2) @ J via reverse-mode AD."""
    J = jax.jacrev(predict_fn)(log_params, *args)
    return J.T @ jnp.diag(sigma_inv_sq) @ J


def find_identifiable_parameters(fim, param_names, threshold=1e-4):
    """Eigenvalue method to find maximal identifiable parameter subset.

    Iteratively removes the parameter most associated with the smallest
    eigenvalue until all remaining eigenvalues exceed the threshold.

    The FIM submatrix is normalised to correlation form (unit diagonal)
    at each iteration so that eigenvalues are scale-independent and the
    threshold has a consistent meaning across parameters and scenarios.

    Algorithm from Quaiser & Mönnigmann, BMC Syst Biol 2009 (page 6).

    Returns dict with keys:
      identifiable, unidentifiable, removal_eigenvalues, eigenvalue_trace
    where eigenvalue_trace[i] is the smallest eigenvalue when i parameters
    have been removed (includes the final above-threshold step).
    """
    fim_np = np.array(fim)
    remaining = list(range(len(param_names)))
    removed = []
    removal_eigenvalues = []
    eigenvalue_trace = []  # smallest eigenvalue at each iteration

    while remaining:
        idx = np.array(remaining)
        sub_fim = fim_np[np.ix_(idx, idx)]

        # Normalise to correlation form: F_norm = D^{-1/2} F D^{-1/2}
        d = np.sqrt(np.abs(np.diag(sub_fim)))
        d = np.where(d > 0, d, 1.0)
        norm_fim = sub_fim / np.outer(d, d)

        eigenvalues, eigenvectors = np.linalg.eigh(norm_fim)
        eigenvalue_trace.append(eigenvalues[0])

        if eigenvalues[0] >= threshold:
            break

        # Find parameter with largest component in smallest eigenvector
        worst = np.argmax(np.abs(eigenvectors[:, 0]))
        removal_eigenvalues.append(eigenvalues[0])
        removed.append(remaining.pop(worst))

    identifiable = [param_names[i] for i in remaining]
    unidentifiable = [param_names[i] for i in removed]
    return {
        "identifiable": identifiable,
        "unidentifiable": unidentifiable,
        "removal_eigenvalues": removal_eigenvalues,
        "eigenvalue_trace": eigenvalue_trace,
    }


def analyze_fim(fim, scenario_name):
    """Eigendecomposition and CRLB analysis."""
    fim_np = np.array(fim)
    eigenvalues = np.sort(np.linalg.eigvalsh(fim_np))[::-1]

    eig_pos = eigenvalues[eigenvalues > 1e-20]
    cond_number = eig_pos[0] / eig_pos[-1] if len(eig_pos) > 1 else float("inf")

    rank = np.linalg.matrix_rank(fim_np)
    if rank == fim_np.shape[0]:
        crlb = np.sqrt(np.diag(np.linalg.inv(fim_np)))
    else:
        crlb = np.full(fim_np.shape[0], np.inf)

    print(f"\n{'=' * 60}")
    print(f"FIM Analysis: {scenario_name}")
    print(f"{'=' * 60}")
    print(f"  Rank: {rank}/{fim_np.shape[0]}")
    print(f"  Eigenvalues (descending):")
    for i, ev in enumerate(eigenvalues):
        print(f"    lambda_{i + 1:2d} = {ev:.4e}")
    det = np.linalg.det(fim_np)
    print(f"  Condition number: {cond_number:.4e}")
    print(f"  det(FIM): {det:.4e}")
    if det != 0:
        print(f"  log10(|det(FIM)|): {np.log10(np.abs(det)):.2f}")
    if np.all(np.isfinite(crlb)):
        print(f"  CRLB (std dev of log-params):")
        for name, c in zip(MCMC_PARAM_NAMES, crlb):
            print(f"    {name:10s}: {c:.6f}")
    else:
        print(f"  FIM is singular — CRLB undefined")
    sys.stdout.flush()

    return {
        "eigenvalues": eigenvalues,
        "crlb": crlb,
        "condition_number": cond_number,
    }


# =========================================================================
# Plotting
# =========================================================================

def plot_eigenvalue_spectrum(results):
    """Grouped bar chart of eigenvalue spectra for all three scenarios."""
    apply_style()

    scenarios = ["No Labeling", "Steady-State", "Instationary"]
    n_params = len(MCMC_PARAM_NAMES)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.45))

    x = np.arange(n_params)
    width = 0.25

    for i, scenario in enumerate(scenarios):
        ev = results[scenario]["eigenvalues"]
        ev_plot = np.maximum(ev, 1e-30)
        ax.bar(
            x + (i - 1) * width,
            ev_plot,
            width,
            label=scenario,
            color=COLORS[scenario],
        )

    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_params)])
    ax.legend()

    return fig


def plot_fim_heatmaps(fim_nl, fim_ss, fim_inst):
    """Heatmaps of the three FIM matrices (symmetric log color scale)."""
    apply_style()

    fims = [np.array(fim_nl), np.array(fim_ss), np.array(fim_inst)]
    titles = ["No Labeling", "Steady-State", "Instationary"]
    labels = [n.replace("k_", "") for n in MCMC_PARAM_NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))

    for ax, fim, title in zip(axes, fims, titles):
        # Symmetric log scale: sign(x) * log10(1 + |x|)
        abs_fim = np.abs(fim)
        sign_fim = np.sign(fim)
        log_fim = sign_fim * np.log10(1.0 + abs_fim)

        vmax = np.max(np.abs(log_fim))
        im = ax.imshow(
            log_fim, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

    fig.colorbar(im, ax=axes, label=r"sign $\times$ log$_{10}$(1 + |FIM|)",
                 shrink=0.8, pad=0.02)
    return fig


def fim_to_correlation(fim):
    """Compute parameter correlation matrix from FIM.

    Uses FIM^{-1} as the covariance matrix (Cramér-Rao bound).
    Falls back to pseudo-inverse for singular FIMs.
    """
    fim_np = np.array(fim)
    rank = np.linalg.matrix_rank(fim_np)
    if rank == fim_np.shape[0]:
        cov = np.linalg.inv(fim_np)
    else:
        cov = np.linalg.pinv(fim_np)

    # Correlation: rho_ij = cov_ij / sqrt(cov_ii * cov_jj)
    std = np.sqrt(np.diag(cov))
    # Guard against zero diagonal entries
    std = np.where(std > 0, std, 1.0)
    corr = cov / np.outer(std, std)
    # Clamp to [-1, 1] for numerical safety
    return np.clip(corr, -1.0, 1.0)


def plot_correlation_heatmaps(fim_nl, fim_ss, fim_inst):
    """Heatmaps of parameter correlation matrices derived from FIM^{-1}."""
    apply_style()

    fims = [fim_nl, fim_ss, fim_inst]
    titles = ["No Labeling", "Steady-State", "Instationary"]
    labels = [n.replace("k_", "") for n in MCMC_PARAM_NAMES]
    n = len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))

    for ax, fim, title in zip(axes, fims, titles):
        corr = fim_to_correlation(fim)
        im = ax.imshow(
            corr, cmap="RdBu_r", vmin=-1, vmax=1,
            aspect="equal", interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)

    fig.colorbar(im, ax=axes, label="Correlation", shrink=0.8, pad=0.02)
    return fig


def plot_identifiability_summary(ident_results, threshold=1e-4):
    """Two-panel figure: eigenvalue progression (left) and grid summary (right).

    Left panel: smallest normalised eigenvalue vs number of removed parameters
    (after Quaiser & Mönnigmann, BMC Syst Biol 2009, Figure 1).
    Right panel: identifiable/unidentifiable grid with removal order.
    """
    apply_style()
    from matplotlib.patches import Patch, Rectangle
    from cmcrameri import cm

    scenarios = ["No Labeling", "Steady-State", "Instationary"]
    labels = [n.replace("k_", "") for n in MCMC_PARAM_NAMES]
    n_params = len(labels)
    n_scenarios = len(scenarios)

    fig, (ax_eig, ax_grid) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.45),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35},
    )

    # ---- Left panel: eigenvalue progression ----
    for scenario in scenarios:
        trace = ident_results[scenario]["eigenvalue_trace"]
        n_removed = np.arange(len(trace))
        # Clip small/negative values for log scale display
        trace_plot = np.maximum(np.abs(trace), 1e-20)
        ax_eig.plot(n_removed, trace_plot, "o-",
                    color=COLORS[scenario], label=scenario, markersize=4)

    ax_eig.axhline(threshold, color="0.4", ls="--", lw=0.8,
                    label=f"$\\varepsilon$ = {threshold:.0e}")
    ax_eig.set_yscale("log")
    ax_eig.set_xlabel("Number of parameters removed")
    ax_eig.set_ylabel("Smallest eigenvalue (normalised FIM)")
    ax_eig.set_xticks(range(n_params))
    ax_eig.legend(fontsize=6)

    # ---- Right panel: identifiability grid ----
    grid = np.zeros((n_params, n_scenarios))
    removal_order = np.full((n_params, n_scenarios), -1, dtype=int)

    # Use batlow endpoints for identifiable/unidentifiable
    color_ident = cm.batlow(0.85)
    color_unident = cm.batlow(0.15)

    for j, scenario in enumerate(scenarios):
        identifiable = ident_results[scenario]["identifiable"]
        unidentifiable = ident_results[scenario]["unidentifiable"]
        for name in identifiable:
            i = MCMC_PARAM_NAMES.index(name)
            grid[i, j] = 1.0
        for order, name in enumerate(unidentifiable):
            i = MCMC_PARAM_NAMES.index(name)
            removal_order[i, j] = order + 1

    for i in range(n_params):
        for j in range(n_scenarios):
            color = color_ident if grid[i, j] == 1.0 else color_unident
            ax_grid.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.5,
            ))
            if removal_order[i, j] > 0:
                ax_grid.text(j, i, str(removal_order[i, j]),
                             ha="center", va="center", fontsize=7,
                             color="white", fontweight="bold")

    ax_grid.set_xlim(-0.5, n_scenarios - 0.5)
    ax_grid.set_ylim(n_params - 0.5, -0.5)
    ax_grid.set_xticks(range(n_scenarios))
    ax_grid.set_xticklabels(scenarios, rotation=30, ha="right")
    ax_grid.set_yticks(range(n_params))
    ax_grid.set_yticklabels(labels)
    ax_grid.set_xlabel("Experiment type")
    ax_grid.set_ylabel("Parameter")
    ax_grid.set_aspect("auto")

    legend_elements = [
        Patch(facecolor=color_ident, label="Identifiable"),
        Patch(facecolor=color_unident, label="Unidentifiable"),
    ]
    ax_grid.legend(handles=legend_elements, fontsize=6,
                   loc="upper left", bbox_to_anchor=(1.02, 1.0))

    return fig


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 80)
    print("Fisher Information Matrix Analysis")
    print("=" * 80)
    sys.stdout.flush()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Single condition: default parameters with Wegscheider constraints
    params = compute_wegscheider_constraints(DEFAULT_PARAMS.copy())
    A_e = params["A_e"]
    E_T = params["E_T"]
    Q_e = params["Q_e"]
    P_e = params["P_e"]

    initial_array = jnp.array([
        DEFAULT_INITIAL["A"], DEFAULT_INITIAL["Q"], DEFAULT_INITIAL["P"],
        DEFAULT_INITIAL["EA"], DEFAULT_INITIAL["EPQ"], DEFAULT_INITIAL["EP"],
    ])

    log_params = jnp.log(params_dict_to_array(params))

    print(f"\nParameters: {MCMC_PARAM_NAMES}")
    print(f"Log-parameters: {log_params}")
    print(f"Condition: A_e={A_e}, E_T={E_T}, Q_e={Q_e}, P_e={P_e}")
    sys.stdout.flush()

    # Load EMU network
    print("\nLoading EMU network...")
    sys.stdout.flush()
    network, precomputed = get_emu_network()

    # Compute reference flux for relative flux_std
    ref_preds = predict_no_labeling(log_params, A_e, E_T, Q_e, P_e, initial_array)
    ref_flux = float(ref_preds[0])
    flux_std = FLUX_STD_REL * abs(ref_flux)
    print(f"Reference flux: {ref_flux:.6f}")
    print(f"Flux std (0.1% relative): {flux_std:.6e}")
    print(f"Conc std: {CONC_STD}")
    print(f"MID std:  {MID_STD}")
    sys.stdout.flush()

    # ------ No Labeling ------
    print("\nComputing FIM: No Labeling...")
    sys.stdout.flush()
    sigma_nl = build_sigma_inv_sq(0, flux_std)
    fim_nl = compute_fim(
        predict_no_labeling, log_params, sigma_nl,
        A_e, E_T, Q_e, P_e, initial_array,
    )
    res_nl = analyze_fim(fim_nl, "No Labeling")

    # ------ Steady-State ------
    print("\nComputing FIM: Steady-State...")
    sys.stdout.flush()
    sigma_ss = build_sigma_inv_sq(N_MIDS, flux_std)
    fim_ss = compute_fim(
        predict_stationary, log_params, sigma_ss,
        A_e, E_T, Q_e, P_e, initial_array, network, precomputed,
    )
    res_ss = analyze_fim(fim_ss, "Steady-State")

    # ------ Instationary ------
    print("\nComputing FIM: Instationary...")
    sys.stdout.flush()
    sigma_inst = build_sigma_inv_sq(N_MIDS * N_TIMEPOINTS, flux_std)
    fim_inst = compute_fim(
        predict_instationary, log_params, sigma_inst,
        A_e, E_T, Q_e, P_e, initial_array, network, precomputed,
    )
    res_inst = analyze_fim(fim_inst, "Instationary")

    # ---- Identifiability analysis ----
    print("\n" + "=" * 60)
    print("Identifiability Analysis (eigenvalue method)")
    print("=" * 60)
    sys.stdout.flush()

    ident_results = {}
    for name, fim in [("No Labeling", fim_nl), ("Steady-State", fim_ss),
                      ("Instationary", fim_inst)]:
        result = find_identifiable_parameters(fim, list(MCMC_PARAM_NAMES))
        ident_results[name] = result
        identifiable = result["identifiable"]
        unidentifiable = result["unidentifiable"]
        removal_eigs = result["removal_eigenvalues"]
        print(f"\n  {name}:")
        print(f"    Identifiable ({len(identifiable)}): {identifiable}")
        if unidentifiable:
            print(f"    Unidentifiable ({len(unidentifiable)}, removal order):")
            for i, (p, ev) in enumerate(zip(unidentifiable, removal_eigs)):
                print(f"      {i + 1}. {p}  (eigenvalue = {ev:.4e})")
        sys.stdout.flush()

    # Collect results
    all_results = {
        "No Labeling": res_nl,
        "Steady-State": res_ss,
        "Instationary": res_inst,
    }

    # Save
    np.savez(
        RESULTS_DIR / "fim_results.npz",
        param_names=MCMC_PARAM_NAMES,
        log_params=np.array(log_params),
        fim_no_labeling=np.array(fim_nl),
        fim_stationary=np.array(fim_ss),
        fim_instationary=np.array(fim_inst),
        eigenvalues_no_labeling=res_nl["eigenvalues"],
        eigenvalues_stationary=res_ss["eigenvalues"],
        eigenvalues_instationary=res_inst["eigenvalues"],
    )
    print(f"\nResults saved to {RESULTS_DIR / 'fim_results.npz'}")

    # Plot eigenvalue spectrum
    fig = plot_eigenvalue_spectrum(all_results)
    fig.savefig(RESULTS_DIR / "fim_eigenvalues.png")
    fig.savefig(FIGURES_DIR / "fim_eigenvalues.pdf")
    print(f"Figures saved to {RESULTS_DIR / 'fim_eigenvalues.png'}")
    print(f"                 {FIGURES_DIR / 'fim_eigenvalues.pdf'}")
    plt.close(fig)

    # Plot FIM heatmaps
    fig = plot_fim_heatmaps(fim_nl, fim_ss, fim_inst)
    fig.savefig(RESULTS_DIR / "fim_heatmaps.png")
    fig.savefig(FIGURES_DIR / "fim_heatmaps.pdf")
    print(f"Figures saved to {RESULTS_DIR / 'fim_heatmaps.png'}")
    print(f"                 {FIGURES_DIR / 'fim_heatmaps.pdf'}")
    plt.close(fig)

    # Plot correlation heatmaps
    fig = plot_correlation_heatmaps(fim_nl, fim_ss, fim_inst)
    fig.savefig(RESULTS_DIR / "fim_correlations.png")
    fig.savefig(FIGURES_DIR / "fim_correlations.pdf")
    print(f"Figures saved to {RESULTS_DIR / 'fim_correlations.png'}")
    print(f"                 {FIGURES_DIR / 'fim_correlations.pdf'}")
    plt.close(fig)

    # Plot identifiability summary
    fig = plot_identifiability_summary(ident_results)
    fig.savefig(RESULTS_DIR / "fim_identifiability.png")
    fig.savefig(FIGURES_DIR / "fim_identifiability.pdf")
    print(f"Figures saved to {RESULTS_DIR / 'fim_identifiability.png'}")
    print(f"                 {FIGURES_DIR / 'fim_identifiability.pdf'}")
    plt.close(fig)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

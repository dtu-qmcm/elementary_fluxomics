#!/usr/bin/env python3
"""Fisher Information Matrix (FIM) calculation for enzyme simulation experiments.

This script calculates the FIM for three enzyme simulation measurement experiments
at true parameter values to compare their information content:

1. No Labeling: Flux + concentrations only
2. Stationary: Flux + concentrations + steady-state MIDs
3. Instationary: Flux + concentrations + time-course MIDs

The FIM quantifies how much information measurements provide about parameters.
For Gaussian likelihoods: FIM = J^T * Sigma^{-1} * J
where J is the Jacobian of model outputs w.r.t. parameters.

For lognormal measurements, we compute Jacobian of log(prediction).
"""

from pathlib import Path
import sys

# Enable float64 precision BEFORE importing JAX
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import diffrax
from diffrax import SteadyStateEvent
import matplotlib.pyplot as plt

try:
    from scripts.isotopomer_simulation import enzyme_ode
    from scripts.emu_jax import (
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
        simulate_ss_mdvs_pure_jax,
        precompute_matrix_structure,
    )
except ModuleNotFoundError:
    from isotopomer_simulation import enzyme_ode
    from emu_jax import (
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
        simulate_ss_mdvs_pure_jax,
        precompute_matrix_structure,
    )


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "fisher_information"


# Default parameters (same as other enzyme simulation scripts)
DEFAULT_PARAMS = {
    "A_e": 5.0,
    "Q_e": 1.0,
    "P_e": 1.0,
    "k_A_ex": 0.1,
    "k_Q_ex": 0.1,
    "k_P_ex": 0.1,
    "k2f": 10.0,  # computed from constraints
    "k2r": 250.0,
    "k3f": 125.0,
    "k3r": 12.5,
    "k4f": 200.0,
    "k4r": 20.0,
    "k5f": 150.0,
    "k5r": 15.0,
    "k_pq_f": 50.0,
    "k_pq_r": 2.0,  # computed from constraints
    "E_T": 1.0,
}

DEFAULT_INITIAL = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
}

# Experimental conditions (same as other scripts)
np.random.seed(12345)
EXPERIMENTAL_CONDITIONS = [
    {"E_T": float(E_T), "A_e": float(A_e)}
    for E_T, A_e in zip(
        np.exp(np.random.normal(1, 0.5, 10)),
        np.exp(np.random.normal(1, 0.5, 10)),
    )
]

# Thermodynamic constants
R_GAS = 8.314e-3
TEMPERATURE = 298.15

FORMATION_ENERGIES = {
    "A": 0.0,
    "P": 0.5,
    "Q": -2.5,
}

# Observation times for instationary (same as enzyme_simulation_instationary.py)
OBSERVATION_TIMES = jnp.logspace(-0.5, 0.5, 5)

# MCMC parameter names (11 parameters)
MCMC_PARAM_NAMES = [
    "k2r", "k3f", "k3r", "k4f", "k4r", "k5f", "k5r",
    "k_A_ex", "k_Q_ex", "k_P_ex", "k_pq_f"
]

# Noise standard deviations
FLUX_STD = 0.1
CONC_STD = 0.1  # log-space
MID_STD = 0.01  # log-space

# Global cache for EMU network
_PURE_JAX_EMU_NETWORK = None


def get_pure_jax_emu_network():
    """Get or initialize the pure JAX EMU network structure."""
    global _PURE_JAX_EMU_NETWORK

    if _PURE_JAX_EMU_NETWORK is None:
        emu_network_path = PROJECT_ROOT / "models" / "simple_network" / "emu_network.json"
        network = load_emu_network_structure(emu_network_path)
        precomputed = precompute_matrix_structure(network)
        _PURE_JAX_EMU_NETWORK = (network, precomputed)

    return _PURE_JAX_EMU_NETWORK


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def compute_wegscheider_constraints_jax(params, formation_energies_array, temperature):
    """JAX-compatible Wegscheider constraint computation."""
    RT = R_GAS * temperature
    mu_A, mu_P, mu_Q = formation_energies_array[0], formation_energies_array[1], formation_energies_array[2]

    delta_G_enzyme = mu_P + mu_Q - mu_A
    K_enzyme = jnp.exp(-delta_G_enzyme / RT)

    delta_G_pq = mu_Q - mu_P
    K_pq = jnp.exp(-delta_G_pq / RT)

    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

    k_pq_r = params["k_pq_f"] / K_pq

    return k2f, k_pq_r


def params_array_to_dict(param_array, A_e, E_T, Q_e, P_e):
    """Convert parameter array to full parameter dictionary."""
    params = {
        "k2r": param_array[0],
        "k3f": param_array[1],
        "k3r": param_array[2],
        "k4f": param_array[3],
        "k4r": param_array[4],
        "k5f": param_array[5],
        "k5r": param_array[6],
        "k_A_ex": param_array[7],
        "k_Q_ex": param_array[8],
        "k_P_ex": param_array[9],
        "k_pq_f": param_array[10],
        "A_e": A_e,
        "Q_e": Q_e,
        "P_e": P_e,
        "E_T": E_T,
    }

    # Compute dependent rate constants
    formation_energies_array = jnp.array([
        FORMATION_ENERGIES["A"],
        FORMATION_ENERGIES["P"],
        FORMATION_ENERGIES["Q"],
    ])
    k2f, k_pq_r = compute_wegscheider_constraints_jax(params, formation_energies_array, TEMPERATURE)
    params["k2f"] = k2f
    params["k_pq_r"] = k_pq_r

    return params


def params_dict_to_array(params):
    """Extract MCMC parameters from dictionary to array."""
    return jnp.array([params[name] for name in MCMC_PARAM_NAMES])


def simulate_enzyme_to_steady_state_jax(params, initial_array):
    """JAX-compatible enzyme ODE simulation to steady state."""
    term = diffrax.ODETerm(enzyme_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    discrete_terminating_event = SteadyStateEvent(rtol=1e-4, atol=1e-6)
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=jnp.inf,
        dt0=1e-10,
        y0=initial_array,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
        discrete_terminating_event=discrete_terminating_event,
        throw=False,
    )

    return sol


def compute_fluxes_array_pure_jax(concs, params, network):
    """Compute flux array for pure JAX EMU simulation."""
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    E = concs["E"]

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

    return jnp.array([fluxes_dict.get(fid, 0.0) for fid in network["flux_ids_all"]])


def compute_concs_array_pure_jax(concs, network):
    """Compute concentration array for pure JAX EMU simulation."""
    return jnp.array([concs.get(mid, 1.0) for mid in network["metab_ids_all"]])


def extract_measurable_mids_from_emu(emu_mdvs):
    """Extract measurable MIDs (A, Q, P) from EMU results."""
    A_mid = emu_mdvs["A_12"]  # size 3
    Q_mid = emu_mdvs["Q_1"]   # size 2
    P_mid = emu_mdvs["P_1"]   # size 2
    return jnp.concatenate([A_mid, Q_mid, P_mid])


# =============================================================================
# Prediction Functions for Each Experiment
# =============================================================================

def predict_no_labeling(param_array, A_e, E_T, Q_e, P_e, initial_array):
    """Predict measurements for no-labeling experiment.

    Returns: [flux, log(A), log(Q), log(P)] (4 values)
    """
    params = params_array_to_dict(param_array, A_e, E_T, Q_e, P_e)

    # Run enzyme ODE to steady state
    sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys

    A, Q, P = y_ss[0], y_ss[1], y_ss[2]

    # Net flux
    flux = params["k_A_ex"] * (A_e - A)

    # Log concentrations (for lognormal likelihood)
    log_A = jnp.log(jnp.maximum(A, 1e-10))
    log_Q = jnp.log(jnp.maximum(Q, 1e-10))
    log_P = jnp.log(jnp.maximum(P, 1e-10))

    return jnp.array([flux, log_A, log_Q, log_P])


def predict_stationary(param_array, A_e, E_T, Q_e, P_e, initial_array, network, precomputed):
    """Predict measurements for stationary labeling experiment.

    Returns: [flux, log(A), log(Q), log(P), log(MIDs...)] (11 values)
    """
    params = params_array_to_dict(param_array, A_e, E_T, Q_e, P_e)

    # Run enzyme ODE to steady state
    sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys

    A, Q, P = y_ss[0], y_ss[1], y_ss[2]
    EA, EPQ, EP = y_ss[3], y_ss[4], y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    # Build concentration dict for EMU
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    # Net flux
    flux = params["k_A_ex"] * (A_e - A)

    # Log concentrations
    log_A = jnp.log(jnp.maximum(A, 1e-10))
    log_Q = jnp.log(jnp.maximum(Q, 1e-10))
    log_P = jnp.log(jnp.maximum(P, 1e-10))

    # Steady-state EMU simulation
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)
    emu_mdvs = simulate_ss_mdvs_pure_jax(fluxes_array, network, precomputed)
    mids = extract_measurable_mids_from_emu(emu_mdvs)

    # Log MIDs
    log_mids = jnp.log(jnp.maximum(mids, 1e-10))

    return jnp.concatenate([jnp.array([flux, log_A, log_Q, log_P]), log_mids])


def predict_instationary(param_array, A_e, E_T, Q_e, P_e, initial_array,
                         observation_times, network, precomputed):
    """Predict measurements for instationary labeling experiment.

    Returns: [flux, log(A), log(Q), log(P), log(MIDs at t1), ..., log(MIDs at t5)] (39 values)
    """
    params = params_array_to_dict(param_array, A_e, E_T, Q_e, P_e)

    # Run enzyme ODE to steady state
    sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys

    A, Q, P = y_ss[0], y_ss[1], y_ss[2]
    EA, EPQ, EP = y_ss[3], y_ss[4], y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    # Build concentration dict for EMU
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    # Net flux
    flux = params["k_A_ex"] * (A_e - A)

    # Log concentrations
    log_A = jnp.log(jnp.maximum(A, 1e-10))
    log_Q = jnp.log(jnp.maximum(Q, 1e-10))
    log_P = jnp.log(jnp.maximum(P, 1e-10))

    # Compute fluxes array once
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)
    concs_array = compute_concs_array_pure_jax(concs_dict, network)

    # Collect MIDs at each observation time
    all_log_mids = []
    for t in observation_times:
        t_float = float(t)
        timepoints_to_t = jnp.array([0.0, t_float])
        emu_mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, network, precomputed
        )
        mids = extract_measurable_mids_from_emu(emu_mdvs)
        log_mids = jnp.log(jnp.maximum(mids, 1e-10))
        all_log_mids.append(log_mids)

    # Stack all time points
    all_log_mids_array = jnp.concatenate(all_log_mids)

    return jnp.concatenate([jnp.array([flux, log_A, log_Q, log_P]), all_log_mids_array])


# =============================================================================
# Fisher Information Matrix Computation
# =============================================================================

def build_sigma_inv_no_labeling():
    """Build inverse covariance matrix for no-labeling experiment.

    Measurements: [flux, log(A), log(Q), log(P)]
    """
    sigmas = jnp.array([FLUX_STD, CONC_STD, CONC_STD, CONC_STD])
    return jnp.diag(1.0 / sigmas**2)


def build_sigma_inv_stationary():
    """Build inverse covariance matrix for stationary experiment.

    Measurements: [flux, log(A), log(Q), log(P), log(7 MIDs)]
    """
    sigmas = jnp.concatenate([
        jnp.array([FLUX_STD]),           # flux
        jnp.ones(3) * CONC_STD,          # 3 concentrations
        jnp.ones(7) * MID_STD,           # 7 MIDs (A:3, Q:2, P:2)
    ])
    return jnp.diag(1.0 / sigmas**2)


def build_sigma_inv_instationary(n_timepoints):
    """Build inverse covariance matrix for instationary experiment.

    Measurements: [flux, log(A), log(Q), log(P), log(7 MIDs at each of n_timepoints)]
    """
    sigmas = jnp.concatenate([
        jnp.array([FLUX_STD]),           # flux
        jnp.ones(3) * CONC_STD,          # 3 concentrations
        jnp.ones(7 * n_timepoints) * MID_STD,  # 7 MIDs at each timepoint
    ])
    return jnp.diag(1.0 / sigmas**2)


def compute_fim_single_condition(jacobian, sigma_inv):
    """Compute FIM for a single condition.

    FIM = J^T * Sigma^{-1} * J
    """
    return jacobian.T @ sigma_inv @ jacobian


def compute_fim_no_labeling(param_array, conditions, Q_e, P_e, initial_array):
    """Compute total FIM for no-labeling experiment across all conditions."""
    n_params = len(param_array)
    total_fim = jnp.zeros((n_params, n_params))
    sigma_inv = build_sigma_inv_no_labeling()

    for cond in conditions:
        A_e = cond["A_e"]
        E_T = cond["E_T"]

        # Compute Jacobian using JAX autodiff
        def predict_fn(params):
            return predict_no_labeling(params, A_e, E_T, Q_e, P_e, initial_array)

        jacobian = jax.jacobian(predict_fn)(param_array)  # Shape: (n_measurements, n_params)
        fim_condition = compute_fim_single_condition(jacobian, sigma_inv)
        total_fim = total_fim + fim_condition

    return total_fim


def compute_fim_stationary(param_array, conditions, Q_e, P_e, initial_array, network, precomputed):
    """Compute total FIM for stationary labeling experiment across all conditions."""
    n_params = len(param_array)
    total_fim = jnp.zeros((n_params, n_params))
    sigma_inv = build_sigma_inv_stationary()

    for cond in conditions:
        A_e = cond["A_e"]
        E_T = cond["E_T"]

        def predict_fn(params):
            return predict_stationary(params, A_e, E_T, Q_e, P_e, initial_array, network, precomputed)

        jacobian = jax.jacobian(predict_fn)(param_array)
        fim_condition = compute_fim_single_condition(jacobian, sigma_inv)
        total_fim = total_fim + fim_condition

    return total_fim


def compute_fim_instationary(param_array, conditions, Q_e, P_e, initial_array,
                             observation_times, network, precomputed):
    """Compute total FIM for instationary labeling experiment across all conditions."""
    n_params = len(param_array)
    n_timepoints = len(observation_times)
    total_fim = jnp.zeros((n_params, n_params))
    sigma_inv = build_sigma_inv_instationary(n_timepoints)

    for cond in conditions:
        A_e = cond["A_e"]
        E_T = cond["E_T"]

        def predict_fn(params):
            return predict_instationary(params, A_e, E_T, Q_e, P_e, initial_array,
                                        observation_times, network, precomputed)

        jacobian = jax.jacobian(predict_fn)(param_array)
        fim_condition = compute_fim_single_condition(jacobian, sigma_inv)
        total_fim = total_fim + fim_condition

    return total_fim


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_fim(fim, param_names, experiment_name):
    """Analyze FIM and print summary statistics."""
    print(f"\n{'='*80}")
    print(f"Fisher Information Matrix Analysis: {experiment_name}")
    print(f"{'='*80}")

    # Check symmetry
    symmetry_error = jnp.max(jnp.abs(fim - fim.T))
    print(f"\nSymmetry check (max |FIM - FIM^T|): {symmetry_error:.2e}")

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(fim)
    eigenvalues = jnp.real(eigenvalues)

    print(f"\nEigenvalues (sorted):")
    sorted_indices = jnp.argsort(eigenvalues)[::-1]
    for i, idx in enumerate(sorted_indices):
        print(f"  λ_{i+1}: {eigenvalues[idx]:12.6e}")

    # Condition number
    min_eig = jnp.min(eigenvalues)
    max_eig = jnp.max(eigenvalues)
    if min_eig > 0:
        cond_number = max_eig / min_eig
        print(f"\nCondition number: {cond_number:.2e}")
    else:
        print(f"\nCondition number: infinite (min eigenvalue = {min_eig:.2e})")

    # Total information metrics
    trace_fim = jnp.trace(fim)
    det_fim = jnp.linalg.det(fim)
    print(f"\nTrace(FIM): {trace_fim:.6e}")
    print(f"Det(FIM): {det_fim:.6e}")
    if det_fim > 0:
        log_det = jnp.log(det_fim)
        print(f"Log-det(FIM): {log_det:.6f}")

    # Diagonal elements (marginal information per parameter)
    print(f"\nDiagonal elements (marginal parameter information):")
    for i, name in enumerate(param_names):
        print(f"  {name:10s}: {fim[i, i]:12.6e}")

    # CRLB (Cramér-Rao Lower Bounds) if FIM is invertible
    try:
        fim_inv = jnp.linalg.inv(fim)
        print(f"\nCramér-Rao Lower Bounds (sqrt of diagonal of FIM^-1):")
        for i, name in enumerate(param_names):
            crlb = jnp.sqrt(fim_inv[i, i])
            print(f"  {name:10s}: {crlb:12.6e}")
    except Exception as e:
        print(f"\nFIM not invertible: {e}")

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "trace": trace_fim,
        "det": det_fim,
        "condition_number": cond_number if min_eig > 0 else jnp.inf,
    }


def plot_fim_comparison(fim_no_label, fim_stationary, fim_instationary, param_names, output_dir):
    """Plot FIM comparison between experiments."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Heatmaps
    fims = [fim_no_label, fim_stationary, fim_instationary]
    titles = ["No Labeling", "Stationary", "Instationary"]

    for ax, fim, title in zip(axes, fims, titles):
        # Log-scale for better visualization
        fim_log = jnp.log10(jnp.abs(fim) + 1e-20)
        im = ax.imshow(fim_log, cmap='viridis', aspect='equal')
        ax.set_xticks(range(len(param_names)))
        ax.set_yticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(param_names, fontsize=8)
        ax.set_title(f"{title}\nlog₁₀(|FIM|)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / "fim_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Eigenvalue comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    eigs_no_label = jnp.sort(jnp.linalg.eigvalsh(fim_no_label))[::-1]
    eigs_stationary = jnp.sort(jnp.linalg.eigvalsh(fim_stationary))[::-1]
    eigs_instationary = jnp.sort(jnp.linalg.eigvalsh(fim_instationary))[::-1]

    x = np.arange(len(param_names))
    width = 0.25

    ax.bar(x - width, np.log10(np.maximum(eigs_no_label, 1e-20)), width, label='No Labeling')
    ax.bar(x, np.log10(np.maximum(eigs_stationary, 1e-20)), width, label='Stationary')
    ax.bar(x + width, np.log10(np.maximum(eigs_instationary, 1e-20)), width, label='Instationary')

    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('log₁₀(Eigenvalue)')
    ax.set_title('FIM Eigenvalues Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'λ_{i+1}' for i in range(len(param_names))])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fim_eigenvalues.png", dpi=150, bbox_inches='tight')
    plt.close()

    # CRLB comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        crlb_no_label = jnp.sqrt(jnp.diag(jnp.linalg.inv(fim_no_label)))
        crlb_stationary = jnp.sqrt(jnp.diag(jnp.linalg.inv(fim_stationary)))
        crlb_instationary = jnp.sqrt(jnp.diag(jnp.linalg.inv(fim_instationary)))

        x = np.arange(len(param_names))
        width = 0.25

        ax.bar(x - width, np.log10(crlb_no_label), width, label='No Labeling')
        ax.bar(x, np.log10(crlb_stationary), width, label='Stationary')
        ax.bar(x + width, np.log10(crlb_instationary), width, label='Instationary')

        ax.set_xlabel('Parameter')
        ax.set_ylabel('log₁₀(CRLB)')
        ax.set_title('Cramér-Rao Lower Bounds Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "fim_crlb.png", dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"Could not compute CRLB: {e}")
    finally:
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def main():
    print("=" * 80)
    print("Fisher Information Matrix Calculation")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()

    # Get true parameters
    true_params = DEFAULT_PARAMS.copy()
    param_array = params_dict_to_array(true_params)

    print(f"\nTrue parameter values:")
    for name, val in zip(MCMC_PARAM_NAMES, param_array):
        print(f"  {name:10s}: {val:.4f}")
    sys.stdout.flush()

    # Initial conditions
    initial_array = jnp.array([
        DEFAULT_INITIAL["A"], DEFAULT_INITIAL["Q"], DEFAULT_INITIAL["P"],
        DEFAULT_INITIAL["EA"], DEFAULT_INITIAL["EPQ"], DEFAULT_INITIAL["EP"]
    ])

    # Fixed parameters
    Q_e = DEFAULT_PARAMS["Q_e"]
    P_e = DEFAULT_PARAMS["P_e"]

    # Load EMU network
    print("\nLoading EMU network...")
    sys.stdout.flush()
    network, precomputed = get_pure_jax_emu_network()
    print(f"  EMU network loaded with {len(network['flux_ids_all'])} fluxes")
    sys.stdout.flush()

    # Compute FIM for each experiment
    print("\n" + "=" * 80)
    print("Computing Fisher Information Matrices")
    print("=" * 80)

    # No Labeling
    print("\nComputing FIM for No Labeling experiment...")
    sys.stdout.flush()
    fim_no_label = compute_fim_no_labeling(
        param_array, EXPERIMENTAL_CONDITIONS, Q_e, P_e, initial_array
    )
    results_no_label = analyze_fim(fim_no_label, MCMC_PARAM_NAMES, "No Labeling")

    # Stationary
    print("\nComputing FIM for Stationary Labeling experiment...")
    sys.stdout.flush()
    fim_stationary = compute_fim_stationary(
        param_array, EXPERIMENTAL_CONDITIONS, Q_e, P_e, initial_array, network, precomputed
    )
    results_stationary = analyze_fim(fim_stationary, MCMC_PARAM_NAMES, "Stationary Labeling")

    # Instationary
    print("\nComputing FIM for Instationary Labeling experiment...")
    sys.stdout.flush()
    fim_instationary = compute_fim_instationary(
        param_array, EXPERIMENTAL_CONDITIONS, Q_e, P_e, initial_array,
        OBSERVATION_TIMES, network, precomputed
    )
    results_instationary = analyze_fim(fim_instationary, MCMC_PARAM_NAMES, "Instationary Labeling")

    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary Comparison")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'No Labeling':>15} {'Stationary':>15} {'Instationary':>15}")
    print("-" * 70)
    print(f"{'Trace(FIM)':<25} {results_no_label['trace']:>15.4e} {results_stationary['trace']:>15.4e} {results_instationary['trace']:>15.4e}")
    print(f"{'Det(FIM)':<25} {results_no_label['det']:>15.4e} {results_stationary['det']:>15.4e} {results_instationary['det']:>15.4e}")
    print(f"{'Condition Number':<25} {results_no_label['condition_number']:>15.4e} {results_stationary['condition_number']:>15.4e} {results_instationary['condition_number']:>15.4e}")
    print(f"{'Max Eigenvalue':<25} {jnp.max(results_no_label['eigenvalues']):>15.4e} {jnp.max(results_stationary['eigenvalues']):>15.4e} {jnp.max(results_instationary['eigenvalues']):>15.4e}")
    print(f"{'Min Eigenvalue':<25} {jnp.min(results_no_label['eigenvalues']):>15.4e} {jnp.min(results_stationary['eigenvalues']):>15.4e} {jnp.min(results_instationary['eigenvalues']):>15.4e}")

    # Measurement counts
    print(f"\n{'Measurements/Condition':<25} {'4':>15} {'11':>15} {'39':>15}")
    print(f"{'Total Measurements':<25} {'40':>15} {'110':>15} {'390':>15}")

    # Information gain relative to no labeling
    print(f"\n{'Information Gain Relative to No Labeling':}")
    print(f"  Stationary trace ratio:    {results_stationary['trace'] / results_no_label['trace']:.2f}x")
    print(f"  Instationary trace ratio:  {results_instationary['trace'] / results_no_label['trace']:.2f}x")
    if results_no_label['det'] > 0 and results_stationary['det'] > 0:
        print(f"  Stationary det ratio:      {results_stationary['det'] / results_no_label['det']:.2e}x")
    if results_no_label['det'] > 0 and results_instationary['det'] > 0:
        print(f"  Instationary det ratio:    {results_instationary['det'] / results_no_label['det']:.2e}x")

    # Generate plots
    print("\nGenerating comparison plots...")
    sys.stdout.flush()
    plot_fim_comparison(fim_no_label, fim_stationary, fim_instationary, MCMC_PARAM_NAMES, output_dir)

    # Save results
    np.savez(
        output_dir / "fim_results.npz",
        fim_no_label=np.array(fim_no_label),
        fim_stationary=np.array(fim_stationary),
        fim_instationary=np.array(fim_instationary),
        param_names=MCMC_PARAM_NAMES,
        param_values=np.array(param_array),
        eigenvalues_no_label=np.array(results_no_label['eigenvalues']),
        eigenvalues_stationary=np.array(results_stationary['eigenvalues']),
        eigenvalues_instationary=np.array(results_instationary['eigenvalues']),
    )
    print(f"\nResults saved to {output_dir / 'fim_results.npz'}")

    print("\n" + "=" * 80)
    print("Fisher Information Analysis Complete!")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Enzyme kinetics ODE simulation with INSTATIONARY isotopomer labeling analysis.

This script performs MCMC parameter estimation using time-course MID data instead
of steady-state MIDs. The system starts at metabolic steady state but isotopically
unlabeled, then labeled substrate is introduced at t=0 and MIDs are measured at
multiple time points during the transient.

Key differences from enzyme_simulation.py (steady-state):
- Isotopomer simulation saves at specified time points, not just steady state
- Likelihood uses time-course MIDs at multiple observation times
- More information content from transient dynamics

Labeling:
  - A_e (external A): 100% labeled on both atoms (isotopomer A_11)
  - Q_e, P_e (external Q, P): 100% unlabeled

Inference approach:
  1. Run enzyme_ode to get metabolic steady-state fluxes and concentrations
  2. Run isotopomer_ode from that steady state (all unlabeled) for time-course MIDs
  3. Use fluxes/concentrations from step 1 and time-course MIDs from step 2 in log-likelihood

Only A, Q, P MIDs are used (measurable metabolites), not enzyme intermediates.
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
import blackjax

try:
    from scripts.isotopomer_simulation import isotopomer_ode, enzyme_ode
except ModuleNotFoundError:
    from isotopomer_simulation import isotopomer_ode, enzyme_ode


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
REACTIONS_FILE = PROJECT_ROOT / "models" / "simple_network" / "reactions.tsv"
RESULTS_DIR = PROJECT_ROOT / "results" / "enzyme_simulation_instationary"


# Observation time points for instationary labeling
OBSERVATION_TIMES = jnp.array([0.5, 2.0, 10.0, 50.0])
N_TIMEPOINTS = len(OBSERVATION_TIMES)

# Number of MID values per time point (A: 3, Q: 2, P: 2 = 7 total)
N_MIDS = 7


# Default parameters
DEFAULT_PARAMS = {
    # External pool concentrations (fixed boundary conditions)
    "A_e": 5.0,          # external A concentration
    "Q_e": 1.0,          # external Q concentration
    "P_e": 1.0,          # external P concentration
    # Exchange rate constants (symmetric mass-action: v = k * (X_e - X))
    "k_A_ex": 0.1,       # A exchange rate constant
    "k_Q_ex": 0.1,       # Q exchange rate constant
    "k_P_ex": 0.1,       # P exchange rate constant
    # Enzyme kinetic rate constants
    "k2f": 10.0,         # A + E -> EA forward (computed from constraints)
    "k2r": 250.0,        # EA -> A + E reverse (25x)
    "k3f": 125.0,        # EA -> EPQ forward (25x)
    "k3r": 12.5,         # EPQ -> EA reverse (25x)
    "k4f": 200.0,        # EPQ -> EP + Q forward (25x)
    "k4r": 20.0,         # EP + Q -> EPQ reverse (25x)
    "k5f": 150.0,        # EP -> E + P forward (25x)
    "k5r": 15.0,         # E + P -> EP reverse (25x)
    "k_pq_f": 50.0,      # P -> Q forward rate constant (25x)
    "k_pq_r": 2.0,       # Q -> P reverse rate constant (computed from constraints)
    "E_T": 1.0,          # total enzyme concentration
}

# Initial conditions (state order: A, Q, P, EA, EPQ, EP)
DEFAULT_INITIAL = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
}

# Experimental conditions for multi-condition fitting
# Fixed seed ensures identical conditions across all scripts
np.random.seed(12345)
EXPERIMENTAL_CONDITIONS = [
    {"E_T": float(E_T), "A_e": float(A_e)}
    for E_T, A_e in zip(
        np.random.uniform(0.5, 2.0, 5),  # E_T varies from 0.5 to 2.0
        np.random.uniform(2.0, 10.0, 5),  # A_e varies from 2.0 to 10.0
    )
]
NUM_CONDITIONS = len(EXPERIMENTAL_CONDITIONS)

# Thermodynamic constants
R_GAS = 8.314e-3  # Gas constant in kJ/(mol·K)
TEMPERATURE = 298.15  # Temperature in K

# Formation energies (chemical potentials) in kJ/mol
FORMATION_ENERGIES = {
    "A": 0.0,    # Reference state
    "P": 0.5,    # kJ/mol
    "Q": -2.5,   # kJ/mol
}


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def compute_wegscheider_constraints(params, formation_energies=None, temperature=None):
    """Compute dependent rate constants from Wegscheider conditions."""
    if formation_energies is None:
        formation_energies = FORMATION_ENERGIES
    if temperature is None:
        temperature = TEMPERATURE

    RT = R_GAS * temperature

    # Gibbs energy for overall enzyme reaction: A → P + Q
    delta_G_enzyme = (
        formation_energies["P"] + formation_energies["Q"] - formation_energies["A"]
    )
    K_enzyme = np.exp(-delta_G_enzyme / RT)

    # Gibbs energy for P → Q interconversion
    delta_G_pq = formation_energies["Q"] - formation_energies["P"]
    K_pq = np.exp(-delta_G_pq / RT)

    # Solve for k2f from Haldane relation
    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

    # Solve for k_pq_r from P↔Q equilibrium
    k_pq_r = params["k_pq_f"] / K_pq

    updated_params = params.copy()
    updated_params["k2f"] = k2f
    updated_params["k_pq_r"] = k_pq_r

    return updated_params


def compute_wegscheider_constraints_jax(params, formation_energies_array, temperature):
    """JAX-compatible version of Wegscheider constraint computation."""
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


def compute_E(EA, EPQ, EP, E_T):
    """Compute free enzyme from conservation constraint."""
    return E_T - EA - EPQ - EP


def simulate_enzyme_to_steady_state_jax(params, initial_array):
    """JAX-compatible enzyme ODE simulation to steady state.

    Uses enzyme_ode (total concentrations without isotopomers) to find
    metabolic steady state. This gives the fluxes and concentrations for
    the log-likelihood calculation.

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    initial_array : jnp.array
        Initial concentrations [A, Q, P, EA, EPQ, EP]

    Returns
    -------
    diffrax.Solution
        ODE solution at steady state
    """
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


def simulate_isotopomers_timecourse(params, initial_total_array, observation_times):
    """Simulate isotopomer dynamics and return states at specified times.

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    initial_total_array : jnp.array
        Initial total concentrations [A, Q, P, EA, EPQ, EP]
    observation_times : jnp.array
        Time points at which to observe MIDs

    Returns
    -------
    diffrax.Solution
        ODE solution with states at observation times
    """
    # Build initial isotopomer state (all unlabeled at t=0)
    y0 = jnp.zeros(18)
    # A starts unlabeled (A_00 at index 0)
    y0 = y0.at[0].set(initial_total_array[0])
    # EA starts unlabeled (EA_00 at index 4)
    y0 = y0.at[4].set(initial_total_array[3])
    # EPQ starts unlabeled (EPQ_00 at index 8)
    y0 = y0.at[8].set(initial_total_array[4])
    # EP starts unlabeled (EP_0 at index 12)
    y0 = y0.at[12].set(initial_total_array[5])
    # Q starts unlabeled (Q_0 at index 14)
    y0 = y0.at[14].set(initial_total_array[1])
    # P starts unlabeled (P_0 at index 16)
    y0 = y0.at[16].set(initial_total_array[2])

    term = diffrax.ODETerm(isotopomer_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Save at specified observation times
    saveat = diffrax.SaveAt(ts=observation_times)

    t_final = observation_times[-1]

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_final,
        dt0=1e-10,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
        throw=False,
    )

    return sol


def compute_mids_measurable_jax(y):
    """JAX-compatible MID computation for measurable metabolites only (A, Q, P).

    Parameters
    ----------
    y : jnp.array
        Isotopomer state vector (18 components)

    Returns
    -------
    jnp.array
        MID array (7,): [A_M0, A_M1, A_M2, Q_M0, Q_M1, P_M0, P_M1]
    """
    # A: 2 atoms, MID has 3 components (M+0, M+1, M+2)
    A_00, A_01, A_10, A_11 = y[0], y[1], y[2], y[3]
    A_total = A_00 + A_01 + A_10 + A_11
    A_mid = jnp.array([A_00, A_01 + A_10, A_11]) / jnp.maximum(A_total, 1e-10)

    # Q: 1 atom, MID has 2 components (M+0, M+1)
    Q_0, Q_1 = y[14], y[15]
    Q_total = Q_0 + Q_1
    Q_mid = jnp.array([Q_0, Q_1]) / jnp.maximum(Q_total, 1e-10)

    # P: 1 atom, MID has 2 components (M+0, M+1)
    P_0, P_1 = y[16], y[17]
    P_total = P_0 + P_1
    P_mid = jnp.array([P_0, P_1]) / jnp.maximum(P_total, 1e-10)

    # Concatenate: A (3) + Q (2) + P (2) = 7 total
    return jnp.concatenate([A_mid, Q_mid, P_mid])


def compute_rates(y, params):
    """Compute all reaction rates from state and parameters."""
    A, Q, P, EA, EPQ, EP = y
    E = compute_E(EA, EPQ, EP, params["E_T"])

    # Exchange reactions with external pools (symmetric mass-action)
    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)
    v_Q_ex = params["k_Q_ex"] * (params["Q_e"] - Q)
    v_P_ex = params["k_P_ex"] * (params["P_e"] - P)

    # Enzyme-catalyzed reactions
    v2 = A * E * params["k2f"] - EA * params["k2r"]
    v3 = EA * params["k3f"] - EPQ * params["k3r"]
    v4 = EPQ * params["k4f"] - Q * EP * params["k4r"]
    v5 = EP * params["k5f"] - E * P * params["k5r"]

    # P <-> Q interconversion
    v_pq = P * params["k_pq_f"] - Q * params["k_pq_r"]

    return {
        "v_A_ex": v_A_ex, "v2": v2, "v3": v3, "v4": v4, "v5": v5,
        "v_Q_ex": v_Q_ex, "v_P_ex": v_P_ex, "v_pq": v_pq
    }


def ode_system(t, y, args):
    """ODE right-hand side for diffrax (total concentrations)."""
    params = args
    rates = compute_rates(y, params)

    dA_dt = rates["v_A_ex"] - rates["v2"]
    dQ_dt = rates["v4"] + rates["v_Q_ex"] + rates["v_pq"]
    dP_dt = rates["v5"] + rates["v_P_ex"] - rates["v_pq"]
    dEA_dt = rates["v2"] - rates["v3"]
    dEPQ_dt = rates["v3"] - rates["v4"]
    dEP_dt = rates["v4"] - rates["v5"]

    return jnp.array([dA_dt, dQ_dt, dP_dt, dEA_dt, dEPQ_dt, dEP_dt])


def simulate_ode_to_steady_state(params, initial):
    """Simulate total concentration ODE to steady state."""
    params = compute_wegscheider_constraints(params)

    y0 = jnp.array([initial[k] for k in ["A", "Q", "P", "EA", "EPQ", "EP"]])

    term = diffrax.ODETerm(ode_system)
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
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
        discrete_terminating_event=discrete_terminating_event,
    )

    return sol


# Parameter names that will be sampled
MCMC_PARAM_NAMES = [
    "k2r", "k3f", "k3r", "k4f", "k4r", "k5f", "k5r",
    "k_A_ex", "k_Q_ex", "k_P_ex", "k_pq_f"
]


def params_array_to_dict(param_array, fixed_params):
    """Convert parameter array to full parameter dictionary."""
    params = fixed_params.copy()
    for i, name in enumerate(MCMC_PARAM_NAMES):
        params[name] = param_array[i]
    return params


def params_dict_to_array(params):
    """Extract MCMC parameters from dictionary to array."""
    return jnp.array([params[name] for name in MCMC_PARAM_NAMES])


def generate_reference_data_single_condition(params, initial, observation_times):
    """Generate reference observations including time-course MIDs.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run isotopomer_ode from that steady state (all unlabeled) for time-course MIDs

    The fluxes and concentrations come from enzyme_ode (step 1).
    The time-course MIDs come from isotopomer_ode (step 2) with A_e = A_11 (fully labeled).

    Parameters
    ----------
    params : dict
        Full parameter dictionary
    initial : dict
        Initial conditions
    observation_times : jnp.array
        Time points for MID measurements

    Returns
    -------
    dict
        Reference data containing fluxes, concentrations, timecourse_mids
    """
    # Apply constraints to get updated params
    params = compute_wegscheider_constraints(params)

    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"]
    ])

    # Step 1: Run enzyme_ode to get metabolic steady state
    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Extract concentrations from enzyme_ode steady state
    A, Q, P = float(enzyme_y_ss[0]), float(enzyme_y_ss[1]), float(enzyme_y_ss[2])
    EA, EPQ, EP = float(enzyme_y_ss[3]), float(enzyme_y_ss[4]), float(enzyme_y_ss[5])
    E = params["E_T"] - EA - EPQ - EP

    concs = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    # Exchange fluxes with external pools from enzyme_ode steady state
    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)
    v_Q_ex = params["k_Q_ex"] * (params["Q_e"] - Q)
    v_P_ex = params["k_P_ex"] * (params["P_e"] - P)

    net_rates = {
        "v_A_ex": v_A_ex,
        "v2": A * E * params["k2f"] - EA * params["k2r"],
        "v3": EA * params["k3f"] - EPQ * params["k3r"],
        "v4": EPQ * params["k4f"] - Q * EP * params["k4r"],
        "v5": EP * params["k5f"] - E * P * params["k5r"],
        "v_Q_ex": v_Q_ex,
        "v_P_ex": v_P_ex,
        "v_pq": P * params["k_pq_f"] - Q * params["k_pq_r"],
        "v_net": v_A_ex,
    }

    # Step 2: Run isotopomer_ode time-course from enzyme_ode steady state (all unlabeled)
    # This simulates the labeling experiment with A_e = A_11 (fully labeled)
    iso_sol = simulate_isotopomers_timecourse(params, enzyme_y_ss, observation_times)

    # Extract MIDs at each time point
    timecourse_mids = {}
    for t_idx in range(len(observation_times)):
        t = float(observation_times[t_idx])
        y_t = iso_sol.ys[t_idx]
        mids_t = compute_mids_measurable_jax(y_t)
        timecourse_mids[t] = np.array(mids_t)

    return {
        "fluxes": net_rates,
        "concentrations": concs,
        "timecourse_mids": timecourse_mids,
        "enzyme_steady_state": np.array(enzyme_y_ss),  # For initializing isotopomer sim
    }


def generate_reference_data_multi_condition(base_params, initial, conditions, observation_times):
    """Generate reference data for multiple experimental conditions."""
    ref_data_list = []
    for cond in conditions:
        params = base_params.copy()
        params["A_e"] = cond["A_e"]
        params["E_T"] = cond["E_T"]
        ref_data = generate_reference_data_single_condition(params, initial, observation_times)
        ref_data_list.append(ref_data)
    return ref_data_list


def compute_condition_likelihood(
    A_e,
    E_T,
    ref_flux,
    ref_concs,
    ref_mids_timecourse,
    param_array,
    Q_e,
    P_e,
    initial_array,
    observation_times,
):
    """Compute log-likelihood for a single condition with timecourse MIDs.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run isotopomer_ode from that steady state (all unlabeled) for time-course MIDs

    Fluxes and concentrations are from enzyme_ode (step 1).
    Time-course MIDs are from isotopomer_ode (step 2) with A_e = A_11 (fully labeled).

    Parameters
    ----------
    A_e : float
        External A concentration for this condition
    E_T : float
        Total enzyme concentration for this condition
    ref_flux : float
        Reference net flux value
    ref_concs : jnp.array
        Reference concentration values (3,): [A, Q, P]
    ref_mids_timecourse : jnp.array
        Reference MID values (n_timepoints, 7)
    param_array : jnp.array
        Rate constants (11,)
    Q_e : float
        External Q concentration (fixed)
    P_e : float
        External P concentration (fixed)
    initial_array : jnp.array
        Initial conditions for enzyme_ode (6,): [A, Q, P, EA, EPQ, EP]
    observation_times : jnp.array
        Observation time points

    Returns
    -------
    float
        Log-likelihood for this condition
    """
    # Build parameter dict
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

    # Step 1: Run enzyme_ode to get metabolic steady state
    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Check if enzyme solver failed
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    # Extract concentrations from enzyme_ode steady state (for flux/conc likelihood)
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    sim_concs = jnp.array([A, Q, P])

    # Net flux from enzyme_ode steady state
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution)
    flux_std = 0.1
    flux_residual = (ref_flux - sim_flux) / flux_std
    ll_flux = -0.5 * flux_residual**2 - jnp.log(flux_std * jnp.sqrt(2 * jnp.pi))

    # Log-likelihood for concentrations (lognormal)
    conc_std = 0.1
    log_ref_concs = jnp.log(jnp.maximum(ref_concs, 1e-10))
    log_sim_concs = jnp.log(jnp.maximum(sim_concs, 1e-10))
    conc_residuals = (log_ref_concs - log_sim_concs) / conc_std
    ll_concs = jnp.sum(
        -0.5 * conc_residuals**2 - jnp.log(conc_std * jnp.sqrt(2 * jnp.pi))
    )

    # Step 2: Run isotopomer_ode time-course from enzyme_ode steady state (all unlabeled)
    iso_sol = simulate_isotopomers_timecourse(params, enzyme_y_ss, observation_times)

    # Check if isotopomer solver failed
    iso_failed = jnp.any(jnp.isnan(iso_sol.ys)) | (iso_sol.result != diffrax.RESULTS.successful)

    # Log-likelihood for time-course MIDs (lognormal)
    mid_std = 0.01

    # Loop over time points
    def compute_timepoint_ll(t_idx):
        y_t = iso_sol.ys[t_idx]
        sim_mids_t = compute_mids_measurable_jax(y_t)
        ref_mids_t = ref_mids_timecourse[t_idx]

        log_ref = jnp.log(jnp.maximum(ref_mids_t, 1e-10))
        log_sim = jnp.log(jnp.maximum(sim_mids_t, 1e-10))
        residuals = (log_ref - log_sim) / mid_std
        return jnp.sum(
            -0.5 * residuals**2 - jnp.log(mid_std * jnp.sqrt(2 * jnp.pi))
        )

    # Use fori_loop for JAX-compatible iteration
    def body_fn(t_idx, ll_acc):
        return ll_acc + compute_timepoint_ll(t_idx)

    ll_mids = jax.lax.fori_loop(0, len(observation_times), body_fn, 0.0)

    ll_condition = ll_flux + ll_concs + ll_mids

    # Return -inf if either solver failed
    solver_failed = enzyme_failed | iso_failed
    return jnp.where(solver_failed, -jnp.inf, ll_condition)


def compute_log_posterior_jax(
    log_param_array,
    fixed_params_constant,
    fixed_params_varying,
    initial_array,
    ref_flux_array,
    ref_concs_arrays,
    ref_mids_arrays,
    observation_times,
    prior_mean,
    prior_std,
):
    """JAX-compatible log-posterior for multi-condition MCMC with timecourse MIDs.

    Parameters
    ----------
    log_param_array : jnp.array
        Log-transformed MCMC parameters
    fixed_params_constant : jnp.array
        Constant fixed parameters [Q_e, P_e]
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (n_conditions, 2)
    initial_array : jnp.array
        Initial conditions for enzyme_ode (6,): [A, Q, P, EA, EPQ, EP]
    ref_flux_array : jnp.array
        Reference net flux values (n_conditions,)
    ref_concs_arrays : jnp.array
        Reference concentrations (n_conditions, 3)
    ref_mids_arrays : jnp.array
        Reference MIDs (n_conditions, n_timepoints, 7)
    observation_times : jnp.array
        Observation time points (n_timepoints,)
    prior_mean : jnp.array
        Prior mean (log-scale)
    prior_std : float
        Prior std (log-scale)

    Returns
    -------
    float
        Log-posterior value
    """
    # Transform from log-space
    param_array = jnp.exp(log_param_array)

    # Normal prior on log-parameters
    log_prior = jnp.sum(
        -0.5 * ((log_param_array - prior_mean) / prior_std) ** 2
        - jnp.log(prior_std * jnp.sqrt(2 * jnp.pi))
    )

    Q_e = fixed_params_constant[0]
    P_e = fixed_params_constant[1]

    # vmap over conditions
    condition_lls = jax.vmap(
        lambda A_e, E_T, rf, rc, rm: compute_condition_likelihood(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e,
            initial_array, observation_times
        )
    )(
        fixed_params_varying[:, 0],
        fixed_params_varying[:, 1],
        ref_flux_array,
        ref_concs_arrays,
        ref_mids_arrays,
    )

    total_log_likelihood = jnp.sum(condition_lls)

    return log_prior + total_log_likelihood


def run_mcmc(
    ref_params,
    initial,
    conditions=None,
    observation_times=None,
    n_warmup=50,
    n_samples=100,
    seed=42,
):
    """Run MCMC sampling with instationary labeling data.

    Parameters
    ----------
    ref_params : dict
        Reference parameter set
    initial : dict
        Initial conditions
    conditions : list of dict, optional
        Experimental conditions
    observation_times : jnp.array, optional
        Observation time points
    n_warmup : int
        Number of warmup iterations
    n_samples : int
        Number of samples to collect
    seed : int
        Random seed

    Returns
    -------
    dict
        MCMC results
    """
    if conditions is None:
        conditions = EXPERIMENTAL_CONDITIONS
    if observation_times is None:
        observation_times = OBSERVATION_TIMES

    num_conditions = len(conditions)
    n_timepoints = len(observation_times)

    print("=" * 80)
    print(f"Running MCMC with INSTATIONARY isotope labeling ({num_conditions} conditions)")
    print(f"Observation times: {observation_times}")
    print(f"MIDs per time point: {N_MIDS} (A: 3, Q: 2, P: 2)")
    print("=" * 80)
    print(f"\nExperimental conditions:")
    for i, cond in enumerate(conditions):
        print(f"  Condition {i+1}: E_T={cond['E_T']:.4f}, A_e={cond['A_e']:.4f}")
    sys.stdout.flush()

    # Fixed parameters
    fixed_params_constant = jnp.array([
        ref_params["Q_e"],
        ref_params["P_e"],
    ])

    fixed_params_varying = jnp.array([
        [cond["A_e"], cond["E_T"]] for cond in conditions
    ])

    # Initial conditions array (same for all conditions)
    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"]
    ])

    # Generate reference data
    print("\nGenerating reference data for all conditions...")
    sys.stdout.flush()
    ref_data_list = generate_reference_data_multi_condition(
        ref_params, initial, conditions, observation_times
    )

    # Stack reference data
    ref_flux_array = jnp.array([
        rd["fluxes"]["v_net"] for rd in ref_data_list
    ])

    ref_concs_arrays = jnp.array([
        [rd["concentrations"]["A"], rd["concentrations"]["Q"], rd["concentrations"]["P"]]
        for rd in ref_data_list
    ])

    # Stack timecourse MIDs: (n_conditions, n_timepoints, 7)
    ref_mids_arrays = jnp.array([
        jnp.array([
            rd["timecourse_mids"][float(t)]
            for t in observation_times
        ])
        for rd in ref_data_list
    ])

    print(f"Reference data shapes:")
    print(f"  ref_flux_array: {ref_flux_array.shape}")
    print(f"  ref_concs_arrays: {ref_concs_arrays.shape}")
    print(f"  ref_mids_arrays: {ref_mids_arrays.shape}")
    print(f"  initial_array: {initial_array.shape}")
    sys.stdout.flush()

    # Prior
    param_array_ref = params_dict_to_array(ref_params)
    prior_mean = jnp.log(param_array_ref)
    prior_std = 0.5

    # Log-posterior function
    def logdensity_fn(log_param_array):
        return compute_log_posterior_jax(
            log_param_array,
            fixed_params_constant,
            fixed_params_varying,
            initial_array,
            ref_flux_array,
            ref_concs_arrays,
            ref_mids_arrays,
            observation_times,
            prior_mean,
            prior_std,
        )

    # Initialize
    rng_key = jax.random.PRNGKey(seed)
    initial_position = jnp.log(param_array_ref)

    print(f"\nInitial position (log-space): {initial_position}")
    print(f"Initial log-posterior: {logdensity_fn(initial_position)}")
    sys.stdout.flush()

    # NUTS warmup
    print(f"\nRunning {n_warmup} warmup iterations...")
    sys.stdout.flush()

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        num_steps=n_warmup,
        progress_bar=True,
    )

    rng_key, warmup_key = jax.random.split(rng_key)
    state, kernel, *_ = warmup.run(warmup_key, initial_position)

    print(f"Warmup complete.")
    sys.stdout.flush()

    # Sampling
    print(f"\nCollecting {n_samples} samples...")
    sys.stdout.flush()

    samples = []
    log_probs = []

    for i in range(n_samples):
        rng_key, sample_key = jax.random.split(rng_key)
        state, info = kernel(sample_key, state)
        samples.append(state.position)
        log_probs.append(-state.potential_energy)

        if (i + 1) % 100 == 0:
            print(f"  Collected {i + 1}/{n_samples} samples")
            sys.stdout.flush()

    log_samples = jnp.stack(samples)
    log_probs = jnp.array(log_probs)
    samples = jnp.exp(log_samples)

    # Summary
    print("\n" + "=" * 80)
    print("MCMC Results Summary (Instationary Labeling)")
    print("=" * 80)

    results = {
        "samples": samples,
        "log_samples": log_samples,
        "log_probs": log_probs,
        "param_names": MCMC_PARAM_NAMES,
        "reference": param_array_ref,
    }

    print(f"\nParameter estimates (mean ± std, natural scale):")
    for i, name in enumerate(MCMC_PARAM_NAMES):
        mean = float(jnp.mean(samples[:, i]))
        std = float(jnp.std(samples[:, i]))
        true_val = float(param_array_ref[i])
        print(f"  {name:10s}: {mean:8.4f} ± {std:6.4f}  (true: {true_val:.4f})")

    sys.stdout.flush()

    return results


def main():
    print("=" * 80)
    print("Enzyme Kinetics with INSTATIONARY Isotope Labeling Analysis")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()
    params = DEFAULT_PARAMS.copy()
    initial = DEFAULT_INITIAL.copy()

    # Apply thermodynamic constraints
    print("\nApplying Wegscheider constraints...")
    params = compute_wegscheider_constraints(params)
    print(f"  k2f = {params['k2f']:.6f}")
    print(f"  k_pq_r = {params['k_pq_r']:.6f}")
    sys.stdout.flush()

    # Generate reference data for first condition
    print("\nGenerating reference data (single condition)...")
    ref_data = generate_reference_data_single_condition(params, initial, OBSERVATION_TIMES)

    print("\nTime-course MIDs (A, Q, P only):")
    for t, mids in ref_data["timecourse_mids"].items():
        print(f"  t={t:5.1f}: A_MID={mids[:3]}, Q_MID={mids[3:5]}, P_MID={mids[5:7]}")
    sys.stdout.flush()

    # Run MCMC
    print("\n" + "=" * 80)
    print("Running MCMC Parameter Estimation")
    print("=" * 80)
    sys.stdout.flush()

    mcmc_results = run_mcmc(
        ref_params=DEFAULT_PARAMS,
        initial=DEFAULT_INITIAL,
    )

    # Save MCMC results
    np.savez(
        output_dir / "mcmc_results.npz",
        samples=mcmc_results["samples"],
        log_samples=mcmc_results["log_samples"],
        log_probs=mcmc_results["log_probs"],
        param_names=mcmc_results["param_names"],
        reference=mcmc_results["reference"],
    )
    print(f"\nResults saved to {output_dir / 'mcmc_results.npz'}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Simulation Complete!")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

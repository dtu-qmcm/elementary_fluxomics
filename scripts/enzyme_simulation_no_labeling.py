#!/usr/bin/env python3
"""Enzyme kinetics ODE simulation without isotope labeling.

This script simulates an enzyme-catalyzed reaction mechanism with external pool
exchange kinetics:

dA/dt = v_A_ex - v2        (exchange with external A pool, binding to enzyme)
dQ/dt = v4 + v_Q_ex + v_pq (produced by enzyme, exchange with external Q pool)
dP/dt = v5 + v_P_ex - v_pq (produced by enzyme, exchange with external P pool)
dEA/dt = v2 - v3
dEPQ/dt = v3 - v4
dEP/dt = v4 - v5

Exchange reactions use symmetric mass-action kinetics:
  v_X_ex = k_X_ex * (X_e - X)   where X_e is the fixed external concentration

Enzyme conservation: E + EA + EPQ + EP = E_T
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


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "enzyme_simulation_no_labeling"


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
# These constrain the Gibbs energies of reactions via:
#   ΔG°(A → P + Q) = μ_P + μ_Q - μ_A = -2 kJ/mol
#   ΔG°(P → Q) = μ_Q - μ_P = -3 kJ/mol
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
    """Compute dependent rate constants from Wegscheider conditions.

    The Wegscheider condition ensures thermodynamic consistency by requiring
    that equilibrium constants around reaction cycles satisfy:
        K_cycle = exp(-ΔG°_cycle / RT)

    For the enzyme mechanism (A → P + Q through enzyme intermediates):
        K_enzyme = (k2f · k3f · k4f · k5f) / (k2r · k3r · k4r · k5r) = exp(-ΔG°/RT)
        where ΔG° = μ_P + μ_Q - μ_A

    For the P ⇌ Q interconversion:
        K_pq = k_pq_f / k_pq_r = exp(-ΔG°_pq/RT)
        where ΔG°_pq = μ_Q - μ_P

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants (k2f and k_pq_r will be overwritten)
    formation_energies : dict, optional
        Formation energies for A, P, Q in kJ/mol. Defaults to FORMATION_ENERGIES.
    temperature : float, optional
        Temperature in K. Defaults to TEMPERATURE.

    Returns
    -------
    dict
        Updated parameter dictionary with k2f and k_pq_r computed from constraints
    """
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

    # Solve for k2f from Haldane relation (enzyme cycle constraint)
    # K_enzyme = (k2f · k3f · k4f · k5f) / (k2r · k3r · k4r · k5r)
    # k2f = K_enzyme · (k2r · k3r · k4r · k5r) / (k3f · k4f · k5f)
    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

    # Solve for k_pq_r from P↔Q equilibrium constraint
    # K_pq = k_pq_f / k_pq_r
    # k_pq_r = k_pq_f / K_pq
    k_pq_r = params["k_pq_f"] / K_pq

    # Create updated params dict
    updated_params = params.copy()
    updated_params["k2f"] = k2f
    updated_params["k_pq_r"] = k_pq_r

    return updated_params


def compute_wegscheider_constraints_jax(params, formation_energies_array, temperature):
    """JAX-compatible version of Wegscheider constraint computation.

    Parameters
    ----------
    params : dict
        Parameter dictionary (JAX-traced values)
    formation_energies_array : jnp.array
        Formation energies [μ_A, μ_P, μ_Q] in kJ/mol
    temperature : float
        Temperature in K

    Returns
    -------
    tuple
        (k2f, k_pq_r) computed from thermodynamic constraints
    """
    RT = R_GAS * temperature
    mu_A, mu_P, mu_Q = formation_energies_array[0], formation_energies_array[1], formation_energies_array[2]

    # Gibbs energy for overall enzyme reaction: A → P + Q
    delta_G_enzyme = mu_P + mu_Q - mu_A
    K_enzyme = jnp.exp(-delta_G_enzyme / RT)

    # Gibbs energy for P → Q interconversion
    delta_G_pq = mu_Q - mu_P
    K_pq = jnp.exp(-delta_G_pq / RT)

    # Solve for k2f from Haldane relation
    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

    # Solve for k_pq_r from P↔Q equilibrium
    k_pq_r = params["k_pq_f"] / K_pq

    return k2f, k_pq_r


def compute_E(EA, EPQ, EP, E_T):
    """Compute free enzyme from conservation constraint."""
    return E_T - EA - EPQ - EP


def compute_rates(y, params):
    """Compute all reaction rates from state and parameters.

    Exchange reactions use symmetric mass-action kinetics:
    v_X_ex = k_X_ex * (X_e - X)
    Positive v_X_ex means net influx, negative means net efflux.
    """
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
    """ODE right-hand side for diffrax.

    State variables: [A, Q, P, EA, EPQ, EP]

    Mass balances:
    - A: exchange with external pool, consumed by enzyme binding
    - Q: produced by v4, exchange with external pool, interconversion with P
    - P: produced by v5, exchange with external pool, interconversion with Q
    - EA, EPQ, EP: enzyme intermediates
    """
    params = args
    rates = compute_rates(y, params)

    # A: exchange with external pool, consumed by enzyme binding
    dA_dt = rates["v_A_ex"] - rates["v2"]

    # Q: produced by v4, exchange with external pool, interconversion with P
    dQ_dt = rates["v4"] + rates["v_Q_ex"] + rates["v_pq"]

    # P: produced by v5, exchange with external pool, interconversion with Q
    dP_dt = rates["v5"] + rates["v_P_ex"] - rates["v_pq"]

    # Enzyme intermediates (unchanged)
    dEA_dt = rates["v2"] - rates["v3"]
    dEPQ_dt = rates["v3"] - rates["v4"]
    dEP_dt = rates["v4"] - rates["v5"]

    return jnp.array([dA_dt, dQ_dt, dP_dt, dEA_dt, dEPQ_dt, dEP_dt])


def simulate_ode_to_steady_state(params, initial):
    """Simulate the ODE system to steady state using diffrax event detection.

    Wegscheider thermodynamic constraints are always applied to compute
    k2f and k_pq_r from the formation energies before simulation.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    initial : dict
        Initial conditions

    Returns
    -------
    diffrax.Solution
        ODE solution at steady state
    """
    # Apply thermodynamic constraints to compute dependent rate constants
    params = compute_wegscheider_constraints(params)

    y0 = jnp.array([initial[k] for k in ["A", "Q", "P", "EA", "EPQ", "EP"]])

    term = diffrax.ODETerm(ode_system)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Steady-state event: terminates when norm(f) < atol + rtol * norm(y)
    discrete_terminating_event = SteadyStateEvent(rtol=1e-4, atol=1e-6)

    # Save only the final state
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=jnp.inf,  # Run until event triggers
        dt0=1e-10,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
        discrete_terminating_event=discrete_terminating_event,
        throw=False,
    )

    return sol


def simulate_ode_to_steady_state_jax(params, initial_array):
    """JAX-compatible ODE simulation to steady state.

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
    term = diffrax.ODETerm(ode_system)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Steady-state event: terminates when norm(f) < atol + rtol * norm(y)
    discrete_terminating_event = SteadyStateEvent(rtol=1e-4, atol=1e-6)

    # Save only the final state
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=jnp.inf,  # Run until event triggers
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


def get_steady_state(sol, params):
    """Extract steady-state concentrations and fluxes."""
    # With SaveAt(t1=True), sol.ys is (1, 6) - extract the single row
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys
    concentrations = {
        "A": float(y_ss[0]),
        "Q": float(y_ss[1]),
        "P": float(y_ss[2]),
        "EA": float(y_ss[3]),
        "EPQ": float(y_ss[4]),
        "EP": float(y_ss[5]),
        "E": float(compute_E(y_ss[3], y_ss[4], y_ss[5], params["E_T"])),
    }

    rates = compute_rates(y_ss, params)

    return concentrations, rates


def compute_forward_backward_fluxes(concs, params):
    """Compute forward and backward fluxes for each reversible reaction."""
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    E = concs["E"]

    fluxes = {
        # Exchange reactions (net flux)
        "v_A_ex": params["k_A_ex"] * (params["A_e"] - A),
        "v_Q_ex": params["k_Q_ex"] * (params["Q_e"] - Q),
        "v_P_ex": params["k_P_ex"] * (params["P_e"] - P),
        # Enzyme reactions (forward and backward)
        "v2_f": A * E * params["k2f"],
        "v2_b": EA * params["k2r"],
        "v3_f": EA * params["k3f"],
        "v3_b": EPQ * params["k3r"],
        "v4_f": EPQ * params["k4f"],
        "v4_b": Q * EP * params["k4r"],
        "v5_f": EP * params["k5f"],
        "v5_b": E * P * params["k5r"],
        # P <-> Q interconversion
        "vpq_f": P * params["k_pq_f"],
        "vpq_b": Q * params["k_pq_r"],
    }

    return fluxes


def compute_log_likelihood(
    obs_fluxes,
    sim_fluxes,
    obs_concentrations,
    sim_concentrations,
    flux_std=0.1,
    conc_std=0.1,
):
    """Compute total log-likelihood of observations given simulated values.

    Parameters
    ----------
    obs_fluxes : dict
        Observed flux values {flux_id: value}
    sim_fluxes : dict
        Simulated flux values {flux_id: value}
    obs_concentrations : dict
        Observed concentration values {metabolite_id: value}
    sim_concentrations : dict
        Simulated concentration values {metabolite_id: value}
    flux_std : float
        Standard deviation for flux normal distribution (default 0.1)
    conc_std : float
        Standard deviation for concentration lognormal distribution (default 0.1)

    Returns
    -------
    dict
        Log-likelihoods for each component and total
    """
    log_likelihood = {
        "fluxes": 0.0,
        "concentrations": 0.0,
        "total": 0.0,
    }

    # 1. Fluxes: Normal distribution
    for flux_id in obs_fluxes:
        if flux_id in sim_fluxes:
            obs = obs_fluxes[flux_id]
            sim = sim_fluxes[flux_id]
            # Normal log-likelihood: -0.5 * ((obs - sim) / sigma)^2 - log(sigma * sqrt(2*pi))
            residual = (obs - sim) / flux_std
            log_likelihood["fluxes"] += -0.5 * residual**2 - jnp.log(flux_std * jnp.sqrt(2 * jnp.pi))

    # 2. Concentrations: Lognormal distribution
    # If X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma)
    # We compare log(obs) to log(sim) with given std_dev
    for met_id in obs_concentrations:
        if met_id in sim_concentrations:
            obs = obs_concentrations[met_id]
            sim = sim_concentrations[met_id]
            if obs > 0 and sim > 0:
                log_obs = jnp.log(obs)
                log_sim = jnp.log(sim)
                residual = (log_obs - log_sim) / conc_std
                log_likelihood["concentrations"] += -0.5 * residual**2 - jnp.log(conc_std * jnp.sqrt(2 * jnp.pi))
                # Jacobian term for lognormal: -log(obs)
                log_likelihood["concentrations"] += -jnp.log(obs)

    # Total log-likelihood
    log_likelihood["total"] = (
        log_likelihood["fluxes"]
        + log_likelihood["concentrations"]
    )

    return log_likelihood


# Parameter names that will be sampled (independent rate constants only)
# k2f and k_pq_r are computed from Wegscheider thermodynamic constraints
# A_e, Q_e, P_e, E_T are fixed (experimental conditions)
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


def generate_reference_data_single_condition(params, initial):
    """Generate reference observations from a parameter set for a single condition.

    Only includes experimentally measurable quantities:
    - Concentrations: A, Q, P (not enzyme intermediates)
    - Flux: net flux through the system (v_A_ex at steady state)

    Parameters
    ----------
    params : dict
        Full parameter dictionary
    initial : dict
        Initial conditions

    Returns
    -------
    dict
        Reference data containing fluxes and concentrations
    """
    # Run ODE simulation to steady state (constraints applied inside)
    sol = simulate_ode_to_steady_state(params, initial)
    # When using SaveAt(t1=True), sol.ys is (1, 6) - extract the single row
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys

    # Re-apply constraints to get updated params for flux calculation
    params = compute_wegscheider_constraints(params)

    # Extract measurable concentrations only (A, Q, P)
    A, Q, P = float(y_ss[0]), float(y_ss[1]), float(y_ss[2])
    concs = {"A": A, "Q": Q, "P": P}

    # Compute net flux at steady state (exchange flux = net throughput)
    v_net = params["k_A_ex"] * (params["A_e"] - A)

    return {
        "fluxes": {"v_net": v_net},
        "concentrations": concs,
    }


def generate_reference_data_multi_condition(base_params, initial, conditions):
    """Generate reference data for multiple experimental conditions.

    Parameters
    ----------
    base_params : dict
        Base parameter dictionary (A_e, E_T will be overridden per condition)
    initial : dict
        Initial conditions (same across all conditions)
    conditions : list of dict
        Experimental conditions with keys 'A_e', 'E_T'

    Returns
    -------
    list of dict
        Reference datasets, one per condition
    """
    ref_data_list = []
    for cond in conditions:
        params = base_params.copy()
        params["A_e"] = cond["A_e"]
        params["E_T"] = cond["E_T"]
        ref_data = generate_reference_data_single_condition(params, initial)
        ref_data_list.append(ref_data)
    return ref_data_list


def compute_condition_likelihood(
    A_e,
    E_T,
    ref_flux,
    ref_concs,
    param_array,
    Q_e,
    P_e,
    initial_array,
):
    """Compute log-likelihood for a single experimental condition.

    Only uses experimentally measurable quantities:
    - Concentrations: A, Q, P (not enzyme intermediates)
    - Flux: net flux through the system

    Parameters
    ----------
    A_e : float
        External A concentration for this condition
    E_T : float
        Total enzyme concentration for this condition
    ref_flux : float
        Reference net flux value (v_net at steady state)
    ref_concs : jnp.array
        Reference concentration values (3,): [A, Q, P]
    param_array : jnp.array
        Rate constants (11,): [k2r, k3f, k3r, k4f, k4r, k5f, k5r, k_A_ex, k_Q_ex, k_P_ex, k_pq_f]
    Q_e : float
        External Q concentration (fixed)
    P_e : float
        External P concentration (fixed)
    initial_array : jnp.array
        Initial conditions (6,)

    Returns
    -------
    float
        Log-likelihood for this condition
    """
    # Build parameter dict for ODE (matching MCMC_PARAM_NAMES order)
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

    # Compute dependent rate constants from Wegscheider constraints
    formation_energies_array = jnp.array([
        FORMATION_ENERGIES["A"],
        FORMATION_ENERGIES["P"],
        FORMATION_ENERGIES["Q"],
    ])
    k2f, k_pq_r = compute_wegscheider_constraints_jax(params, formation_energies_array, TEMPERATURE)
    params["k2f"] = k2f
    params["k_pq_r"] = k_pq_r

    # Run ODE simulation to steady state
    sol = simulate_ode_to_steady_state_jax(params, initial_array)

    # Check if solver failed
    solver_failed = jnp.any(jnp.isnan(sol.ys))

    # Extract steady-state (SaveAt(t1=True) gives shape (1, 6) or (6,))
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys

    # Extract measurable concentrations only (A, Q, P)
    A, Q, P = y_ss[0], y_ss[1], y_ss[2]
    sim_concs = jnp.array([A, Q, P])

    # Compute net flux (exchange flux at steady state)
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1)
    flux_std = 0.1
    flux_residual = (ref_flux - sim_flux) / flux_std
    ll_flux = -0.5 * flux_residual**2 - jnp.log(flux_std * jnp.sqrt(2 * jnp.pi))

    # Log-likelihood for concentrations (lognormal distribution, std=0.1)
    conc_std = 0.1
    log_ref_concs = jnp.log(jnp.maximum(ref_concs, 1e-10))
    log_sim_concs = jnp.log(jnp.maximum(sim_concs, 1e-10))
    conc_residuals = (log_ref_concs - log_sim_concs) / conc_std
    ll_concs = jnp.sum(
        -0.5 * conc_residuals**2
        - jnp.log(conc_std * jnp.sqrt(2 * jnp.pi))
    )

    ll_condition = ll_flux + ll_concs

    # Return -inf if solver failed
    return jnp.where(solver_failed, -jnp.inf, ll_condition)


def compute_log_posterior_jax(
    log_param_array,
    fixed_params_constant,
    fixed_params_varying,
    initial_array,
    ref_flux_array,
    ref_concs_arrays,
    prior_mean,
    prior_std,
):
    """JAX-compatible log-posterior for multi-condition MCMC sampling.

    Only uses experimentally measurable quantities in likelihood:
    - Concentrations: A, Q, P
    - Flux: net flux through system (v_net)

    Parameters
    ----------
    log_param_array : jnp.array
        Array of log-transformed MCMC parameters (log rate constants)
    fixed_params_constant : jnp.array
        Constant fixed parameters [Q_e, P_e]
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (num_conditions, 2) [[A_e, E_T], ...]
    initial_array : jnp.array
        Initial conditions [A, Q, P, EA, EPQ, EP]
    ref_flux_array : jnp.array
        Reference net flux values (num_conditions,)
    ref_concs_arrays : jnp.array
        Reference concentration values (num_conditions, 3) for [A, Q, P]
    prior_mean : jnp.array
        Prior mean for each parameter (log-scale)
    prior_std : float
        Prior standard deviation (log-scale)

    Returns
    -------
    float
        Log-posterior value
    """
    # Transform from log-space to natural scale
    param_array = jnp.exp(log_param_array)

    # Normal prior on log-parameters (no Jacobian needed since we sample in log-space)
    log_prior = jnp.sum(
        -0.5 * ((log_param_array - prior_mean) / prior_std) ** 2
        - jnp.log(prior_std * jnp.sqrt(2 * jnp.pi))
    )

    # Extract constant parameters (external Q and P concentrations)
    Q_e = fixed_params_constant[0]
    P_e = fixed_params_constant[1]

    # Use vmap to compute likelihood for all conditions in parallel
    condition_lls = jax.vmap(
        lambda A_e, E_T, rf, rc: compute_condition_likelihood(
            A_e, E_T, rf, rc, param_array, Q_e, P_e, initial_array
        )
    )(
        fixed_params_varying[:, 0],  # A_es
        fixed_params_varying[:, 1],  # E_Ts
        ref_flux_array,
        ref_concs_arrays,
    )

    # Sum likelihoods across conditions
    total_log_likelihood = jnp.sum(condition_lls)

    log_posterior = log_prior + total_log_likelihood
    return log_posterior


def run_mcmc(
    ref_params,
    initial,
    conditions=None,
    n_warmup=200,
    n_samples=200,
    seed=42,
):
    """Run MCMC sampling using blackjax NUTS with multiple experimental conditions.

    Parameters
    ----------
    ref_params : dict
        Reference parameter set (used to generate synthetic data)
    initial : dict
        Initial conditions for ODE
    conditions : list of dict, optional
        Experimental conditions with keys 'A_e', 'E_T'. Defaults to EXPERIMENTAL_CONDITIONS.
    n_warmup : int
        Number of warmup iterations
    n_samples : int
        Number of samples to collect
    seed : int
        Random seed

    Returns
    -------
    dict
        MCMC results containing samples and diagnostics
    """
    if conditions is None:
        conditions = EXPERIMENTAL_CONDITIONS

    num_conditions = len(conditions)

    print("=" * 80)
    print(f"Running MCMC with blackjax NUTS ({num_conditions} conditions, no isotope labeling)")
    print("=" * 80)
    print(f"\nExperimental conditions:")
    for i, cond in enumerate(conditions):
        print(f"  Condition {i+1}: E_T={cond['E_T']:.4f}, A_e={cond['A_e']:.4f}")
    sys.stdout.flush()

    # Split fixed parameters into constant and varying
    # Constant: Q_e, P_e (same across all conditions)
    # Varying: A_e, E_T (different per condition)
    fixed_params_constant = jnp.array([
        ref_params["Q_e"],
        ref_params["P_e"],
    ])

    fixed_params_varying = jnp.array([
        [cond["A_e"], cond["E_T"]] for cond in conditions
    ])  # Shape: (num_conditions, 2)

    initial_array = jnp.array([
        initial["A"],
        initial["Q"],
        initial["P"],
        initial["EA"],
        initial["EPQ"],
        initial["EP"],
    ])

    # Generate reference data for all conditions
    print("\nGenerating reference data for all conditions...")
    sys.stdout.flush()
    ref_data_list = generate_reference_data_multi_condition(
        ref_params, initial, conditions
    )

    # Stack reference data into arrays (measurable quantities only)
    # Net flux (scalar per condition)
    ref_flux_array = jnp.array([
        rd["fluxes"]["v_net"]
        for rd in ref_data_list
    ])  # Shape: (num_conditions,)

    # Measurable concentrations: A, Q, P only
    ref_concs_arrays = jnp.array([
        [
            rd["concentrations"]["A"],
            rd["concentrations"]["Q"],
            rd["concentrations"]["P"],
        ]
        for rd in ref_data_list
    ])  # Shape: (num_conditions, 3)

    # Prior: log-normal centered on reference values
    param_array_ref = params_dict_to_array(ref_params)
    prior_mean = jnp.log(param_array_ref)
    prior_std = 0.5  # Allow reasonable variation

    # Log-posterior function (JAX-compatible, takes log-transformed params)
    def logdensity_fn(log_param_array):
        return compute_log_posterior_jax(
            log_param_array,
            fixed_params_constant,
            fixed_params_varying,
            initial_array,
            ref_flux_array,
            ref_concs_arrays,
            prior_mean,
            prior_std,
        )

    # Initialize at reference values (in log-space)
    rng_key = jax.random.PRNGKey(seed)
    initial_position = jnp.log(param_array_ref)

    print(f"Initial position (log-space): {initial_position}")
    print(f"Initial position (natural): {param_array_ref}")
    print(f"Initial log-posterior: {logdensity_fn(initial_position)}")
    sys.stdout.flush()

    # Setup NUTS sampler with window adaptation
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

    # Sampling loop
    print(f"\nCollecting {n_samples} samples...")
    sys.stdout.flush()

    samples = []
    log_probs = []

    for i in range(n_samples):
        rng_key, sample_key = jax.random.split(rng_key)
        state, info = kernel(sample_key, state)
        samples.append(state.position)
        # potential_energy is -log_density
        log_probs.append(-state.potential_energy)

        if (i + 1) % 100 == 0:
            print(f"  Collected {i + 1}/{n_samples} samples")
            sys.stdout.flush()

    log_samples = jnp.stack(samples)
    log_probs = jnp.array(log_probs)

    # Transform samples back to natural scale
    samples = jnp.exp(log_samples)

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("MCMC Results Summary")
    print("=" * 80)

    results = {
        "samples": samples,  # Natural scale
        "log_samples": log_samples,  # Log scale
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


def print_results(concs, net_rates, fluxes):
    """Print simulation results."""
    print("\n" + "=" * 80)
    print("Steady-State Concentrations")
    print("=" * 80)
    for met_id, conc in concs.items():
        print(f"  {met_id}: {conc:.6f}")

    enzyme_total = concs["E"] + concs["EA"] + concs["EPQ"] + concs["EP"]
    print(f"\n  Enzyme conservation check: E + EA + EPQ + EP = {enzyme_total:.6f}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Steady-State Net Fluxes")
    print("=" * 80)
    for rate_id, rate_val in net_rates.items():
        print(f"  {rate_id}: {rate_val:.6f}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Forward/Backward Fluxes")
    print("=" * 80)
    for flux_id, flux_val in fluxes.items():
        print(f"  {flux_id}: {flux_val:.6f}")
    sys.stdout.flush()


def main():
    print("=" * 80)
    print("Enzyme Kinetics ODE Simulation (No Isotope Labeling)")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()
    params = DEFAULT_PARAMS.copy()
    initial = DEFAULT_INITIAL.copy()

    # Step 1: Apply thermodynamic constraints
    print("\n" + "=" * 80)
    print("Step 1: Applying Wegscheider Thermodynamic Constraints")
    print("=" * 80)
    print(f"  Formation energies: {FORMATION_ENERGIES}")
    print(f"  Temperature: {TEMPERATURE} K")

    # Compute Gibbs energies for display
    RT = R_GAS * TEMPERATURE
    delta_G_enzyme = FORMATION_ENERGIES["P"] + FORMATION_ENERGIES["Q"] - FORMATION_ENERGIES["A"]
    delta_G_pq = FORMATION_ENERGIES["Q"] - FORMATION_ENERGIES["P"]
    K_enzyme = np.exp(-delta_G_enzyme / RT)
    K_pq = np.exp(-delta_G_pq / RT)

    print(f"  ΔG°(A → P + Q) = {delta_G_enzyme:.2f} kJ/mol  →  K_eq = {K_enzyme:.4f}")
    print(f"  ΔG°(P → Q) = {delta_G_pq:.2f} kJ/mol  →  K_pq = {K_pq:.4f}")

    # Apply constraints to compute dependent rate constants
    params = compute_wegscheider_constraints(params)
    print(f"\n  Computed dependent rate constants:")
    print(f"    k2f = {params['k2f']:.6f}")
    print(f"    k_pq_r = {params['k_pq_r']:.6f}")
    sys.stdout.flush()

    # Step 2: Simulate ODE to steady state
    print("\n" + "=" * 80)
    print("Step 2: Simulating ODE System to Steady State")
    print("=" * 80)
    print(f"  Parameters: {params}")
    print(f"  Initial conditions: {initial}")
    sys.stdout.flush()

    sol = simulate_ode_to_steady_state(params, initial)

    # Check if steady state was reached (event_occurred) or integration was successful
    if sol.result not in (diffrax.RESULTS.successful, diffrax.RESULTS.discrete_terminating_event_occurred):
        print(f"  ERROR: ODE integration failed: {sol.result}")
        sys.exit(1)

    print(f"  ODE integration completed (result: {sol.result})")
    sys.stdout.flush()

    # Step 3: Extract steady state
    print("\n" + "=" * 80)
    print("Step 3: Extracting Steady-State Values")
    print("=" * 80)
    sys.stdout.flush()

    concs, net_rates = get_steady_state(sol, params)
    fluxes = compute_forward_backward_fluxes(concs, params)

    # Print results
    print_results(concs, net_rates, fluxes)

    # Step 4: Run MCMC parameter estimation
    print("\n" + "=" * 80)
    print("Step 4: Running MCMC Parameter Estimation")
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

"""Shared constants and functions for ordered-mechanism enzyme simulations.

This module contains the common code used by enzyme_simulation.py,
enzyme_simulation_instationary.py, and enzyme_simulation_no_labeling.py.
It includes thermodynamic constraints, ODE definitions, and parameter utilities
for the ordered uni-bi enzyme mechanism (A → P + Q).
"""

import numpy as np
import jax.numpy as jnp
import diffrax
from diffrax import SteadyStateEvent

try:
    from scripts.isotopomer_simulation import enzyme_ode
except ModuleNotFoundError:
    from isotopomer_simulation import enzyme_ode


# ============================================================================
# Thermodynamic constants
# ============================================================================

R_GAS = 8.314e-3  # Gas constant in kJ/(mol·K)
TEMPERATURE = 298.15  # Temperature in K

FORMATION_ENERGIES = {
    "A": 0.0,    # Reference state
    "P": 0.5,    # kJ/mol
    "Q": -2.5,   # kJ/mol
}


# ============================================================================
# Default parameters and initial conditions
# ============================================================================

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

DEFAULT_INITIAL = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
}

# Parameter names that will be sampled in MCMC
MCMC_PARAM_NAMES = [
    "k2r", "k3f", "k3r", "k4f", "k4r", "k5f", "k5r",
    "k_A_ex", "k_Q_ex", "k_P_ex", "k_pq_f"
]


# ============================================================================
# Experimental conditions (shared random seed for reproducibility)
# ============================================================================

np.random.seed(12345)
_n_conditions = 10
EXPERIMENTAL_CONDITIONS = [
    {
        "E_T": float(E_T),
        "A_e": float(A_e),
        "Q_e": float(Q_e),
        "P_e": float(P_e),
    }
    for E_T, A_e, Q_e, P_e in zip(
        np.exp(np.random.normal(0, 0.8, _n_conditions)),      # E_T: wider range ~[0.2, 5]
        np.exp(np.random.normal(1.5, 0.8, _n_conditions)),    # A_e: wider range ~[1, 20]
        np.exp(np.random.normal(0, 0.8, _n_conditions)),      # Q_e: wider range ~[0.2, 5]
        np.exp(np.random.normal(0, 0.8, _n_conditions)),      # P_e: wider range ~[0.2, 5]
    )
]
NUM_CONDITIONS = len(EXPERIMENTAL_CONDITIONS)


# ============================================================================
# Wegscheider thermodynamic constraints
# ============================================================================

def compute_wegscheider_constraints(params, formation_energies=None, temperature=None):
    """Compute dependent rate constants from Wegscheider conditions.

    Enforces thermodynamic consistency by computing k2f and k_pq_r from the
    Haldane relation and equilibrium constant.
    """
    if formation_energies is None:
        formation_energies = FORMATION_ENERGIES
    if temperature is None:
        temperature = TEMPERATURE

    RT = R_GAS * temperature

    delta_G_enzyme = (
        formation_energies["P"] + formation_energies["Q"] - formation_energies["A"]
    )
    K_enzyme = np.exp(-delta_G_enzyme / RT)

    delta_G_pq = formation_energies["Q"] - formation_energies["P"]
    K_pq = np.exp(-delta_G_pq / RT)

    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

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


# ============================================================================
# Enzyme ODE utilities
# ============================================================================

def compute_E(EA, EPQ, EP, E_T):
    """Compute free enzyme from conservation constraint."""
    return E_T - EA - EPQ - EP


def compute_rates(y, params):
    """Compute all reaction rates from state and parameters."""
    A, Q, P, EA, EPQ, EP = y
    E = compute_E(EA, EPQ, EP, params["E_T"])

    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)
    v_Q_ex = params["k_Q_ex"] * (params["Q_e"] - Q)
    v_P_ex = params["k_P_ex"] * (params["P_e"] - P)

    v2 = A * E * params["k2f"] - EA * params["k2r"]
    v3 = EA * params["k3f"] - EPQ * params["k3r"]
    v4 = EPQ * params["k4f"] - Q * EP * params["k4r"]
    v5 = EP * params["k5f"] - E * P * params["k5r"]

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


# ============================================================================
# Simulation utilities
# ============================================================================

def simulate_enzyme_to_steady_state_jax(params, initial_array):
    """JAX-compatible enzyme ODE simulation to steady state.

    Uses enzyme_ode (total concentrations without isotopomers) to find
    metabolic steady state.
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


def simulate_ode_to_steady_state(params, initial):
    """Simulate total concentration ODE to steady state (applies Wegscheider constraints)."""
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


# ============================================================================
# Post-processing utilities
# ============================================================================

def get_steady_state(sol, params):
    """Extract steady-state concentrations and fluxes."""
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
        "v_A_ex": params["k_A_ex"] * (params["A_e"] - A),
        "v_Q_ex": params["k_Q_ex"] * (params["Q_e"] - Q),
        "v_P_ex": params["k_P_ex"] * (params["P_e"] - P),
        "v2_f": A * E * params["k2f"],
        "v2_b": EA * params["k2r"],
        "v3_f": EA * params["k3f"],
        "v3_b": EPQ * params["k3r"],
        "v4_f": EPQ * params["k4f"],
        "v4_b": Q * EP * params["k4r"],
        "v5_f": EP * params["k5f"],
        "v5_b": E * P * params["k5r"],
        "vpq_f": P * params["k_pq_f"],
        "vpq_b": Q * params["k_pq_r"],
    }

    return fluxes


# ============================================================================
# Parameter conversion utilities
# ============================================================================

def params_array_to_dict(param_array, fixed_params):
    """Convert parameter array to full parameter dictionary."""
    params = fixed_params.copy()
    for i, name in enumerate(MCMC_PARAM_NAMES):
        params[name] = param_array[i]
    return params


def params_dict_to_array(params):
    """Extract MCMC parameters from dictionary to array."""
    return jnp.array([params[name] for name in MCMC_PARAM_NAMES])

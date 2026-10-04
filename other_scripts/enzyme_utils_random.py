"""Shared constants and functions for random-mechanism enzyme simulations.

This module contains the enzyme ODE for the random uni-bi mechanism
(A -> P + Q with two product release pathways) so that MCMC scripts
can use it without duplication.
"""

import numpy as np
import jax.numpy as jnp


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
# Default parameters for the random mechanism
# ============================================================================

DEFAULT_PARAMS_RANDOM = {
    "A_e": 5.0,
    "Q_e": 1.0,
    "P_e": 1.0,
    "k_A_ex": 0.1,
    "k_Q_ex": 0.1,
    "k_P_ex": 0.1,
    "k2f": 10.0,
    "k2r": 250.0,
    "k3f": 125.0,
    "k3r": 12.5,
    "k4af": 100.0,
    "k4ar": 10.0,
    "k5af": 75.0,
    "k5ar": 7.5,
    "k4bf": 100.0,
    "k4br": 10.0,
    "k5bf": 75.0,
    "k5br": 7.5,
    "k_pq_f": 50.0,
    "k_pq_r": 15.0,
    "E_T": 1.0,
}

DEFAULT_INITIAL_TOTAL_RANDOM = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
    "EQ": 0.0,
}


# ============================================================================
# Wegscheider constraints for random mechanism
# ============================================================================

def compute_wegscheider_constraints_random(params, formation_energies=None, temperature=None):
    """Compute dependent rate constants from Wegscheider conditions for random mechanism.

    Three constraints:
    1. Overall enzyme equilibrium: k2f from K_enzyme
    2. Loop closure: k5br from K4a * K5a = K4b * K5b
    3. P <-> Q equilibrium: k_pq_r from K_pq
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

    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4ar"] * params["k5ar"]
    ) / (params["k3f"] * params["k4af"] * params["k5af"])

    k5br = (
        params["k4bf"] * params["k5bf"] * params["k4ar"] * params["k5ar"]
    ) / (params["k4af"] * params["k5af"] * params["k4br"])

    delta_G_pq = formation_energies["Q"] - formation_energies["P"]
    K_pq = np.exp(-delta_G_pq / RT)
    k_pq_r = params["k_pq_f"] / K_pq

    updated_params = params.copy()
    updated_params["k2f"] = k2f
    updated_params["k5br"] = k5br
    updated_params["k_pq_r"] = k_pq_r

    return updated_params


# ============================================================================
# Enzyme ODE for random mechanism (total concentrations, no isotopomers)
# ============================================================================

def enzyme_ode_random(t, y, params):
    """ODE system for total concentrations (no isotope labeling) - random mechanism.

    State variables: [A, Q, P, EA, EPQ, EP, EQ]

    Enzyme conservation: E = E_T - EA - EPQ - EP - EQ
    """
    A, Q, P, EA, EPQ, EP, EQ = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

    E = params["E_T"] - EA - EPQ - EP - EQ

    A_e = params["A_e"]
    Q_e = params["Q_e"]
    P_e = params["P_e"]
    k_A_ex = params["k_A_ex"]
    k_Q_ex = params["k_Q_ex"]
    k_P_ex = params["k_P_ex"]
    k2f, k2r = params["k2f"], params["k2r"]
    k3f, k3r = params["k3f"], params["k3r"]
    k4af, k4ar = params["k4af"], params["k4ar"]
    k5af, k5ar = params["k5af"], params["k5ar"]
    k4bf, k4br = params["k4bf"], params["k4br"]
    k5bf, k5br = params["k5bf"], params["k5br"]
    k_pq_f, k_pq_r = params["k_pq_f"], params["k_pq_r"]

    v_A_ex = k_A_ex * (A_e - A)
    v_Q_ex = k_Q_ex * (Q_e - Q)
    v_P_ex = k_P_ex * (P_e - P)

    v2 = A * E * k2f - EA * k2r
    v3 = EA * k3f - EPQ * k3r
    v4a = EPQ * k4af - Q * EP * k4ar
    v5a = EP * k5af - E * P * k5ar
    v4b = EPQ * k4bf - P * EQ * k4br
    v5b = EQ * k5bf - E * Q * k5br

    v_pq = P * k_pq_f - Q * k_pq_r

    dA_dt = v_A_ex - v2
    dQ_dt = v4a + v5b + v_Q_ex + v_pq
    dP_dt = v4b + v5a + v_P_ex - v_pq
    dEA_dt = v2 - v3
    dEPQ_dt = v3 - v4a - v4b
    dEP_dt = v4a - v5a
    dEQ_dt = v4b - v5b

    return jnp.array([dA_dt, dQ_dt, dP_dt, dEA_dt, dEPQ_dt, dEP_dt, dEQ_dt])

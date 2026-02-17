#!/usr/bin/env python3
"""Enzyme kinetics ODE simulation with INSTATIONARY EMU labeling analysis.

Random Uni-Bi Enzyme Mechanism Implementation

This script simulates a random uni-bi enzyme mechanism where either product
(P or Q) can release first from the EPQ complex.

Ordered (original): E + A -> EA -> EPQ -> EP + Q -> E + P (Q must release first)

Random (new):
                    +-> EP + Q -> E + P + Q  (path 1: Q released first, v4a/v5a)
E + A <-> EA <-> EPQ -|
                    +-> EQ + P -> E + Q + P  (path 2: P released first, v4b/v5b)

Wegscheider Constraints:
1. Overall enzyme equilibrium: k2f computed from K_enzyme and other rate constants
2. Loop closure: k5br computed to satisfy K4a * K5a = K4b * K5b
3. P <-> Q equilibrium: k_pq_r computed from K_pq

MCMC Parameters:
- Sampled (14): k2r, k3f, k3r, k4af, k4ar, k5af, k5ar, k4bf, k4br, k5bf,
                k_A_ex, k_Q_ex, k_P_ex, k_pq_f
- Computed (3): k2f, k5br, k_pq_r
- Fixed per condition: A_e, E_T

Labeling:
  - A_ext: first atom labeled (pattern "10")
  - Q_ext_in, P_ext_in: 100% unlabeled
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
from jax_tqdm import scan_tqdm

try:
    from scripts.enzyme_utils_random import enzyme_ode_random
    from scripts.emu_jax import (
        build_emu_network,
        simulate_inst_mdvs_pure_jax,
        precompute_matrix_structure,
    )
    # Import ordered model functions for generating reference data
    from scripts.mcmc.enzyme_simulation_instationary import (
        generate_reference_data_multi_condition as generate_reference_data_ordered,
        DEFAULT_PARAMS as ORDERED_DEFAULT_PARAMS,
        DEFAULT_INITIAL as ORDERED_DEFAULT_INITIAL,
        compute_wegscheider_constraints as compute_wegscheider_ordered,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from enzyme_utils_random import enzyme_ode_random
    from emu_jax import (
        build_emu_network,
        simulate_inst_mdvs_pure_jax,
        precompute_matrix_structure,
    )
    # Import ordered model functions for generating reference data
    from enzyme_simulation_instationary import (
        generate_reference_data_multi_condition as generate_reference_data_ordered,
        DEFAULT_PARAMS as ORDERED_DEFAULT_PARAMS,
        DEFAULT_INITIAL as ORDERED_DEFAULT_INITIAL,
        compute_wegscheider_constraints as compute_wegscheider_ordered,
    )


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REACTIONS_FILE = PROJECT_ROOT / "models" / "simple_network_random" / "reactions.tsv"
RESULTS_DIR = PROJECT_ROOT / "results" / "enzyme_simulation_random_instationary"

# Global cache for EMU network (loaded once, reused)
_PURE_JAX_EMU_NETWORK_RANDOM = None


# Observation time points for instationary labeling (5 log-spaced from 10^-1 to 10^2)
# MUST match enzyme_simulation_instationary.py for consistent reference data
OBSERVATION_TIMES = jnp.logspace(-1, 2, 5)
N_TIMEPOINTS = len(OBSERVATION_TIMES)

# Number of MID values per time point (A: 3, Q: 2, P: 2 = 7 total)
N_MIDS = 7


# Default parameters for the random mechanism
# Initialized to match ordered mechanism: path 1 (Q first) = ordered, path 2 = same as path 1
DEFAULT_PARAMS = {
    # External pool concentrations (fixed boundary conditions)
    "A_e": 5.0,          # external A concentration
    "Q_e": 1.0,          # external Q concentration
    "P_e": 1.0,          # external P concentration
    # Exchange rate constants
    "k_A_ex": 0.1,       # A exchange rate constant
    "k_Q_ex": 0.1,       # Q exchange rate constant
    "k_P_ex": 0.1,       # P exchange rate constant
    # Enzyme kinetic rate constants
    "k2f": 10.0,         # A + E -> EA forward (computed from Wegscheider)
    "k2r": 250.0,        # EA -> A + E reverse
    "k3f": 125.0,        # EA -> EPQ forward
    "k3r": 12.5,         # EPQ -> EA reverse
    # Path 1: Q releases first (EPQ -> EP + Q -> E + P + Q) - matches ordered mechanism
    "k4af": 200.0,       # EPQ -> EP + Q forward (same as ordered k4f)
    "k4ar": 20.0,        # EP + Q -> EPQ reverse (same as ordered k4r)
    "k5af": 150.0,       # EP -> E + P forward (same as ordered k5f)
    "k5ar": 15.0,        # E + P -> EP reverse (same as ordered k5r)
    # Path 2: P releases first (EPQ -> EQ + P -> E + Q + P) - identical to path 1
    "k4bf": 200.0,       # EPQ -> EQ + P forward (same as k4af)
    "k4br": 20.0,        # EQ + P -> EPQ reverse (same as k4ar)
    "k5bf": 150.0,       # EQ -> E + Q forward (same as k5af)
    "k5br": 15.0,        # E + Q -> EQ reverse (computed from Wegscheider loop)
    # P <-> Q interconversion
    "k_pq_f": 50.0,      # P -> Q forward
    "k_pq_r": 15.0,      # Q -> P reverse (computed from Wegscheider)
    "E_T": 1.0,          # total enzyme concentration
}

# Initial conditions (state order: A, Q, P, EA, EPQ, EP, EQ)
DEFAULT_INITIAL = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
    "EQ": 0.0,
}

# Experimental conditions for multi-condition fitting
# Fixed seed ensures identical conditions across all scripts
# 30 conditions with varying A_e, E_T, Q_e, P_e
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

# Thermodynamic constants
R_GAS = 8.314e-3  # Gas constant in kJ/(mol*K)
TEMPERATURE = 298.15  # Temperature in K

# Formation energies (chemical potentials) in kJ/mol
FORMATION_ENERGIES = {
    "A": 0.0,    # Reference state
    "P": 0.5,    # kJ/mol
    "Q": -2.5,   # kJ/mol
}


# =============================================================================
# Wegscheider Constraints for Random Mechanism
# =============================================================================


def compute_wegscheider_constraints_random(params, formation_energies=None, temperature=None):
    """Compute dependent rate constants from Wegscheider conditions for random mechanism.

    Three constraints:
    1. Overall enzyme equilibrium: k2f from K_enzyme
    2. Loop closure: k5br from K4a * K5a = K4b * K5b
    3. P <-> Q equilibrium: k_pq_r from K_pq

    Parameters
    ----------
    params : dict
        Parameter dictionary with sampled rate constants
    formation_energies : dict, optional
        Chemical potentials for A, P, Q
    temperature : float, optional
        Temperature in K

    Returns
    -------
    dict
        Updated parameter dictionary with computed k2f, k5br, k_pq_r
    """
    if formation_energies is None:
        formation_energies = FORMATION_ENERGIES
    if temperature is None:
        temperature = TEMPERATURE

    RT = R_GAS * temperature

    # Constraint 1: Overall enzyme equilibrium A -> P + Q
    delta_G_enzyme = (
        formation_energies["P"] + formation_energies["Q"] - formation_energies["A"]
    )
    K_enzyme = np.exp(-delta_G_enzyme / RT)

    # Using path 1 (v4a/v5a) for the Haldane relation:
    # K_enzyme = (k2f * k3f * k4af * k5af) / (k2r * k3r * k4ar * k5ar)
    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4ar"] * params["k5ar"]
    ) / (params["k3f"] * params["k4af"] * params["k5af"])

    # Constraint 2: Loop closure
    # Both pathways must yield the same overall equilibrium:
    # K4a * K5a = K4b * K5b
    # where K4a = k4af/k4ar, K5a = k5af/k5ar, K4b = k4bf/k4br, K5b = k5bf/k5br
    #
    # k5br = (k4bf * k5bf * k4ar * k5ar) / (k4af * k5af * k4br)
    k5br = (
        params["k4bf"] * params["k5bf"] * params["k4ar"] * params["k5ar"]
    ) / (params["k4af"] * params["k5af"] * params["k4br"])

    # Constraint 3: P <-> Q equilibrium
    delta_G_pq = formation_energies["Q"] - formation_energies["P"]
    K_pq = np.exp(-delta_G_pq / RT)
    k_pq_r = params["k_pq_f"] / K_pq

    updated_params = params.copy()
    updated_params["k2f"] = k2f
    updated_params["k5br"] = k5br
    updated_params["k_pq_r"] = k_pq_r

    return updated_params


def compute_wegscheider_constraints_random_jax(params, formation_energies_array, temperature):
    """JAX-compatible version of Wegscheider constraint computation for random mechanism.

    Parameters
    ----------
    params : dict
        Parameter dictionary with sampled rate constants
    formation_energies_array : jnp.array
        [mu_A, mu_P, mu_Q] formation energies
    temperature : float
        Temperature in K

    Returns
    -------
    tuple
        (k2f, k5br, k_pq_r) computed rate constants
    """
    RT = R_GAS * temperature
    mu_A, mu_P, mu_Q = formation_energies_array[0], formation_energies_array[1], formation_energies_array[2]

    # Constraint 1: Overall enzyme equilibrium
    delta_G_enzyme = mu_P + mu_Q - mu_A
    K_enzyme = jnp.exp(-delta_G_enzyme / RT)

    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4ar"] * params["k5ar"]
    ) / (params["k3f"] * params["k4af"] * params["k5af"])

    # Constraint 2: Loop closure
    k5br = (
        params["k4bf"] * params["k5bf"] * params["k4ar"] * params["k5ar"]
    ) / (params["k4af"] * params["k5af"] * params["k4br"])

    # Constraint 3: P <-> Q equilibrium
    delta_G_pq = mu_Q - mu_P
    K_pq = jnp.exp(-delta_G_pq / RT)
    k_pq_r = params["k_pq_f"] / K_pq

    return k2f, k5br, k_pq_r


def verify_wegscheider_loop_closure(params):
    """Verify that the loop closure constraint is satisfied.

    K4a * K5a should equal K4b * K5b

    Parameters
    ----------
    params : dict
        Parameter dictionary with all rate constants

    Returns
    -------
    tuple
        (K4a * K5a, K4b * K5b, relative_error)
    """
    K4a = params["k4af"] / params["k4ar"]
    K5a = params["k5af"] / params["k5ar"]
    K4b = params["k4bf"] / params["k4br"]
    K5b = params["k5bf"] / params["k5br"]

    path1_product = K4a * K5a
    path2_product = K4b * K5b

    rel_error = abs(path1_product - path2_product) / max(path1_product, path2_product)

    return path1_product, path2_product, rel_error


# =============================================================================
# ODE Simulation Functions
# =============================================================================


def compute_E_random(EA, EPQ, EP, EQ, E_T):
    """Compute free enzyme from conservation constraint for random mechanism."""
    return E_T - EA - EPQ - EP - EQ


def simulate_enzyme_to_steady_state_random_jax(params, initial_array):
    """JAX-compatible enzyme ODE simulation to steady state for random mechanism.

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    initial_array : jnp.array
        Initial concentrations [A, Q, P, EA, EPQ, EP, EQ]

    Returns
    -------
    diffrax.Solution
        ODE solution at steady state
    """
    term = diffrax.ODETerm(enzyme_ode_random)
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


# =============================================================================
# Pure JAX EMU Simulation (Fast, JAX-Traceable)
# =============================================================================


def get_pure_jax_emu_network_random():
    """Get or build the pure JAX EMU network for the random mechanism.

    Returns
    -------
    tuple
        - network: EMU network structure dict
        - precomputed: Precomputed matrix structure arrays for JAX tracing
    """
    global _PURE_JAX_EMU_NETWORK_RANDOM

    if _PURE_JAX_EMU_NETWORK_RANDOM is None:
        network = build_emu_network(
            reactions_file=REACTIONS_FILE,
            target_emus={"A": ["12"], "EA": ["12"], "EPQ": ["12"],
                         "EP": ["1"], "EQ": ["1"], "Q": ["1"], "P": ["1"]},
            labeling_substrate="A_ext",
            labeling_pattern=["10"],
            labeling_percentage=[1.0],
            labeling_purity=[1.0],
        )
        precomputed = precompute_matrix_structure(network)
        _PURE_JAX_EMU_NETWORK_RANDOM = (network, precomputed)

    return _PURE_JAX_EMU_NETWORK_RANDOM


def compute_fluxes_array_pure_jax_random(concs, params, network):
    """Compute flux array for pure JAX EMU simulation (random mechanism).

    Parameters
    ----------
    concs : dict
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, EQ, E
    params : dict
        Rate constants
    network : dict
        EMU network structure

    Returns
    -------
    jnp.ndarray
        Flux array in order matching network["flux_ids_all"]
    """
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    EQ = concs["EQ"]
    E = concs["E"]

    # Compute all fluxes (matching reactions.tsv order for random mechanism)
    fluxes_dict = {
        "v1": params["k_A_ex"] * params["A_e"],  # A_ext -> A (input)
        "v1_out": params["k_A_ex"] * A,  # A -> A_sink (output)
        "v2_f": A * E * params["k2f"],
        "v2_b": EA * params["k2r"],
        "v3_f": EA * params["k3f"],
        "v3_b": EPQ * params["k3r"],
        # Path 1: Q releases first
        "v4a_f": EPQ * params["k4af"],
        "v4a_b": Q * EP * params["k4ar"],
        # Path 2: P releases first
        "v4b_f": EPQ * params["k4bf"],
        "v4b_b": P * EQ * params["k4br"],
        # Product release
        "v5a_f": EP * params["k5af"],
        "v5a_b": E * P * params["k5ar"],
        "v5b_f": EQ * params["k5bf"],
        "v5b_b": E * Q * params["k5br"],
        # Exchange fluxes
        "vq_out": params["k_Q_ex"] * Q,  # Q -> Q_ext (output)
        "vq_in": params["k_Q_ex"] * params["Q_e"],  # Q_ext_in -> Q (input)
        "vp_out": params["k_P_ex"] * P,  # P -> P_ext (output)
        "vp_in": params["k_P_ex"] * params["P_e"],  # P_ext_in -> P (input)
        # P <-> Q interconversion
        "vpq_f": P * params["k_pq_f"],
        "vpq_b": Q * params["k_pq_r"],
    }

    # Build array in network order
    return jnp.array([fluxes_dict[fid] for fid in network["flux_ids_all"]])


def compute_concs_array_pure_jax_random(concs, network):
    """Compute concentration array for pure JAX EMU simulation.

    Parameters
    ----------
    concs : dict
        Concentration values
    network : dict
        EMU network structure

    Returns
    -------
    jnp.ndarray
        Concentration array in order matching network["metab_ids_all"]
    """
    return jnp.array([concs.get(mid, 1.0) for mid in network["metab_ids_all"]])


def extract_measurable_mids_from_emu_random(emu_mdvs):
    """Extract measurable MIDs (A, Q, P) from EMU results.

    Parameters
    ----------
    emu_mdvs : dict
        {emu_id: mdv_array} from simulate_inst_mdvs_pure_jax

    Returns
    -------
    jnp.ndarray
        MID array (7,): [A_M0, A_M1, A_M2, Q_M0, Q_M1, P_M0, P_M1]
    """
    # A_12 gives M+0, M+1, M+2 (size 3)
    A_mid = emu_mdvs["A_12"]

    # Q_1 gives M+0, M+1 (size 2)
    Q_mid = emu_mdvs["Q_1"]

    # P_1 gives M+0, M+1 (size 2)
    P_mid = emu_mdvs["P_1"]

    return jnp.concatenate([A_mid, Q_mid, P_mid])


def simulate_emu_timecourse_random(params, concs_dict, observation_times, network, precomputed):
    """Simulate EMU dynamics and return MIDs at specified times (random mechanism).

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    concs_dict : dict
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, EQ, E
    observation_times : jnp.array
        Time points at which to observe MIDs
    network : dict
        EMU network structure
    precomputed : dict
        Precomputed matrix structure arrays for JAX tracing

    Returns
    -------
    dict
        {time: mids_array} where mids_array is (7,) for measurable metabolites
    """
    # Include t=0 in timepoints for simulation
    all_timepoints = jnp.concatenate([jnp.array([0.0]), observation_times])

    # Compute fluxes and concentrations arrays
    fluxes_array = compute_fluxes_array_pure_jax_random(concs_dict, params, network)
    concs_array = compute_concs_array_pure_jax_random(concs_dict, network)

    # Run pure JAX EMU simulation for each timepoint
    # Note: simulate_inst_mdvs_pure_jax returns MIDs at the final timepoint
    timecourse_mids = {}
    for t in observation_times:
        t_float = float(t)
        timepoints_to_t = jnp.array([0.0, t_float])
        emu_mdvs_at_t = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, network, precomputed
        )
        timecourse_mids[t_float] = extract_measurable_mids_from_emu_random(emu_mdvs_at_t)

    return timecourse_mids


# =============================================================================
# Isotopomer Simulation using ODE (kept for reference data generation)
# =============================================================================



# =============================================================================
# Reference Data Generation
# =============================================================================


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def generate_reference_data_single_condition_random(params, initial, observation_times):
    """Generate reference observations including time-course MIDs for random mechanism.

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
    params = compute_wegscheider_constraints_random(params)

    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"], initial["EQ"]
    ])

    # Step 1: Run enzyme_ode_random to get metabolic steady state
    enzyme_sol = simulate_enzyme_to_steady_state_random_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Extract concentrations from enzyme_ode steady state
    A, Q, P = float(enzyme_y_ss[0]), float(enzyme_y_ss[1]), float(enzyme_y_ss[2])
    EA, EPQ, EP, EQ = float(enzyme_y_ss[3]), float(enzyme_y_ss[4]), float(enzyme_y_ss[5]), float(enzyme_y_ss[6])
    E = params["E_T"] - EA - EPQ - EP - EQ

    concs = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "EQ": EQ, "E": E}

    # Exchange fluxes with external pools
    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)
    v_Q_ex = params["k_Q_ex"] * (params["Q_e"] - Q)
    v_P_ex = params["k_P_ex"] * (params["P_e"] - P)

    net_rates = {
        "v_A_ex": v_A_ex,
        "v2": A * E * params["k2f"] - EA * params["k2r"],
        "v3": EA * params["k3f"] - EPQ * params["k3r"],
        "v4a": EPQ * params["k4af"] - Q * EP * params["k4ar"],
        "v5a": EP * params["k5af"] - E * P * params["k5ar"],
        "v4b": EPQ * params["k4bf"] - P * EQ * params["k4br"],
        "v5b": EQ * params["k5bf"] - E * Q * params["k5br"],
        "v_Q_ex": v_Q_ex,
        "v_P_ex": v_P_ex,
        "v_pq": P * params["k_pq_f"] - Q * params["k_pq_r"],
        "v_net": v_A_ex,
    }

    # Step 2: Run pure JAX EMU simulation for time-course MIDs
    # (Using EMU ensures consistency with likelihood computation)
    network, precomputed = get_pure_jax_emu_network_random()
    timecourse_mids = simulate_emu_timecourse_random(
        params, concs, observation_times, network, precomputed
    )

    # Convert to numpy arrays for storage
    timecourse_mids = {t: np.array(mids) for t, mids in timecourse_mids.items()}

    return {
        "fluxes": net_rates,
        "concentrations": concs,
        "timecourse_mids": timecourse_mids,
        "enzyme_steady_state": np.array(enzyme_y_ss),
    }


# =============================================================================
# MCMC Setup
# =============================================================================


# Parameter names that will be sampled (14 parameters)
MCMC_PARAM_NAMES_RANDOM = [
    "k2r", "k3f", "k3r",           # binding/isomerization
    "k4af", "k4ar", "k5af", "k5ar",  # path 1: Q first
    "k4bf", "k4br", "k5bf",          # path 2: P first
    "k_A_ex", "k_Q_ex", "k_P_ex", "k_pq_f"  # exchange
]


def params_dict_to_array_random(params):
    """Extract MCMC parameters from dictionary to array."""
    return jnp.array([params[name] for name in MCMC_PARAM_NAMES_RANDOM])


def compute_condition_likelihood_random(
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
    t_final,
):
    """Compute log-likelihood for a single condition with timecourse MIDs.

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
        Rate constants (14,)
    Q_e : float
        External Q concentration (fixed)
    P_e : float
        External P concentration (fixed)
    initial_array : jnp.array
        Initial conditions for enzyme_ode (7,): [A, Q, P, EA, EPQ, EP, EQ]
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
        "k4af": param_array[3],
        "k4ar": param_array[4],
        "k5af": param_array[5],
        "k5ar": param_array[6],
        "k4bf": param_array[7],
        "k4br": param_array[8],
        "k5bf": param_array[9],
        "k_A_ex": param_array[10],
        "k_Q_ex": param_array[11],
        "k_P_ex": param_array[12],
        "k_pq_f": param_array[13],
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
    k2f, k5br, k_pq_r = compute_wegscheider_constraints_random_jax(
        params, formation_energies_array, TEMPERATURE
    )
    params["k2f"] = k2f
    params["k5br"] = k5br
    params["k_pq_r"] = k_pq_r

    # Step 1: Run enzyme_ode_random to get metabolic steady state
    enzyme_sol = simulate_enzyme_to_steady_state_random_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Check if enzyme solver failed
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (
        enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred
    )

    # Extract concentrations from enzyme_ode steady state
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    EQ = enzyme_y_ss[6]
    E = params["E_T"] - EA - EPQ - EP - EQ

    sim_concs = jnp.array([A, Q, P])

    # Build concentration dict for EMU simulation
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "EQ": EQ, "E": E}

    # Net flux from enzyme_ode steady state
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1% of |ref_flux|)
    flux_std = 0.01 * jnp.abs(ref_flux)
    flux_residual = (ref_flux - sim_flux) / flux_std
    ll_flux = -0.5 * flux_residual**2 - jnp.log(flux_std * jnp.sqrt(2 * jnp.pi))

    # Log-likelihood for concentrations (lognormal)
    conc_std = 0.01
    log_ref_concs = jnp.log(jnp.maximum(ref_concs, 1e-10))
    log_sim_concs = jnp.log(jnp.maximum(sim_concs, 1e-10))
    conc_residuals = (log_ref_concs - log_sim_concs) / conc_std
    ll_concs = jnp.sum(
        -0.5 * conc_residuals**2 - jnp.log(conc_std * jnp.sqrt(2 * jnp.pi))
    )

    # Step 2: Run pure JAX EMU simulation for time-course MIDs
    # Get the pre-loaded EMU network and precomputed structure
    network, precomputed = get_pure_jax_emu_network_random()

    # Compute fluxes and concentrations arrays for EMU simulation
    fluxes_array = compute_fluxes_array_pure_jax_random(concs_dict, params, network)
    concs_array = compute_concs_array_pure_jax_random(concs_dict, network)

    # Log-likelihood for time-course MIDs (lognormal)
    mid_std = 0.01

    def compute_timepoint_ll(t_idx, ll_acc):
        # Get observation time for this index
        t = observation_times[t_idx]

        # Run EMU simulation to this timepoint
        timepoints_to_t = jnp.array([0.0, t])
        emu_mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, network, precomputed
        )

        # Extract measurable MIDs
        sim_mids_t = extract_measurable_mids_from_emu_random(emu_mdvs)
        ref_mids_t = ref_mids_timecourse[t_idx]

        log_ref = jnp.log(jnp.maximum(ref_mids_t, 1e-10))
        log_sim = jnp.log(jnp.maximum(sim_mids_t, 1e-10))
        residuals = (log_ref - log_sim) / mid_std
        ll_t = jnp.sum(
            -0.5 * residuals**2 - jnp.log(mid_std * jnp.sqrt(2 * jnp.pi))
        )
        return ll_acc + ll_t

    ll_mids = jax.lax.fori_loop(0, len(observation_times), compute_timepoint_ll, 0.0)

    ll_condition = ll_flux + ll_concs + ll_mids

    # Return -inf if enzyme solver failed
    return jnp.where(enzyme_failed, -jnp.inf, ll_condition)


def compute_log_posterior_jax_random(
    log_param_array,
    fixed_params_varying,
    initial_array,
    ref_flux_array,
    ref_concs_arrays,
    ref_mids_arrays,
    observation_times,
    t_final,
    prior_mean,
    prior_std,
):
    """JAX-compatible log-posterior for multi-condition MCMC with timecourse MIDs.

    Parameters
    ----------
    log_param_array : jnp.array
        Log-transformed MCMC parameters (14,)
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (n_conditions, 4): [A_e, E_T, Q_e, P_e]
    initial_array : jnp.array
        Initial conditions for enzyme_ode (7,): [A, Q, P, EA, EPQ, EP, EQ]
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

    # vmap over conditions
    condition_lls = jax.vmap(
        lambda A_e, E_T, Q_e, P_e, rf, rc, rm: compute_condition_likelihood_random(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e,
            initial_array, observation_times, t_final
        )
    )(
        fixed_params_varying[:, 0],  # A_e
        fixed_params_varying[:, 1],  # E_T
        fixed_params_varying[:, 2],  # Q_e
        fixed_params_varying[:, 3],  # P_e
        ref_flux_array,
        ref_concs_arrays,
        ref_mids_arrays,
    )

    total_log_likelihood = jnp.sum(condition_lls)

    return log_prior + total_log_likelihood


def run_mcmc_random(
    ref_params,
    initial,
    conditions=None,
    observation_times=None,
    n_warmup=1000,
    n_samples=1000,
    n_chains=2,
    seed=42,
):
    """Run MCMC sampling with instationary labeling data for random mechanism.

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
    n_chains : int
        Number of independent chains to run
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
    print(f"Running MCMC with RANDOM MECHANISM INSTATIONARY labeling ({num_conditions} conditions, {n_chains} chains)")
    print(f"Observation times: {observation_times}")
    print(f"MIDs per time point: {N_MIDS} (A: 3, Q: 2, P: 2)")
    print("=" * 80)
    print(f"\nExperimental conditions:")
    for i, cond in enumerate(conditions):
        Q_e = cond.get("Q_e", ref_params["Q_e"])
        P_e = cond.get("P_e", ref_params["P_e"])
        print(f"  Condition {i+1}: E_T={cond['E_T']:.4f}, A_e={cond['A_e']:.4f}, Q_e={Q_e:.4f}, P_e={P_e:.4f}")
    sys.stdout.flush()

    # Fixed parameters varying per condition: [A_e, E_T, Q_e, P_e]
    fixed_params_varying = jnp.array([
        [cond["A_e"], cond["E_T"],
         cond.get("Q_e", ref_params["Q_e"]),
         cond.get("P_e", ref_params["P_e"])]
        for cond in conditions
    ])

    # Initial conditions array (same for all conditions)
    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"], initial["EQ"]
    ])

    # Generate reference data using ORDERED model (same measurements as enzyme_simulation_instationary.py)
    # This allows us to fit ordered-model-generated data with the random model
    print("\nGenerating reference data for all conditions using ORDERED model...")
    print("  (Data generated with ordered mechanism, fitted with random mechanism)")
    sys.stdout.flush()

    # Use ordered model's default params and initial conditions for reference data
    ordered_params = ORDERED_DEFAULT_PARAMS.copy()
    ordered_params = compute_wegscheider_ordered(ordered_params)
    ordered_initial = ORDERED_DEFAULT_INITIAL.copy()

    ref_data_list = generate_reference_data_ordered(
        ordered_params, ordered_initial, conditions, observation_times
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

    # Generic prior: all true parameter values within 2 standard deviations
    # Range in log-space: -2.3 (k_ex=0.1) to 5.5 (k2r=250)
    # Center at log(5)≈1.6, std=2.0 covers range within 2 std devs
    param_array_ref = params_dict_to_array_random(ref_params)
    n_params = len(param_array_ref)
    prior_mean = jnp.ones(n_params) * 1.6  # log(5) ≈ 1.6
    prior_std = 4.0

    # Extract t_final as concrete value before JIT
    t_final = float(observation_times[-1])

    # Log-posterior function (JIT-compiled for performance)
    @jax.jit
    def logdensity_fn(log_param_array):
        return compute_log_posterior_jax_random(
            log_param_array,
            fixed_params_varying,
            initial_array,
            ref_flux_array,
            ref_concs_arrays,
            ref_mids_arrays,
            observation_times,
            t_final,
            prior_mean,
            prior_std,
        )

    # Initialize
    rng_key = jax.random.PRNGKey(seed)
    initial_position = jnp.log(param_array_ref)

    print(f"\nInitial position (log-space): {initial_position}")
    print(f"Initial position (natural): {param_array_ref}")
    print(f"Initial log-posterior: {logdensity_fn(initial_position)}")
    sys.stdout.flush()

    # NUTS warmup
    print(f"\nRunning {n_warmup} warmup iterations...")
    sys.stdout.flush()

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        num_steps=n_warmup,
        target_acceptance_rate=0.80,
        progress_bar=True,
    )

    rng_key, warmup_key = jax.random.split(rng_key)
    state, kernel, *_ = warmup.run(warmup_key, initial_position)

    print(f"Warmup complete.")
    sys.stdout.flush()

    # Sampling with multiple chains using vmap for parallelism
    print(f"\nCollecting {n_samples} samples from {n_chains} chains (parallel via vmap)...")
    sys.stdout.flush()

    # Define single chain sampling function with progress bar
    def run_single_chain(chain_key, initial_state):
        """Run sampling for a single chain with progress bar."""
        @scan_tqdm(n_samples, print_rate=10)
        def one_step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            state, info = kernel(subkey, state)
            return (state, key), (state.position, -state.potential_energy)

        (final_state, _), (positions, log_probs) = jax.lax.scan(
            one_step, (initial_state, chain_key), jnp.arange(n_samples)
        )
        return positions, log_probs

    # Generate keys for each chain
    chain_keys = jax.random.split(rng_key, n_chains)

    # Use vmap to run all chains in parallel
    run_chains_parallel = jax.vmap(run_single_chain, in_axes=(0, None))
    log_samples, log_probs = run_chains_parallel(chain_keys, state)

    print(f"  All {n_chains} chains complete (parallel execution)")
    sys.stdout.flush()

    # Transform samples back to natural scale
    samples = jnp.exp(log_samples)

    # Compute R-hat for convergence
    def compute_rhat(chains):
        n_chains, n_samples, n_params = chains.shape
        rhats = []
        for p in range(n_params):
            param_chains = chains[:, :, p]
            chain_vars = jnp.var(param_chains, axis=1, ddof=1)
            W = jnp.mean(chain_vars)
            chain_means = jnp.mean(param_chains, axis=1)
            B = n_samples * jnp.var(chain_means, ddof=1)
            var_plus = ((n_samples - 1) * W + B) / n_samples
            rhat = jnp.sqrt(var_plus / W) if W > 0 else jnp.nan
            rhats.append(float(rhat))
        return jnp.array(rhats)

    rhats = compute_rhat(log_samples)

    # Flatten chains for summary
    samples_flat = samples.reshape(-1, samples.shape[-1])

    # Summary
    print("\n" + "=" * 80)
    print("MCMC Results Summary (Random Mechanism - Instationary Labeling)")
    print("=" * 80)

    results = {
        "samples": samples,
        "samples_flat": samples_flat,
        "log_samples": log_samples,
        "log_probs": log_probs,
        "param_names": MCMC_PARAM_NAMES_RANDOM,
        "reference": param_array_ref,
        "n_chains": n_chains,
        "n_samples": n_samples,
        "rhats": np.array(rhats),
    }

    print(f"\nParameter estimates (mean +/- std, R-hat, natural scale, {n_chains} chains x {n_samples} samples):")
    for i, name in enumerate(MCMC_PARAM_NAMES_RANDOM):
        mean = float(jnp.mean(samples_flat[:, i]))
        std = float(jnp.std(samples_flat[:, i]))
        true_val = float(param_array_ref[i])
        rhat = float(rhats[i])
        print(f"  {name:10s}: {mean:8.4f} +/- {std:6.4f}  (true: {true_val:.4f}, R-hat: {rhat:.3f})")

    sys.stdout.flush()

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 80)
    print("Random Mechanism Fit to ORDERED Model Data")
    print("(Reference data from ordered mechanism, fit with random mechanism)")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()

    # Show random mechanism constraints (for reference)
    params = DEFAULT_PARAMS.copy()
    print("\nRandom mechanism Wegscheider constraints:")
    params = compute_wegscheider_constraints_random(params)
    print(f"  k2f = {params['k2f']:.6f} (from overall equilibrium)")
    print(f"  k5br = {params['k5br']:.6f} (from loop closure)")
    print(f"  k_pq_r = {params['k_pq_r']:.6f} (from P-Q equilibrium)")

    # Verify loop closure
    path1, path2, rel_error = verify_wegscheider_loop_closure(params)
    print(f"\nLoop closure verification:")
    print(f"  K4a * K5a = {path1:.6f}")
    print(f"  K4b * K5b = {path2:.6f}")
    print(f"  Relative error: {rel_error:.2e}")
    sys.stdout.flush()

    # Generate reference data using ORDERED model (for preview)
    print("\n" + "-" * 40)
    print("Reference data generated using ORDERED model")
    print("-" * 40)
    ordered_params = ORDERED_DEFAULT_PARAMS.copy()
    ordered_params = compute_wegscheider_ordered(ordered_params)
    ordered_initial = ORDERED_DEFAULT_INITIAL.copy()

    print("\nOrdered model Wegscheider constraints:")
    print(f"  k2f = {ordered_params['k2f']:.6f}")
    print(f"  k_pq_r = {ordered_params['k_pq_r']:.6f}")

    # Import single-condition function for preview
    try:
        from scripts.mcmc.enzyme_simulation_instationary import generate_reference_data_single_condition
    except ModuleNotFoundError:
        from enzyme_simulation_instationary import generate_reference_data_single_condition

    ref_data = generate_reference_data_single_condition(ordered_params, ordered_initial, OBSERVATION_TIMES)

    print("\nSteady-state concentrations (from ordered model):")
    for name, conc in ref_data["concentrations"].items():
        print(f"  {name}: {conc:.6f}")

    print("\nSteady-state fluxes (from ordered model):")
    for name, flux in ref_data["fluxes"].items():
        print(f"  {name}: {flux:.6f}")

    print("\nTime-course MIDs (A, Q, P only, from ordered model):")
    for t, mids in ref_data["timecourse_mids"].items():
        print(f"  t={t:5.1f}: A_MID={mids[:3]}, Q_MID={mids[3:5]}, P_MID={mids[5:7]}")
    sys.stdout.flush()

    # Run MCMC with random model fitting ordered-model data
    print("\n" + "=" * 80)
    print("Running MCMC Parameter Estimation")
    print("(Fitting RANDOM mechanism to ORDERED model data)")
    print("=" * 80)
    sys.stdout.flush()

    mcmc_results = run_mcmc_random(
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
        rhats=mcmc_results["rhats"],
    )
    print(f"\nResults saved to {output_dir / 'mcmc_results.npz'}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Simulation Complete!")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

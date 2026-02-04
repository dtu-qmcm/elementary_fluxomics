#!/usr/bin/env python3
"""Enzyme kinetics ODE simulation with isotopomer labeling analysis.

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

Atom mapping:
  A(ab) <-> EPQ(ab)
  EPQ(ab) <-> EP(a) + Q(b)
  P(a) <-> Q(a)  (interconversion)

Labeling:
  - A_e (external A): 100% labeled on both atoms (isotopomer A_11)
  - Q_e, P_e (external Q, P): 100% unlabeled

Inference approach:
  1. Run enzyme_ode to get metabolic steady-state fluxes and concentrations
  2. Run isotopomer_ode from that steady state (all unlabeled) to get MIDs
  3. Use fluxes/concentrations from step 1 and MIDs from step 2 in log-likelihood
"""

from pathlib import Path
import sys

# Enable float64 precision BEFORE importing JAX
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
import diffrax
from diffrax import SteadyStateEvent
import blackjax
from jax_tqdm import scan_tqdm

try:
    from scripts.isotopomer_simulation import isotopomer_ode, enzyme_ode
    from scripts.emu_jax import (
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
        simulate_ss_mdvs_pure_jax,
        precompute_matrix_structure,
    )
    from scripts.enzyme_utils import (
        DEFAULT_PARAMS,
        DEFAULT_INITIAL,
        EXPERIMENTAL_CONDITIONS,
        NUM_CONDITIONS,
        R_GAS,
        TEMPERATURE,
        FORMATION_ENERGIES,
        MCMC_PARAM_NAMES,
        compute_wegscheider_constraints,
        compute_wegscheider_constraints_jax,
        compute_E,
        compute_rates,
        ode_system,
        simulate_enzyme_to_steady_state_jax,
        simulate_ode_to_steady_state,
        get_steady_state,
        compute_forward_backward_fluxes,
        params_array_to_dict,
        params_dict_to_array,
    )
except ModuleNotFoundError:
    from isotopomer_simulation import isotopomer_ode, enzyme_ode
    from emu_jax import (
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
        simulate_ss_mdvs_pure_jax,
        precompute_matrix_structure,
    )
    from enzyme_utils import (
        DEFAULT_PARAMS,
        DEFAULT_INITIAL,
        EXPERIMENTAL_CONDITIONS,
        NUM_CONDITIONS,
        R_GAS,
        TEMPERATURE,
        FORMATION_ENERGIES,
        MCMC_PARAM_NAMES,
        compute_wegscheider_constraints,
        compute_wegscheider_constraints_jax,
        compute_E,
        compute_rates,
        ode_system,
        simulate_enzyme_to_steady_state_jax,
        simulate_ode_to_steady_state,
        get_steady_state,
        compute_forward_backward_fluxes,
        params_array_to_dict,
        params_dict_to_array,
    )


# Global cache for EMU network (initialized once)
_PURE_JAX_EMU_NETWORK = None


def get_pure_jax_emu_network():
    """Get or initialize the pure JAX EMU network structure.

    This loads the serialized EMU network from JSON for fully JAX-traceable
    simulation compatible with gradient-based MCMC. Also precomputes the
    matrix structure for efficient JAX tracing.

    Returns
    -------
    tuple
        (network, precomputed) where:
        - network: Serialized EMU network structure
        - precomputed: Precomputed matrix structure arrays for JAX tracing
    """
    global _PURE_JAX_EMU_NETWORK

    if _PURE_JAX_EMU_NETWORK is None:
        emu_network_path = PROJECT_ROOT / "models" / "simple_network" / "emu_network.json"
        network = load_emu_network_structure(emu_network_path)
        precomputed = precompute_matrix_structure(network)
        _PURE_JAX_EMU_NETWORK = (network, precomputed)

    return _PURE_JAX_EMU_NETWORK


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
REACTIONS_FILE = PROJECT_ROOT / "models" / "simple_network" / "reactions.tsv"
RESULTS_DIR = PROJECT_ROOT / "results" / "enzyme_simulation"


# Constants, default params, and shared functions imported from enzyme_utils


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def compute_fluxes_for_emu(concs, params, use_jax_arrays=False):
    """Compute forward/backward fluxes for EMU simulation.

    The exchange reactions are modeled as separate in/out fluxes:
    - v1: A_ext -> A (input of labeled substrate)
    - v1_out: A -> A_sink (output, removes isotopomers proportionally)
    - vq_in/vq_out, vp_in/vp_out: Q and P exchanges

    Parameters
    ----------
    concs : dict
        Steady-state concentrations {A, Q, P, EA, EPQ, EP, E}
    params : dict
        Rate constants
    use_jax_arrays : bool
        If True, keep values as JAX arrays (for tracing)

    Returns
    -------
    dict
        Fluxes in format expected by freeflux/EMU
    """
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    E = concs["E"]

    # External exchange fluxes (separate in/out for proper isotope tracking)
    v1 = params["k_A_ex"] * params["A_e"]  # Input from A_ext (labeled)
    v1_out = params["k_A_ex"] * A  # Output to A_sink
    vq_out = params["k_Q_ex"] * Q  # Output to Q_ext
    vq_in = params["k_Q_ex"] * params["Q_e"]  # Input from Q_ext_in (unlabeled)
    vp_out = params["k_P_ex"] * P  # Output to P_ext
    vp_in = params["k_P_ex"] * params["P_e"]  # Input from P_ext_in (unlabeled)

    # Enzyme reactions (forward/backward)
    v2_f = A * E * params["k2f"]
    v2_b = EA * params["k2r"]
    v3_f = EA * params["k3f"]
    v3_b = EPQ * params["k3r"]
    v4_f = EPQ * params["k4f"]
    v4_b = Q * EP * params["k4r"]
    v5_f = EP * params["k5f"]
    v5_b = E * P * params["k5r"]

    # P <-> Q interconversion
    vpq_f = P * params["k_pq_f"]
    vpq_b = Q * params["k_pq_r"]

    fluxes = {
        "v1": v1,
        "v1_out": v1_out,
        "v2_f": v2_f,
        "v2_b": v2_b,
        "v3_f": v3_f,
        "v3_b": v3_b,
        "v4_f": v4_f,
        "v4_b": v4_b,
        "v5_f": v5_f,
        "v5_b": v5_b,
        "vq_out": vq_out,
        "vq_in": vq_in,
        "vp_out": vp_out,
        "vp_in": vp_in,
        "vpq_f": vpq_f,
        "vpq_b": vpq_b,
    }

    # Convert to Python floats if not using JAX arrays (for freeflux compatibility)
    if not use_jax_arrays:
        fluxes = {k: float(v) for k, v in fluxes.items()}

    return fluxes


def simulate_isotopomers_to_steady_state(params, initial_total_array):
    """JAX-compatible isotopomer simulation to steady state.

    Starts from the given total concentrations (all isotopomers unlabeled)
    and runs until isotopic steady state is reached. The boundary condition
    A_e is fully labeled (A_11).

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    initial_total_array : jnp.array
        Initial total concentrations [A, Q, P, EA, EPQ, EP] from enzyme_ode
        steady state. All isotopomers start unlabeled.

    Returns
    -------
    diffrax.Solution
        ODE solution at steady state
    """
    # Build initial isotopomer state (all unlabelled)
    y0 = jnp.zeros(18)
    # A starts unlabelled (A_00 at index 0)
    y0 = y0.at[0].set(initial_total_array[0])
    # EA starts unlabelled (EA_00 at index 4)
    y0 = y0.at[4].set(initial_total_array[3])
    # EPQ starts unlabelled (EPQ_00 at index 8)
    y0 = y0.at[8].set(initial_total_array[4])
    # EP starts unlabelled (EP_0 at index 12)
    y0 = y0.at[12].set(initial_total_array[5])
    # Q starts unlabelled (Q_0 at index 14)
    y0 = y0.at[14].set(initial_total_array[1])
    # P starts unlabelled (P_0 at index 16)
    y0 = y0.at[16].set(initial_total_array[2])

    term = diffrax.ODETerm(isotopomer_ode)
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


def compute_mids_jax(y):
    """JAX-compatible MID computation from isotopomer state.

    Parameters
    ----------
    y : jnp.array
        Isotopomer state vector (18 components)

    Returns
    -------
    dict
        MIDs for each metabolite
    """
    # A: 2 atoms, MID has 3 components (M+0, M+1, M+2)
    A_00, A_01, A_10, A_11 = y[0], y[1], y[2], y[3]
    A_total = A_00 + A_01 + A_10 + A_11
    A_mid = jnp.array([A_00, A_01 + A_10, A_11]) / jnp.maximum(A_total, 1e-10)

    # EA: 2 atoms
    EA_00, EA_01, EA_10, EA_11 = y[4], y[5], y[6], y[7]
    EA_total = EA_00 + EA_01 + EA_10 + EA_11
    EA_mid = jnp.array([EA_00, EA_01 + EA_10, EA_11]) / jnp.maximum(EA_total, 1e-10)

    # EPQ: 2 atoms
    EPQ_00, EPQ_01, EPQ_10, EPQ_11 = y[8], y[9], y[10], y[11]
    EPQ_total = EPQ_00 + EPQ_01 + EPQ_10 + EPQ_11
    EPQ_mid = jnp.array([EPQ_00, EPQ_01 + EPQ_10, EPQ_11]) / jnp.maximum(EPQ_total, 1e-10)

    # EP: 1 atom, MID has 2 components (M+0, M+1)
    EP_0, EP_1 = y[12], y[13]
    EP_total = EP_0 + EP_1
    EP_mid = jnp.array([EP_0, EP_1]) / jnp.maximum(EP_total, 1e-10)

    # Q: 1 atom
    Q_0, Q_1 = y[14], y[15]
    Q_total = Q_0 + Q_1
    Q_mid = jnp.array([Q_0, Q_1]) / jnp.maximum(Q_total, 1e-10)

    # P: 1 atom
    P_0, P_1 = y[16], y[17]
    P_total = P_0 + P_1
    P_mid = jnp.array([P_0, P_1]) / jnp.maximum(P_total, 1e-10)

    return {
        "A_12": A_mid,
        "EA_12": EA_mid,
        "EPQ_12": EPQ_mid,
        "EP_1": EP_mid,
        "Q_1": Q_mid,
        "P_1": P_mid,
    }


def generate_reference_data_single_condition(params, initial):
    """Generate reference observations from a parameter set for a single condition.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run isotopomer_ode from that steady state (all unlabeled) to get MIDs

    The fluxes and concentrations come from enzyme_ode (step 1).
    The MIDs come from isotopomer_ode (step 2) with A_e = A_11 (fully labeled).

    Parameters
    ----------
    params : dict
        Full parameter dictionary
    initial : dict
        Initial conditions

    Returns
    -------
    dict
        Reference data containing fluxes, concentrations, ss_mids
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

    # Compute net fluxes at steady state from enzyme_ode
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
        "v_net": v_A_ex,  # Net flux through the system at steady state
    }

    # Step 2: Run steady-state EMU simulation from enzyme_ode steady state
    # Use the same SS solver as the likelihood function for consistency
    network, precomputed = get_pure_jax_emu_network()
    ss_mids_jax = compute_mids_pure_jax_emu_ss(concs, params, network, precomputed)

    # Convert JAX arrays to numpy for compatibility
    ss_mids_np = {
        "A_12": np.array(ss_mids_jax["A_12"]),
        "Q_1": np.array(ss_mids_jax["Q_1"]),
        "P_1": np.array(ss_mids_jax["P_1"]),
    }

    return {
        "fluxes": net_rates,
        "concentrations": concs,
        "ss_mids": ss_mids_np,
        "enzyme_steady_state": np.array(enzyme_y_ss),  # For initializing isotopomer sim
    }


def generate_reference_data_multi_condition(base_params, initial, conditions):
    """Generate reference data for multiple experimental conditions.

    Parameters
    ----------
    base_params : dict
        Base parameter dictionary (A_e, E_T, Q_e, P_e will be overridden per condition)
    initial : dict
        Initial conditions (same across all conditions)
    conditions : list of dict
        Experimental conditions with keys 'A_e', 'E_T', 'Q_e', 'P_e'

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
        params["Q_e"] = cond.get("Q_e", base_params["Q_e"])
        params["P_e"] = cond.get("P_e", base_params["P_e"])
        ref_data = generate_reference_data_single_condition(params, initial)
        ref_data_list.append(ref_data)
    return ref_data_list


def compute_condition_likelihood(
    A_e,
    E_T,
    ref_flux,
    ref_concs,
    ref_mids,
    param_array,
    Q_e,
    P_e,
    initial_array,
):
    """Compute log-likelihood for a single experimental condition.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run isotopomer_ode from that steady state (all unlabeled) to get MIDs

    Fluxes and concentrations are from enzyme_ode (step 1).
    MIDs are from isotopomer_ode (step 2) with A_e = A_11 (fully labeled).

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
    ref_mids : jnp.array
        Reference MID values (7,): measurable metabolites [A_12(3), Q_1(2), P_1(2)]
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

    # Step 1: Run enzyme_ode to get metabolic steady state
    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Check if enzyme solver failed
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    # Extract concentrations from enzyme_ode steady state (for flux/conc likelihood)
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]

    # Measurable concentrations only (A, Q, P)
    sim_concs = jnp.array([A, Q, P])

    # Compute net flux from enzyme_ode steady state
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1)
    flux_std = 0.1
    ll_flux = norm.logpdf(ref_flux, loc=sim_flux, scale=flux_std)

    # Log-likelihood for concentrations (lognormal distribution, std=0.1)
    conc_std = 0.1
    ll_concs = jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_concs, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_concs, 1e-10)),
        scale=conc_std
    ))

    # Step 2: Run isotopomer_ode from enzyme_ode steady state (all unlabeled)
    iso_sol = simulate_isotopomers_to_steady_state(params, enzyme_y_ss)

    # Check if isotopomer solver failed
    iso_failed = jnp.any(jnp.isnan(iso_sol.ys)) | (iso_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    # Extract steady-state isotopomer state
    iso_y_ss = iso_sol.ys[0] if iso_sol.ys.ndim > 1 else iso_sol.ys

    # Compute simulated MIDs from isotopomer state (measurable metabolites only)
    sim_mids_dict = compute_mids_jax(iso_y_ss)

    sim_mids_array = jnp.concatenate([
        jnp.array(sim_mids_dict["A_12"]),
        jnp.array(sim_mids_dict["Q_1"]),
        jnp.array(sim_mids_dict["P_1"]),
    ])

    # Log-likelihood for MIDs (lognormal distribution, std=0.01)
    # Multiplied by 5 to match information content of instationary data (5 timepoints)
    mid_std = 0.01
    mid_multiplier = 5
    ll_mids = mid_multiplier * jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_mids, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_mids_array, 1e-10)),
        scale=mid_std
    ))

    ll_condition = ll_flux + ll_concs + ll_mids

    # Return -inf if either solver failed
    solver_failed = enzyme_failed | iso_failed
    return jnp.where(solver_failed, -jnp.inf, ll_condition)


def compute_fluxes_array_pure_jax(concs, params, network):
    """Compute flux array for pure JAX EMU simulation.

    Computes forward/backward fluxes in the order expected by the
    serialized EMU network structure.

    Parameters
    ----------
    concs : dict or arrays
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, E
    params : dict
        Rate constants
    network : dict
        Serialized EMU network structure

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
    E = concs["E"]

    # Compute all fluxes (matching reactions.tsv order)
    fluxes_dict = {
        "v1": params["k_A_ex"] * params["A_e"],  # A_ext -> A (input)
        "v1_out": params["k_A_ex"] * A,  # A -> A_sink
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

    # Create array in network order
    return jnp.array([fluxes_dict.get(fid, 0.0) for fid in network["flux_ids_all"]])


def compute_concs_array_pure_jax(concs, network):
    """Compute concentration array for pure JAX EMU simulation.

    Parameters
    ----------
    concs : dict
        Concentration values
    network : dict
        Serialized EMU network structure

    Returns
    -------
    jnp.ndarray
        Concentration array in order matching network["metab_ids_all"]
    """
    return jnp.array([concs.get(mid, 1.0) for mid in network["metab_ids_all"]])


def compute_mids_pure_jax_emu(concs_dict, params, network, timepoints, precomputed=None):
    """Compute MIDs using pure JAX EMU simulation (instationary).

    This is fully JAX-traceable and compatible with gradient-based MCMC.

    Parameters
    ----------
    concs_dict : dict
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, E
    params : dict
        Rate constants including k2f, k_pq_r
    network : dict
        Serialized EMU network structure
    timepoints : jnp.ndarray
        Simulation timepoints (including t=0)
    precomputed : dict, optional
        Precomputed matrix structure arrays for JAX tracing

    Returns
    -------
    dict
        {emu_id: mdv_array} for all product EMUs at final timepoint
    """
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)
    concs_array = compute_concs_array_pure_jax(concs_dict, network)

    return simulate_inst_mdvs_pure_jax(fluxes_array, concs_array, timepoints, network, precomputed)


def compute_mids_pure_jax_emu_ss(concs_dict, params, network, precomputed=None):
    """Compute MIDs using pure JAX steady-state EMU simulation.

    This is fully JAX-traceable and compatible with gradient-based MCMC.
    Simpler and faster than instationary simulation since it directly solves
    the steady-state equation X = pinv(A) @ B @ Y without time integration.

    Parameters
    ----------
    concs_dict : dict
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, E
    params : dict
        Rate constants including k2f, k_pq_r
    network : dict
        Serialized EMU network structure
    precomputed : dict, optional
        Precomputed matrix structure arrays for JAX tracing

    Returns
    -------
    dict
        {emu_id: mdv_array} for all product EMUs at isotopic steady state
    """
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)

    return simulate_ss_mdvs_pure_jax(fluxes_array, network, precomputed)


def compute_condition_likelihood_emu(
    A_e,
    E_T,
    ref_flux,
    ref_concs,
    ref_mids,
    param_array,
    Q_e,
    P_e,
    initial_array,
    emu_network,
    precomputed,
):
    """Compute log-likelihood using pure JAX steady-state EMU simulation.

    This version uses the pure JAX steady-state EMU simulation instead of
    instationary/isotopomer ODE, enabling gradient-based MCMC with simpler
    and faster computation.

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
    ref_mids : jnp.array
        Reference MID values (7,): measurable metabolites [A_12(3), Q_1(2), P_1(2)]
    param_array : jnp.array
        Rate constants (11,): [k2r, k3f, k3r, k4f, k4r, k5f, k5r, k_A_ex, k_Q_ex, k_P_ex, k_pq_f]
    Q_e : float
        External Q concentration (fixed)
    P_e : float
        External P concentration (fixed)
    initial_array : jnp.array
        Initial conditions (6,)
    emu_network : dict
        Serialized EMU network structure
    precomputed : dict
        Precomputed matrix structure arrays for JAX tracing

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

    # Step 1: Run enzyme_ode to get metabolic steady state
    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Check if enzyme solver failed
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    # Extract concentrations from enzyme_ode steady state
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    # Measurable concentrations only (A, Q, P)
    sim_concs = jnp.array([A, Q, P])

    # Compute net flux from enzyme_ode steady state
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1)
    flux_std = 0.1
    ll_flux = norm.logpdf(ref_flux, loc=sim_flux, scale=flux_std)

    # Log-likelihood for concentrations (lognormal distribution, std=0.1)
    conc_std = 0.1
    ll_concs = jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_concs, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_concs, 1e-10)),
        scale=conc_std
    ))

    # Step 2: Run pure JAX steady-state EMU simulation from enzyme_ode steady state
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}
    sim_mids_dict = compute_mids_pure_jax_emu_ss(concs_dict, params, emu_network, precomputed)

    # Extract MIDs for measurable metabolites only (A, Q, P)
    sim_mids_array = jnp.concatenate([
        sim_mids_dict["A_12"],
        sim_mids_dict["Q_1"],
        sim_mids_dict["P_1"],
    ])

    # Log-likelihood for MIDs (lognormal distribution, std=0.01)
    # Multiplied by 5 to match information content of instationary data (5 timepoints)
    mid_std = 0.01
    mid_multiplier = 5
    ll_mids = mid_multiplier * jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_mids, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_mids_array, 1e-10)),
        scale=mid_std
    ))

    ll_condition = ll_flux + ll_concs + ll_mids

    # Return -inf if enzyme solver failed
    return jnp.where(enzyme_failed, -jnp.inf, ll_condition)


def compute_log_posterior_jax(
    log_param_array,
    fixed_params_varying,
    initial_array,
    ref_flux_array,
    ref_concs_arrays,
    ref_mids_arrays,
    prior_mean,
    prior_std,
):
    """JAX-compatible log-posterior for multi-condition MCMC sampling.

    Parameters
    ----------
    log_param_array : jnp.array
        Array of log-transformed MCMC parameters (log rate constants)
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (num_conditions, 4) [[A_e, E_T, Q_e, P_e], ...]
    initial_array : jnp.array
        Initial conditions [A, Q, P, EA, EPQ, EP]
    ref_flux_array : jnp.array
        Reference net flux values (num_conditions,)
    ref_concs_arrays : jnp.array
        Reference concentration values (num_conditions, 3): [A, Q, P]
    ref_mids_arrays : jnp.array
        Reference MID values (num_conditions, 7): measurable metabolites only
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

    # Use vmap to compute likelihood for all conditions in parallel
    condition_lls = jax.vmap(
        lambda A_e, E_T, Q_e, P_e, rf, rc, rm: compute_condition_likelihood(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e, initial_array
        )
    )(
        fixed_params_varying[:, 0],  # A_e values
        fixed_params_varying[:, 1],  # E_T values
        fixed_params_varying[:, 2],  # Q_e values
        fixed_params_varying[:, 3],  # P_e values
        ref_flux_array,
        ref_concs_arrays,
        ref_mids_arrays,
    )

    # Sum likelihoods across conditions
    total_log_likelihood = jnp.sum(condition_lls)

    log_posterior = log_prior + total_log_likelihood
    return log_posterior


def compute_log_posterior_jax_emu(
    log_param_array,
    fixed_params_varying,
    initial_array,
    ref_flux_array,
    ref_concs_arrays,
    ref_mids_arrays,
    prior_mean,
    prior_std,
    emu_network,
    precomputed,
):
    """JAX-compatible log-posterior using pure JAX steady-state EMU simulation.

    This version uses the pure JAX steady-state EMU simulation instead of
    instationary/isotopomer ODE, enabling gradient-based MCMC with full JAX
    tracing and simpler/faster computation.

    Parameters
    ----------
    log_param_array : jnp.array
        Array of log-transformed MCMC parameters (log rate constants)
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (num_conditions, 4) [[A_e, E_T, Q_e, P_e], ...]
    initial_array : jnp.array
        Initial conditions [A, Q, P, EA, EPQ, EP]
    ref_flux_array : jnp.array
        Reference net flux values (num_conditions,)
    ref_concs_arrays : jnp.array
        Reference concentration values (num_conditions, 3): [A, Q, P]
    ref_mids_arrays : jnp.array
        Reference MID values (num_conditions, 7) - measurable metabolites only [A_12(3), Q_1(2), P_1(2)]
    prior_mean : jnp.array
        Prior mean for each parameter (log-scale)
    prior_std : float
        Prior standard deviation (log-scale)
    emu_network : dict
        Serialized EMU network structure
    precomputed : dict
        Precomputed matrix structure arrays for JAX tracing

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

    # Use vmap to compute likelihood for all conditions in parallel
    condition_lls = jax.vmap(
        lambda A_e, E_T, Q_e, P_e, rf, rc, rm: compute_condition_likelihood_emu(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e, initial_array,
            emu_network, precomputed
        )
    )(
        fixed_params_varying[:, 0],  # A_e values
        fixed_params_varying[:, 1],  # E_T values
        fixed_params_varying[:, 2],  # Q_e values
        fixed_params_varying[:, 3],  # P_e values
        ref_flux_array,
        ref_concs_arrays,
        ref_mids_arrays,
    )

    # Sum likelihoods across conditions
    total_log_likelihood = jnp.sum(condition_lls)

    log_posterior = log_prior + total_log_likelihood
    return log_posterior


def run_mcmc(
    ref_params,
    initial,
    conditions=None,
    n_warmup=400,
    n_samples=400,
    n_chains=2,
    seed=42,
    use_pure_jax_emu=True,
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
    n_chains : int
        Number of independent chains to run in parallel
    seed : int
        Random seed
    use_pure_jax_emu : bool
        If True (default), use pure JAX EMU simulation for isotope labeling.
        This enables gradient-based MCMC with full JAX tracing.
        If False, use isotopomer ODE (legacy, slower).

    Returns
    -------
    dict
        MCMC results containing samples and diagnostics
    """
    if conditions is None:
        conditions = EXPERIMENTAL_CONDITIONS

    num_conditions = len(conditions)

    method = "pure JAX steady-state EMU" if use_pure_jax_emu else "isotopomer ODE"
    print("=" * 80)
    print(f"Running MCMC with blackjax NUTS ({num_conditions} conditions, {n_chains} chains, {method})")
    print("=" * 80)
    print(f"\nExperimental conditions:")
    for i, cond in enumerate(conditions):
        print(f"  Condition {i+1}: E_T={cond['E_T']:.4f}, A_e={cond['A_e']:.4f}, Q_e={cond['Q_e']:.4f}, P_e={cond['P_e']:.4f}")
    sys.stdout.flush()

    # All varying fixed parameters per condition: A_e, E_T, Q_e, P_e
    fixed_params_varying = jnp.array([
        [cond["A_e"], cond["E_T"], cond["Q_e"], cond["P_e"]] for cond in conditions
    ])  # Shape: (num_conditions, 4)

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

    # MIDs for measurable metabolites only (A, Q, P - not enzyme intermediates)
    ref_mids_arrays = jnp.array([
        np.concatenate([
            np.array(rd["ss_mids"]["A_12"]),
            np.array(rd["ss_mids"]["Q_1"]),
            np.array(rd["ss_mids"]["P_1"]),
        ])
        for rd in ref_data_list
    ])  # Shape: (num_conditions, 7) - measurable metabolites only

    # Generic prior: all true parameter values within 2 standard deviations
    # Range in log-space: -2.3 (k_ex=0.1) to 5.5 (k2r=250)
    # Center at log(5)≈1.6, std=2.0 covers range within 2 std devs
    param_array_ref = params_dict_to_array(ref_params)
    n_params = len(param_array_ref)
    prior_mean = jnp.ones(n_params) * 1.6  # log(5) ≈ 1.6
    prior_std = 2.0

    # Log-posterior function (JAX-compatible, takes log-transformed params)
    if use_pure_jax_emu:
        # Load serialized EMU network and precomputed structure
        emu_network, precomputed = get_pure_jax_emu_network()

        @jax.jit
        def logdensity_fn(log_param_array):
            return compute_log_posterior_jax_emu(
                log_param_array,
                fixed_params_varying,
                initial_array,
                ref_flux_array,
                ref_concs_arrays,
                ref_mids_arrays,
                prior_mean,
                prior_std,
                emu_network,
                precomputed,
            )
    else:
        @jax.jit
        def logdensity_fn(log_param_array):
            return compute_log_posterior_jax(
                log_param_array,
                fixed_params_varying,
                initial_array,
                ref_flux_array,
                ref_concs_arrays,
                ref_mids_arrays,
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
    # All chains start from the same adapted state
    run_chains_parallel = jax.vmap(run_single_chain, in_axes=(0, None))
    log_samples, log_probs = run_chains_parallel(chain_keys, state)

    print(f"  All {n_chains} chains complete (parallel execution)")

    # Transform samples back to natural scale
    samples = jnp.exp(log_samples)

    # Compute R-hat (Gelman-Rubin diagnostic) for convergence
    # Shape: log_samples is (n_chains, n_samples, n_params)
    def compute_rhat(chains):
        """Compute R-hat for each parameter from multiple chains."""
        n_chains, n_samples, n_params = chains.shape
        rhats = []
        for p in range(n_params):
            # Extract chains for this parameter
            param_chains = chains[:, :, p]  # (n_chains, n_samples)

            # Within-chain variance
            chain_vars = jnp.var(param_chains, axis=1, ddof=1)  # (n_chains,)
            W = jnp.mean(chain_vars)

            # Between-chain variance
            chain_means = jnp.mean(param_chains, axis=1)  # (n_chains,)
            B = n_samples * jnp.var(chain_means, ddof=1)

            # Pooled variance estimate
            var_plus = ((n_samples - 1) * W + B) / n_samples

            # R-hat
            rhat = jnp.sqrt(var_plus / W) if W > 0 else jnp.nan
            rhats.append(float(rhat))
        return jnp.array(rhats)

    rhats = compute_rhat(log_samples)

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("MCMC Results Summary")
    print("=" * 80)

    # Flatten chains for summary: shape (n_chains * n_samples, n_params)
    samples_flat = samples.reshape(-1, samples.shape[-1])
    log_samples_flat = log_samples.reshape(-1, log_samples.shape[-1])
    log_probs_flat = log_probs.reshape(-1)

    results = {
        "samples": samples,  # Natural scale, shape (n_chains, n_samples, n_params)
        "samples_flat": samples_flat,  # Flattened, shape (n_chains * n_samples, n_params)
        "log_samples": log_samples,  # Log scale
        "log_probs": log_probs,
        "param_names": MCMC_PARAM_NAMES,
        "reference": param_array_ref,
        "n_chains": n_chains,
        "n_samples": n_samples,
        "rhats": np.array(rhats),
    }

    print(f"\nParameter estimates (mean ± std, R-hat, natural scale, {n_chains} chains × {n_samples} samples):")
    for i, name in enumerate(MCMC_PARAM_NAMES):
        mean = float(jnp.mean(samples_flat[:, i]))
        std = float(jnp.std(samples_flat[:, i]))
        true_val = float(param_array_ref[i])
        rhat = float(rhats[i])
        print(f"  {name:10s}: {mean:8.4f} ± {std:6.4f}  (true: {true_val:.4f}, R-hat: {rhat:.3f})")

    sys.stdout.flush()

    return results


def print_results(concs, net_rates, fluxes, ss_mdvs):
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

    print("\n" + "=" * 80)
    print("Steady-State Labeling (MID)")
    print("=" * 80)
    for emu_id, mdv in ss_mdvs.items():
        print(f"  {emu_id}: {mdv}")
    sys.stdout.flush()


def main():
    print("=" * 80)
    print("Enzyme Kinetics ODE Simulation with Isotope Labeling Analysis")
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

    if sol.result not in (diffrax.RESULTS.successful, diffrax.RESULTS.discrete_terminating_event_occurred):
        print(f"  ERROR: ODE integration failed: {sol.result}")
        sys.exit(1)

    print(f"  ODE integration successful (diffrax, steady-state event)")
    sys.stdout.flush()

    # Step 3: Extract steady state
    print("\n" + "=" * 80)
    print("Step 3: Extracting Steady-State Values")
    print("=" * 80)
    sys.stdout.flush()

    concs, net_rates = get_steady_state(sol, params)
    fluxes = compute_forward_backward_fluxes(concs, params)

    # Step 4: Run EMU simulation to steady state
    print("\n" + "=" * 80)
    print("Step 4: Running EMU Simulation to Steady State")
    print("=" * 80)
    sys.stdout.flush()

    ref_data = generate_reference_data_single_condition(params, initial)
    ss_mdvs = ref_data["ss_mids"]

    print(f"  EMU simulation complete")
    print(f"  Simulated EMUs: {list(ss_mdvs.keys())}")
    sys.stdout.flush()

    # Print results
    print_results(concs, net_rates, fluxes, ss_mdvs)

    # Step 5: Run MCMC parameter estimation
    print("\n" + "=" * 80)
    print("Step 5: Running MCMC Parameter Estimation")
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

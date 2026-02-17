#!/usr/bin/env python3
"""Enzyme kinetics ODE simulation with INSTATIONARY isotopomer labeling analysis.

This script performs MCMC parameter estimation using time-course MID data instead
of steady-state MIDs. The system starts at metabolic steady state but isotopically
unlabeled, then labeled substrate is introduced at t=0 and MIDs are measured at
multiple time points during the transient.

Key differences from enzyme_simulation.py (steady-state):
- EMU simulation saves at specified time points, not just steady state
- Likelihood uses time-course MIDs at multiple observation times
- More information content from transient dynamics

Labeling:
  - A_e (external A): 100% labeled on both atoms (isotopomer A_11)
  - Q_e, P_e (external Q, P): 100% unlabeled

Inference approach:
  1. Run enzyme_ode to get metabolic steady-state fluxes and concentrations
  2. Run pure JAX EMU simulation for time-course MIDs
  3. Use fluxes/concentrations from step 1 and time-course MIDs from step 2 in log-likelihood

Only A, Q, P MIDs are used (measurable metabolites), not enzyme intermediates.
Uses pure JAX EMU simulation for gradient-based MCMC compatibility.
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
import blackjax
from jax_tqdm import scan_tqdm

try:
    from scripts.emu_jax import (
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
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
        simulate_enzyme_to_steady_state_jax,
        compute_rates,
        ode_system,
        simulate_ode_to_steady_state,
        params_array_to_dict,
        params_dict_to_array,
    )
except ModuleNotFoundError:
    from emu_jax import (
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
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
        simulate_enzyme_to_steady_state_jax,
        compute_rates,
        ode_system,
        simulate_ode_to_steady_state,
        params_array_to_dict,
        params_dict_to_array,
    )


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
REACTIONS_FILE = PROJECT_ROOT / "models" / "simple_network" / "reactions.tsv"
RESULTS_DIR = PROJECT_ROOT / "results" / "enzyme_simulation_instationary"


# Observation time points for instationary labeling (5 log-spaced from 10^-1 to 10^2)
OBSERVATION_TIMES = jnp.logspace(-1, 2, 5)
N_TIMEPOINTS = len(OBSERVATION_TIMES)

# Number of MID values per time point (A: 3, Q: 2, P: 2 = 7 total)
N_MIDS = 7



# Constants, default params, and shared functions imported from enzyme_utils


# Global cache for EMU network
_PURE_JAX_EMU_NETWORK = None


def get_pure_jax_emu_network():
    """Get or initialize the pure JAX EMU network structure.

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


def compute_fluxes_array_pure_jax(concs, params, network):
    """Compute flux array for pure JAX EMU simulation.

    Parameters
    ----------
    concs : dict
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
        "v1_out": params["k_A_ex"] * A,  # A -> A_sink (output)
        "v2_f": A * E * params["k2f"],
        "v2_b": EA * params["k2r"],
        "v3_f": EA * params["k3f"],
        "v3_b": EPQ * params["k3r"],
        "v4_f": EPQ * params["k4f"],
        "v4_b": Q * EP * params["k4r"],
        "v5_f": EP * params["k5f"],
        "v5_b": E * P * params["k5r"],
        "vq_out": params["k_Q_ex"] * Q,  # Q -> Q_ext (output)
        "vq_in": params["k_Q_ex"] * params["Q_e"],  # Q_ext_in -> Q (input)
        "vp_out": params["k_P_ex"] * P,  # P -> P_ext (output)
        "vp_in": params["k_P_ex"] * params["P_e"],  # P_ext_in -> P (input)
        "vpq_f": P * params["k_pq_f"],
        "vpq_b": Q * params["k_pq_r"],
    }

    # Build array in network order
    return jnp.array([fluxes_dict[fid] for fid in network["flux_ids_all"]])


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


def extract_measurable_mids_from_emu(emu_mdvs):
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


def simulate_emu_timecourse(params, concs_dict, observation_times, network, precomputed):
    """Simulate EMU dynamics and return MIDs at specified times.

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    concs_dict : dict
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, E
    observation_times : jnp.array
        Time points at which to observe MIDs
    network : dict
        Serialized EMU network structure
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
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)
    concs_array = compute_concs_array_pure_jax(concs_dict, network)

    # Run pure JAX EMU simulation
    emu_mdvs = simulate_inst_mdvs_pure_jax(
        fluxes_array, concs_array, all_timepoints, network, precomputed
    )

    # Extract measurable MIDs at final timepoint
    # Note: simulate_inst_mdvs_pure_jax returns MIDs at the final timepoint
    # For time-course, we need to run multiple simulations or modify the function
    # For now, we'll compute MIDs at each observation time separately
    timecourse_mids = {}
    for t in observation_times:
        t_float = float(t)
        timepoints_to_t = jnp.array([0.0, t_float])
        emu_mdvs_at_t = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, network, precomputed
        )
        timecourse_mids[t_float] = extract_measurable_mids_from_emu(emu_mdvs_at_t)

    return timecourse_mids


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def generate_reference_data_single_condition(params, initial, observation_times):
    """Generate reference observations including time-course MIDs.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run pure JAX EMU simulation for time-course MIDs

    The fluxes and concentrations come from enzyme_ode (step 1).
    The time-course MIDs come from EMU simulation (step 2) with A_ext fully labeled.

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

    # Step 2: Run pure JAX EMU simulation for time-course MIDs
    network, precomputed = get_pure_jax_emu_network()
    timecourse_mids = simulate_emu_timecourse(
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


def generate_reference_data_multi_condition(base_params, initial, conditions, observation_times):
    """Generate reference data for multiple experimental conditions."""
    ref_data_list = []
    for cond in conditions:
        params = base_params.copy()
        params["A_e"] = cond["A_e"]
        params["E_T"] = cond["E_T"]
        params["Q_e"] = cond.get("Q_e", base_params["Q_e"])
        params["P_e"] = cond.get("P_e", base_params["P_e"])
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
    emu_network,
    precomputed,
):
    """Compute log-likelihood for a single condition with timecourse MIDs.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run pure JAX EMU simulation for time-course MIDs

    Fluxes and concentrations are from enzyme_ode (step 1).
    Time-course MIDs are from EMU simulation (step 2) with A_ext fully labeled.

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
    emu_network : dict
        Serialized EMU network structure
    precomputed : dict
        Precomputed matrix structure arrays for JAX tracing

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
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    sim_concs = jnp.array([A, Q, P])

    # Build concentration dict for EMU
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    # Net flux from enzyme_ode steady state
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1% of |ref_flux|)
    flux_std = 0.01 * jnp.abs(ref_flux)
    ll_flux = norm.logpdf(ref_flux, loc=sim_flux, scale=flux_std)

    # Log-likelihood for concentrations (lognormal)
    conc_std = 0.01
    ll_concs = jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_concs, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_concs, 1e-10)),
        scale=conc_std
    ))

    # Step 2: Run pure JAX EMU simulation for time-course MIDs
    # Compute fluxes and concentrations arrays
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, emu_network)
    concs_array = compute_concs_array_pure_jax(concs_dict, emu_network)

    # Log-likelihood for time-course MIDs (lognormal)
    mid_std = 0.01

    # Compute MID likelihood for each timepoint
    def compute_timepoint_ll(t_idx, ll_acc):
        t = observation_times[t_idx]
        timepoints_to_t = jnp.array([0.0, t])
        emu_mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, emu_network, precomputed
        )
        sim_mids_t = extract_measurable_mids_from_emu(emu_mdvs)
        ref_mids_t = ref_mids_timecourse[t_idx]

        ll_t = jnp.sum(norm.logpdf(
            jnp.log(jnp.maximum(ref_mids_t, 1e-10)),
            loc=jnp.log(jnp.maximum(sim_mids_t, 1e-10)),
            scale=mid_std
        ))
        return ll_acc + ll_t

    # Use fori_loop for JAX-compatible iteration
    ll_mids = jax.lax.fori_loop(0, len(observation_times), compute_timepoint_ll, 0.0)

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
    observation_times,
    prior_mean,
    prior_std,
    emu_network,
    precomputed,
):
    """JAX-compatible log-posterior for multi-condition MCMC with timecourse MIDs.

    Parameters
    ----------
    log_param_array : jnp.array
        Log-transformed MCMC parameters
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (n_conditions, 4): [A_e, E_T, Q_e, P_e]
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
    emu_network : dict
        Serialized EMU network structure
    precomputed : dict
        Precomputed matrix structure arrays for JAX tracing

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
        lambda A_e, E_T, Q_e, P_e, rf, rc, rm: compute_condition_likelihood(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e,
            initial_array, observation_times, emu_network, precomputed
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


def run_mcmc(
    ref_params,
    initial,
    conditions=None,
    observation_times=None,
    n_warmup=1000,
    n_samples=1000,
    n_chains=2,
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
    print(f"Running MCMC with INSTATIONARY isotope labeling ({num_conditions} conditions, {n_chains} chains)")
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

    # Load EMU network
    print("\nLoading EMU network...")
    sys.stdout.flush()
    emu_network, precomputed = get_pure_jax_emu_network()
    print(f"  EMU network loaded with {len(emu_network['flux_ids_all'])} fluxes")
    sys.stdout.flush()

    # Generic prior: all true parameter values within 2 standard deviations
    # Range in log-space: -2.3 (k_ex=0.1) to 5.5 (k2r=250)
    # Center at log(5)≈1.6, std=2.0 covers range within 2 std devs
    param_array_ref = params_dict_to_array(ref_params)
    n_params = len(param_array_ref)
    prior_mean = jnp.ones(n_params) * 1.6  # log(5) ≈ 1.6
    prior_std = 4.0

    # Log-posterior function (JIT-compiled for performance)
    @jax.jit
    def logdensity_fn(log_param_array):
        return compute_log_posterior_jax(
            log_param_array,
            fixed_params_varying,
            initial_array,
            ref_flux_array,
            ref_concs_arrays,
            ref_mids_arrays,
            observation_times,
            prior_mean,
            prior_std,
            emu_network,
            precomputed,
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
    # All chains start from the same adapted state
    run_chains_parallel = jax.vmap(run_single_chain, in_axes=(0, None))
    log_samples, log_probs = run_chains_parallel(chain_keys, state)

    print(f"  All {n_chains} chains complete (parallel execution)")
    sys.stdout.flush()

    # Transform samples back to natural scale
    # Shape: (n_chains, n_samples, n_params)
    samples = jnp.exp(log_samples)

    # Compute R-hat (Gelman-Rubin diagnostic) for convergence
    def compute_rhat(chains):
        """Compute R-hat for each parameter from multiple chains."""
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

    # Flatten chains for summary: shape (n_chains * n_samples, n_params)
    samples_flat = samples.reshape(-1, samples.shape[-1])

    # Summary
    print("\n" + "=" * 80)
    print("MCMC Results Summary (Instationary Labeling)")
    print("=" * 80)

    results = {
        "samples": samples,  # Shape (n_chains, n_samples, n_params)
        "samples_flat": samples_flat,  # Shape (n_chains * n_samples, n_params)
        "log_samples": log_samples,
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

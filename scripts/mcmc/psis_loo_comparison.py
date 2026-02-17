#!/usr/bin/env python3
"""PSIS-LOO model comparison for ordered vs random enzyme mechanisms.

This script computes PSIS-LOO (Pareto-Smoothed Importance Sampling Leave-One-Out)
cross-validation using ArviZ to compare the ordered and random enzyme mechanism
models fitted to the same experimental data.

Key steps:
1. Load MCMC results for both models
2. Regenerate reference data (same for both models)
3. Compute pointwise log-likelihoods for all observations
4. Create ArviZ InferenceData objects
5. Run LOO comparison

Observations per condition (39 total):
- 1 flux (Normal, σ=0.1)
- 3 concentrations (Lognormal, σ=0.1)
- 35 MIDs (7 MIDs × 5 timepoints, Lognormal, σ=0.01)

Total observations: 10 conditions × 39 = 390
"""

from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import arviz as az
import xarray as xr
from jax_tqdm import scan_tqdm

# Import from ordered model script
try:
    from scripts.enzyme_simulation_instationary import (
        generate_reference_data_multi_condition,
        EXPERIMENTAL_CONDITIONS,
        OBSERVATION_TIMES,
        DEFAULT_PARAMS as ORDERED_DEFAULT_PARAMS,
        DEFAULT_INITIAL as ORDERED_DEFAULT_INITIAL,
        compute_wegscheider_constraints,
        MCMC_PARAM_NAMES,
        get_pure_jax_emu_network,
        compute_fluxes_array_pure_jax,
        compute_concs_array_pure_jax,
        extract_measurable_mids_from_emu,
        simulate_enzyme_to_steady_state_jax,
        compute_E,
        FORMATION_ENERGIES,
        TEMPERATURE,
        R_GAS,
        compute_wegscheider_constraints_jax,
    )
    from scripts.enzyme_simulation_random_instationary import (
        MCMC_PARAM_NAMES_RANDOM,
        get_pure_jax_emu_network_random,
        compute_fluxes_array_pure_jax_random,
        compute_concs_array_pure_jax_random,
        extract_measurable_mids_from_emu_random,
        simulate_enzyme_to_steady_state_random_jax,
        compute_E_random,
        compute_wegscheider_constraints_random_jax,
        DEFAULT_PARAMS as RANDOM_DEFAULT_PARAMS,
        DEFAULT_INITIAL as RANDOM_DEFAULT_INITIAL,
    )
    from scripts.emu_jax import simulate_inst_mdvs_pure_jax
except ModuleNotFoundError:
    from enzyme_simulation_instationary import (
        generate_reference_data_multi_condition,
        EXPERIMENTAL_CONDITIONS,
        OBSERVATION_TIMES,
        DEFAULT_PARAMS as ORDERED_DEFAULT_PARAMS,
        DEFAULT_INITIAL as ORDERED_DEFAULT_INITIAL,
        compute_wegscheider_constraints,
        MCMC_PARAM_NAMES,
        get_pure_jax_emu_network,
        compute_fluxes_array_pure_jax,
        compute_concs_array_pure_jax,
        extract_measurable_mids_from_emu,
        simulate_enzyme_to_steady_state_jax,
        compute_E,
        FORMATION_ENERGIES,
        TEMPERATURE,
        R_GAS,
        compute_wegscheider_constraints_jax,
    )
    from enzyme_simulation_random_instationary import (
        MCMC_PARAM_NAMES_RANDOM,
        get_pure_jax_emu_network_random,
        compute_fluxes_array_pure_jax_random,
        compute_concs_array_pure_jax_random,
        extract_measurable_mids_from_emu_random,
        simulate_enzyme_to_steady_state_random_jax,
        compute_E_random,
        compute_wegscheider_constraints_random_jax,
        DEFAULT_PARAMS as RANDOM_DEFAULT_PARAMS,
        DEFAULT_INITIAL as RANDOM_DEFAULT_INITIAL,
    )
    from emu_jax import simulate_inst_mdvs_pure_jax

import diffrax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "psis_loo_comparison"

# Likelihood standard deviations (must match MCMC scripts)
FLUX_STD = 0.1
CONC_STD = 0.1
MID_STD = 0.01

N_TIMEPOINTS = len(OBSERVATION_TIMES)
N_MIDS = 7  # A: 3, Q: 2, P: 2
N_CONDITIONS = len(EXPERIMENTAL_CONDITIONS)

# Observations per condition: 1 flux + 3 concs + 35 MIDs = 39
N_OBS_PER_CONDITION = 1 + 3 + N_MIDS * N_TIMEPOINTS


def compute_pointwise_ll_ordered_single(
    param_array,
    A_e,
    E_T,
    Q_e,
    P_e,
    ref_flux,
    ref_concs,
    ref_mids_timecourse,
    initial_array,
    observation_times,
    emu_network,
    precomputed,
):
    """Compute pointwise log-likelihoods for a single condition (ordered model).

    Returns individual log-likelihoods for each observation:
    - 1 flux
    - 3 concentrations
    - 35 MIDs (7 × 5 timepoints)

    Returns
    -------
    jnp.ndarray
        Shape (39,) - individual log-likelihoods
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
    k2f, k_pq_r = compute_wegscheider_constraints_jax(
        params, formation_energies_array, TEMPERATURE
    )
    params["k2f"] = k2f
    params["k_pq_r"] = k_pq_r

    # Run enzyme ODE to steady state
    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Check for solver failure
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (
        enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred
    )

    # Extract concentrations
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    sim_concs = jnp.array([A, Q, P])
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    # Net flux
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Flux log-likelihood (normal)
    flux_residual = (ref_flux - sim_flux) / FLUX_STD
    ll_flux = -0.5 * flux_residual**2 - jnp.log(FLUX_STD * jnp.sqrt(2 * jnp.pi))

    # Concentration log-likelihoods (lognormal) - individual
    log_ref_concs = jnp.log(jnp.maximum(ref_concs, 1e-10))
    log_sim_concs = jnp.log(jnp.maximum(sim_concs, 1e-10))
    conc_residuals = (log_ref_concs - log_sim_concs) / CONC_STD
    ll_concs = -0.5 * conc_residuals**2 - jnp.log(CONC_STD * jnp.sqrt(2 * jnp.pi))

    # EMU simulation for MIDs
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, emu_network)
    concs_array = compute_concs_array_pure_jax(concs_dict, emu_network)

    # MID log-likelihoods for each timepoint
    ll_mids_list = []
    for t_idx in range(len(observation_times)):
        t = observation_times[t_idx]
        timepoints_to_t = jnp.array([0.0, t])
        emu_mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, emu_network, precomputed
        )
        sim_mids_t = extract_measurable_mids_from_emu(emu_mdvs)
        ref_mids_t = ref_mids_timecourse[t_idx]

        log_ref = jnp.log(jnp.maximum(ref_mids_t, 1e-10))
        log_sim = jnp.log(jnp.maximum(sim_mids_t, 1e-10))
        residuals = (log_ref - log_sim) / MID_STD
        ll_mids_t = -0.5 * residuals**2 - jnp.log(MID_STD * jnp.sqrt(2 * jnp.pi))
        ll_mids_list.append(ll_mids_t)

    # Stack MID log-likelihoods: (n_timepoints, 7) -> flatten to (35,)
    ll_mids = jnp.concatenate(ll_mids_list)

    # Combine all: [flux(1), concs(3), mids(35)] = 39
    ll_all = jnp.concatenate([jnp.array([ll_flux]), ll_concs, ll_mids])

    # Set to -inf if solver failed
    return jnp.where(enzyme_failed, jnp.full_like(ll_all, -jnp.inf), ll_all)


def compute_pointwise_ll_random_single(
    param_array,
    A_e,
    E_T,
    Q_e,
    P_e,
    ref_flux,
    ref_concs,
    ref_mids_timecourse,
    initial_array,
    observation_times,
    emu_network,
    precomputed,
):
    """Compute pointwise log-likelihoods for a single condition (random model).

    Returns
    -------
    jnp.ndarray
        Shape (39,) - individual log-likelihoods
    """
    # Build parameter dict for random mechanism
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

    # Run enzyme ODE to steady state (random mechanism has 7 states)
    enzyme_sol = simulate_enzyme_to_steady_state_random_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Check for solver failure
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (
        enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred
    )

    # Extract concentrations (random has EQ as well)
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    EQ = enzyme_y_ss[6]
    E = params["E_T"] - EA - EPQ - EP - EQ

    sim_concs = jnp.array([A, Q, P])
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "EQ": EQ, "E": E}

    # Net flux
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Flux log-likelihood (normal)
    flux_residual = (ref_flux - sim_flux) / FLUX_STD
    ll_flux = -0.5 * flux_residual**2 - jnp.log(FLUX_STD * jnp.sqrt(2 * jnp.pi))

    # Concentration log-likelihoods (lognormal)
    log_ref_concs = jnp.log(jnp.maximum(ref_concs, 1e-10))
    log_sim_concs = jnp.log(jnp.maximum(sim_concs, 1e-10))
    conc_residuals = (log_ref_concs - log_sim_concs) / CONC_STD
    ll_concs = -0.5 * conc_residuals**2 - jnp.log(CONC_STD * jnp.sqrt(2 * jnp.pi))

    # EMU simulation for MIDs
    fluxes_array = compute_fluxes_array_pure_jax_random(concs_dict, params, emu_network)
    concs_array = compute_concs_array_pure_jax_random(concs_dict, emu_network)

    # MID log-likelihoods for each timepoint
    ll_mids_list = []
    for t_idx in range(len(observation_times)):
        t = observation_times[t_idx]
        timepoints_to_t = jnp.array([0.0, t])
        emu_mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, emu_network, precomputed
        )
        sim_mids_t = extract_measurable_mids_from_emu_random(emu_mdvs)
        ref_mids_t = ref_mids_timecourse[t_idx]

        log_ref = jnp.log(jnp.maximum(ref_mids_t, 1e-10))
        log_sim = jnp.log(jnp.maximum(sim_mids_t, 1e-10))
        residuals = (log_ref - log_sim) / MID_STD
        ll_mids_t = -0.5 * residuals**2 - jnp.log(MID_STD * jnp.sqrt(2 * jnp.pi))
        ll_mids_list.append(ll_mids_t)

    # Stack MID log-likelihoods
    ll_mids = jnp.concatenate(ll_mids_list)

    # Combine all
    ll_all = jnp.concatenate([jnp.array([ll_flux]), ll_concs, ll_mids])

    return jnp.where(enzyme_failed, jnp.full_like(ll_all, -jnp.inf), ll_all)


def compute_all_pointwise_ll_ordered(
    samples,
    ref_data_list,
    conditions,
    observation_times,
    initial_array,
):
    """Compute pointwise log-likelihoods for all samples (ordered model).

    Parameters
    ----------
    samples : jnp.ndarray
        Shape (n_chains, n_samples, n_params)
    conditions : list of dict
        Experimental conditions with A_e, E_T, Q_e, P_e

    Returns
    -------
    jnp.ndarray
        Shape (n_chains, n_samples, n_conditions * n_obs_per_condition)
    """
    n_chains, n_samples, n_params = samples.shape
    n_conditions = len(conditions)
    n_obs = n_conditions * N_OBS_PER_CONDITION

    # Load EMU network
    emu_network, precomputed = get_pure_jax_emu_network()

    # Stack reference data
    ref_flux_array = jnp.array([rd["fluxes"]["v_net"] for rd in ref_data_list])
    ref_concs_arrays = jnp.array([
        [rd["concentrations"]["A"], rd["concentrations"]["Q"], rd["concentrations"]["P"]]
        for rd in ref_data_list
    ])
    ref_mids_arrays = jnp.array([
        jnp.array([rd["timecourse_mids"][float(t)] for t in observation_times])
        for rd in ref_data_list
    ])

    # Condition parameters (now including Q_e and P_e per condition)
    A_e_array = jnp.array([c["A_e"] for c in conditions])
    E_T_array = jnp.array([c["E_T"] for c in conditions])
    Q_e_array = jnp.array([c.get("Q_e", ORDERED_DEFAULT_PARAMS["Q_e"]) for c in conditions])
    P_e_array = jnp.array([c.get("P_e", ORDERED_DEFAULT_PARAMS["P_e"]) for c in conditions])

    print(f"Computing pointwise log-likelihoods for ordered model...")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Total observations: {n_obs}")
    sys.stdout.flush()

    # JIT compile the single-sample function
    @jax.jit
    def compute_for_single_sample(param_array):
        """Compute ll for all conditions for a single sample."""
        ll_all_conditions = []
        for c_idx in range(n_conditions):
            ll_condition = compute_pointwise_ll_ordered_single(
                param_array,
                A_e_array[c_idx],
                E_T_array[c_idx],
                Q_e_array[c_idx],
                P_e_array[c_idx],
                ref_flux_array[c_idx],
                ref_concs_arrays[c_idx],
                ref_mids_arrays[c_idx],
                initial_array,
                observation_times,
                emu_network,
                precomputed,
            )
            ll_all_conditions.append(ll_condition)
        return jnp.concatenate(ll_all_conditions)

    # Process all samples
    all_ll = np.zeros((n_chains, n_samples, n_obs))

    for chain_idx in range(n_chains):
        print(f"  Processing chain {chain_idx + 1}/{n_chains}...")
        sys.stdout.flush()
        for sample_idx in range(n_samples):
            if sample_idx % 50 == 0:
                print(f"    Sample {sample_idx}/{n_samples}")
                sys.stdout.flush()
            param_array = samples[chain_idx, sample_idx]
            ll = compute_for_single_sample(param_array)
            all_ll[chain_idx, sample_idx] = np.array(ll)

    return jnp.array(all_ll)


def compute_all_pointwise_ll_random(
    samples,
    ref_data_list,
    conditions,
    observation_times,
    initial_array,
):
    """Compute pointwise log-likelihoods for all samples (random model).

    Parameters
    ----------
    samples : jnp.ndarray
        Shape (n_chains, n_samples, n_params)
    conditions : list of dict
        Experimental conditions with A_e, E_T, Q_e, P_e

    Returns
    -------
    jnp.ndarray
        Shape (n_chains, n_samples, n_conditions * n_obs_per_condition)
    """
    n_chains, n_samples, n_params = samples.shape
    n_conditions = len(conditions)
    n_obs = n_conditions * N_OBS_PER_CONDITION

    # Load EMU network for random mechanism
    emu_network, precomputed = get_pure_jax_emu_network_random()

    # Stack reference data
    ref_flux_array = jnp.array([rd["fluxes"]["v_net"] for rd in ref_data_list])
    ref_concs_arrays = jnp.array([
        [rd["concentrations"]["A"], rd["concentrations"]["Q"], rd["concentrations"]["P"]]
        for rd in ref_data_list
    ])
    ref_mids_arrays = jnp.array([
        jnp.array([rd["timecourse_mids"][float(t)] for t in observation_times])
        for rd in ref_data_list
    ])

    # Condition parameters (now including Q_e and P_e per condition)
    A_e_array = jnp.array([c["A_e"] for c in conditions])
    E_T_array = jnp.array([c["E_T"] for c in conditions])
    Q_e_array = jnp.array([c.get("Q_e", ORDERED_DEFAULT_PARAMS["Q_e"]) for c in conditions])
    P_e_array = jnp.array([c.get("P_e", ORDERED_DEFAULT_PARAMS["P_e"]) for c in conditions])

    print(f"Computing pointwise log-likelihoods for random model...")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Total observations: {n_obs}")
    sys.stdout.flush()

    @jax.jit
    def compute_for_single_sample(param_array):
        """Compute ll for all conditions for a single sample."""
        ll_all_conditions = []
        for c_idx in range(n_conditions):
            ll_condition = compute_pointwise_ll_random_single(
                param_array,
                A_e_array[c_idx],
                E_T_array[c_idx],
                Q_e_array[c_idx],
                P_e_array[c_idx],
                ref_flux_array[c_idx],
                ref_concs_arrays[c_idx],
                ref_mids_arrays[c_idx],
                initial_array,
                observation_times,
                emu_network,
                precomputed,
            )
            ll_all_conditions.append(ll_condition)
        return jnp.concatenate(ll_all_conditions)

    # Process all samples
    all_ll = np.zeros((n_chains, n_samples, n_obs))

    for chain_idx in range(n_chains):
        print(f"  Processing chain {chain_idx + 1}/{n_chains}...")
        sys.stdout.flush()
        for sample_idx in range(n_samples):
            if sample_idx % 50 == 0:
                print(f"    Sample {sample_idx}/{n_samples}")
                sys.stdout.flush()
            param_array = samples[chain_idx, sample_idx]
            ll = compute_for_single_sample(param_array)
            all_ll[chain_idx, sample_idx] = np.array(ll)

    return jnp.array(all_ll)


# =============================================================================
# Posterior Predictive Simulation Functions
# =============================================================================


def simulate_posterior_predictive_ordered_single(
    param_array,
    A_e,
    E_T,
    Q_e,
    P_e,
    initial_array,
    observation_times,
    emu_network,
    precomputed,
):
    """Simulate flux, concentrations, and MID timecourse for a single ordered model sample.

    Parameters
    ----------
    param_array : jnp.ndarray
        Shape (11,) - MCMC parameters
    A_e, E_T, Q_e, P_e : float
        Experimental conditions
    initial_array : jnp.ndarray
        Initial concentrations [A, Q, P, EA, EPQ, EP]
    observation_times : jnp.ndarray
        Time points for MID measurements
    emu_network : dict
        EMU network structure
    precomputed : dict
        Precomputed matrix structure

    Returns
    -------
    tuple
        (flux, concs, mids_timecourse) where:
        - flux: scalar (net flux)
        - concs: (3,) array [A, Q, P]
        - mids_timecourse: (n_timepoints, 7) array
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
    k2f, k_pq_r = compute_wegscheider_constraints_jax(
        params, formation_energies_array, TEMPERATURE
    )
    params["k2f"] = k2f
    params["k_pq_r"] = k_pq_r

    # Run enzyme ODE to steady state
    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Extract concentrations
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    sim_concs = jnp.array([A, Q, P])
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    # Net flux
    sim_flux = params["k_A_ex"] * (A_e - A)

    # EMU simulation for MIDs at each timepoint
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, emu_network)
    concs_array = compute_concs_array_pure_jax(concs_dict, emu_network)

    mids_list = []
    for t_idx in range(len(observation_times)):
        t = observation_times[t_idx]
        timepoints_to_t = jnp.array([0.0, t])
        emu_mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, emu_network, precomputed
        )
        sim_mids_t = extract_measurable_mids_from_emu(emu_mdvs)
        mids_list.append(sim_mids_t)

    mids_timecourse = jnp.stack(mids_list)  # (n_timepoints, 7)

    return sim_flux, sim_concs, mids_timecourse


def simulate_posterior_predictive_random_single(
    param_array,
    A_e,
    E_T,
    Q_e,
    P_e,
    initial_array,
    observation_times,
    emu_network,
    precomputed,
):
    """Simulate flux, concentrations, and MID timecourse for a single random model sample.

    Parameters
    ----------
    param_array : jnp.ndarray
        Shape (14,) - MCMC parameters for random model
    A_e, E_T, Q_e, P_e : float
        Experimental conditions
    initial_array : jnp.ndarray
        Initial concentrations [A, Q, P, EA, EPQ, EP, EQ]
    observation_times : jnp.ndarray
        Time points for MID measurements
    emu_network : dict
        EMU network structure
    precomputed : dict
        Precomputed matrix structure

    Returns
    -------
    tuple
        (flux, concs, mids_timecourse) where:
        - flux: scalar (net flux)
        - concs: (3,) array [A, Q, P]
        - mids_timecourse: (n_timepoints, 7) array
    """
    # Build parameter dict for random mechanism
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

    # Run enzyme ODE to steady state (random mechanism has 7 states)
    enzyme_sol = simulate_enzyme_to_steady_state_random_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Extract concentrations (random has EQ as well)
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    EQ = enzyme_y_ss[6]
    E = params["E_T"] - EA - EPQ - EP - EQ

    sim_concs = jnp.array([A, Q, P])
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "EQ": EQ, "E": E}

    # Net flux
    sim_flux = params["k_A_ex"] * (A_e - A)

    # EMU simulation for MIDs at each timepoint
    fluxes_array = compute_fluxes_array_pure_jax_random(concs_dict, params, emu_network)
    concs_array = compute_concs_array_pure_jax_random(concs_dict, emu_network)

    mids_list = []
    for t_idx in range(len(observation_times)):
        t = observation_times[t_idx]
        timepoints_to_t = jnp.array([0.0, t])
        emu_mdvs = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_to_t, emu_network, precomputed
        )
        sim_mids_t = extract_measurable_mids_from_emu_random(emu_mdvs)
        mids_list.append(sim_mids_t)

    mids_timecourse = jnp.stack(mids_list)  # (n_timepoints, 7)

    return sim_flux, sim_concs, mids_timecourse


def compute_posterior_predictive_ordered(
    samples,
    condition,
    observation_times,
    initial_array,
    max_samples=200,
):
    """Compute posterior predictive for ordered model.

    Parameters
    ----------
    samples : jnp.ndarray
        Shape (n_samples, n_params) - flattened posterior samples
    condition : dict
        Experimental condition with A_e, E_T, Q_e, P_e
    observation_times : jnp.ndarray
        Time points for MID measurements
    initial_array : jnp.ndarray
        Initial concentrations
    max_samples : int
        Maximum number of samples to use (for speed)

    Returns
    -------
    dict
        fluxes: (n_samples,)
        concs: (n_samples, 3)
        mids: (n_samples, n_timepoints, 7)
    """
    n_samples = min(samples.shape[0], max_samples)
    if samples.shape[0] > max_samples:
        # Subsample
        idx = np.linspace(0, samples.shape[0] - 1, max_samples, dtype=int)
        samples = samples[idx]

    A_e = condition["A_e"]
    E_T = condition["E_T"]
    Q_e = condition.get("Q_e", ORDERED_DEFAULT_PARAMS["Q_e"])
    P_e = condition.get("P_e", ORDERED_DEFAULT_PARAMS["P_e"])

    emu_network, precomputed = get_pure_jax_emu_network()

    # JIT compile the single-sample simulation
    @jax.jit
    def simulate_single(param_array):
        return simulate_posterior_predictive_ordered_single(
            param_array, A_e, E_T, Q_e, P_e,
            initial_array, observation_times, emu_network, precomputed
        )

    fluxes = []
    concs = []
    mids = []

    print(f"    Computing posterior predictive for {n_samples} samples...")
    sys.stdout.flush()

    for i in range(n_samples):
        if i % 50 == 0:
            print(f"      Sample {i}/{n_samples}")
            sys.stdout.flush()
        flux, conc, mid = simulate_single(samples[i])
        fluxes.append(float(flux))
        concs.append(np.array(conc))
        mids.append(np.array(mid))

    return {
        "fluxes": np.array(fluxes),
        "concs": np.array(concs),
        "mids": np.array(mids),
    }


def compute_posterior_predictive_random(
    samples,
    condition,
    observation_times,
    initial_array,
    max_samples=200,
):
    """Compute posterior predictive for random model.

    Parameters
    ----------
    samples : jnp.ndarray
        Shape (n_samples, n_params) - flattened posterior samples
    condition : dict
        Experimental condition with A_e, E_T, Q_e, P_e
    observation_times : jnp.ndarray
        Time points for MID measurements
    initial_array : jnp.ndarray
        Initial concentrations
    max_samples : int
        Maximum number of samples to use (for speed)

    Returns
    -------
    dict
        fluxes: (n_samples,)
        concs: (n_samples, 3)
        mids: (n_samples, n_timepoints, 7)
    """
    n_samples = min(samples.shape[0], max_samples)
    if samples.shape[0] > max_samples:
        # Subsample
        idx = np.linspace(0, samples.shape[0] - 1, max_samples, dtype=int)
        samples = samples[idx]

    A_e = condition["A_e"]
    E_T = condition["E_T"]
    Q_e = condition.get("Q_e", ORDERED_DEFAULT_PARAMS["Q_e"])
    P_e = condition.get("P_e", ORDERED_DEFAULT_PARAMS["P_e"])

    emu_network, precomputed = get_pure_jax_emu_network_random()

    # JIT compile the single-sample simulation
    @jax.jit
    def simulate_single(param_array):
        return simulate_posterior_predictive_random_single(
            param_array, A_e, E_T, Q_e, P_e,
            initial_array, observation_times, emu_network, precomputed
        )

    fluxes = []
    concs = []
    mids = []

    print(f"    Computing posterior predictive for {n_samples} samples...")
    sys.stdout.flush()

    for i in range(n_samples):
        if i % 50 == 0:
            print(f"      Sample {i}/{n_samples}")
            sys.stdout.flush()
        flux, conc, mid = simulate_single(samples[i])
        fluxes.append(float(flux))
        concs.append(np.array(conc))
        mids.append(np.array(mid))

    return {
        "fluxes": np.array(fluxes),
        "concs": np.array(concs),
        "mids": np.array(mids),
    }


# =============================================================================
# Posterior Predictive Plotting Functions
# =============================================================================


def plot_flux_conc_comparison(
    ordered_pred,
    random_pred,
    true_flux,
    true_concs,
    output_path,
):
    """Bar chart with 95% CI comparing ordered vs random flux and concentrations.

    Parameters
    ----------
    ordered_pred : dict
        Posterior predictive from ordered model with keys 'fluxes', 'concs'
    random_pred : dict
        Posterior predictive from random model
    true_flux : float
        True net flux
    true_concs : array-like
        True concentrations [A, Q, P]
    output_path : Path
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = ["Net Flux", "[A]", "[Q]", "[P]"]
    x = np.arange(len(labels))
    width = 0.35

    # Compute statistics
    ordered_means = [
        np.mean(ordered_pred["fluxes"]),
        np.mean(ordered_pred["concs"][:, 0]),
        np.mean(ordered_pred["concs"][:, 1]),
        np.mean(ordered_pred["concs"][:, 2]),
    ]
    ordered_lower = [
        np.percentile(ordered_pred["fluxes"], 2.5),
        np.percentile(ordered_pred["concs"][:, 0], 2.5),
        np.percentile(ordered_pred["concs"][:, 1], 2.5),
        np.percentile(ordered_pred["concs"][:, 2], 2.5),
    ]
    ordered_upper = [
        np.percentile(ordered_pred["fluxes"], 97.5),
        np.percentile(ordered_pred["concs"][:, 0], 97.5),
        np.percentile(ordered_pred["concs"][:, 1], 97.5),
        np.percentile(ordered_pred["concs"][:, 2], 97.5),
    ]
    ordered_err = [
        [m - l for m, l in zip(ordered_means, ordered_lower)],
        [u - m for m, u in zip(ordered_means, ordered_upper)],
    ]

    random_means = [
        np.mean(random_pred["fluxes"]),
        np.mean(random_pred["concs"][:, 0]),
        np.mean(random_pred["concs"][:, 1]),
        np.mean(random_pred["concs"][:, 2]),
    ]
    random_lower = [
        np.percentile(random_pred["fluxes"], 2.5),
        np.percentile(random_pred["concs"][:, 0], 2.5),
        np.percentile(random_pred["concs"][:, 1], 2.5),
        np.percentile(random_pred["concs"][:, 2], 2.5),
    ]
    random_upper = [
        np.percentile(random_pred["fluxes"], 97.5),
        np.percentile(random_pred["concs"][:, 0], 97.5),
        np.percentile(random_pred["concs"][:, 1], 97.5),
        np.percentile(random_pred["concs"][:, 2], 97.5),
    ]
    random_err = [
        [m - l for m, l in zip(random_means, random_lower)],
        [u - m for m, u in zip(random_means, random_upper)],
    ]

    true_values = [true_flux, true_concs[0], true_concs[1], true_concs[2]]

    # Plot bars
    bars_ordered = ax.bar(
        x - width/2, ordered_means, width, yerr=ordered_err,
        label="Ordered", color="tab:blue", capsize=5, alpha=0.8
    )
    bars_random = ax.bar(
        x + width/2, random_means, width, yerr=random_err,
        label="Random", color="tab:orange", capsize=5, alpha=0.8
    )

    # Plot true values
    ax.scatter(x, true_values, color="black", marker="D", s=100, zorder=5, label="True")

    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Posterior Predictive: Flux and Concentrations (Condition 1)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved flux/concentration comparison to {output_path}")
    sys.stdout.flush()


def plot_mid_timecourse_comparison(
    ordered_pred,
    random_pred,
    true_mids,
    observation_times,
    output_path,
):
    """Plot MID timecourse with 95% CI comparing ordered vs random.

    Parameters
    ----------
    ordered_pred : dict
        Posterior predictive with 'mids' key, shape (n_samples, n_timepoints, 7)
    random_pred : dict
        Posterior predictive with 'mids' key
    true_mids : dict
        True MIDs at each timepoint, {time: array(7)}
    observation_times : array
        Time points
    output_path : Path
        Output file path
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # MID indices: A (0,1,2), Q (3,4), P (5,6)
    metabolites = [
        ("A", [0, 1, 2], ["M+0", "M+1", "M+2"]),
        ("Q", [3, 4], ["M+0", "M+1"]),
        ("P", [5, 6], ["M+0", "M+1"]),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    times = np.array(observation_times)

    # Get true MIDs as array
    true_mids_array = np.array([true_mids[float(t)] for t in observation_times])  # (n_timepoints, 7)

    for ax_idx, (metab_name, mid_indices, mid_labels) in enumerate(metabolites):
        ax = axes[ax_idx]

        for i, (mid_idx, mid_label) in enumerate(zip(mid_indices, mid_labels)):
            color = colors[i]

            # Ordered model - solid lines
            ordered_mid = ordered_pred["mids"][:, :, mid_idx]  # (n_samples, n_timepoints)
            ordered_mean = np.mean(ordered_mid, axis=0)
            ordered_lower = np.percentile(ordered_mid, 2.5, axis=0)
            ordered_upper = np.percentile(ordered_mid, 97.5, axis=0)

            ax.plot(times, ordered_mean, "-", color=color, linewidth=2,
                    label=f"{mid_label} (Ordered)" if ax_idx == 0 else None)
            ax.fill_between(times, ordered_lower, ordered_upper, color=color, alpha=0.2)

            # Random model - dashed lines
            random_mid = random_pred["mids"][:, :, mid_idx]
            random_mean = np.mean(random_mid, axis=0)
            random_lower = np.percentile(random_mid, 2.5, axis=0)
            random_upper = np.percentile(random_mid, 97.5, axis=0)

            ax.plot(times, random_mean, "--", color=color, linewidth=2,
                    label=f"{mid_label} (Random)" if ax_idx == 0 else None)
            ax.fill_between(times, random_lower, random_upper, color=color, alpha=0.1,
                            hatch="///", edgecolor=color, linewidth=0)

            # True values as markers
            true_vals = true_mids_array[:, mid_idx]
            ax.scatter(times, true_vals, color=color, marker="o", s=60, zorder=5,
                       edgecolors="black", linewidths=0.5)

        ax.set_xlabel("Time")
        ax.set_ylabel("MID Fraction")
        ax.set_title(f"Metabolite {metab_name}")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)

    # Add legend to first subplot
    axes[0].legend(loc="upper left", fontsize=8)

    # Add custom legend explaining line styles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="-", label="Ordered (solid)"),
        Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="Random (dashed)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", label="True values"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved MID timecourse comparison to {output_path}")
    sys.stdout.flush()


def create_arviz_idata(samples, log_likelihood, param_names, model_name):
    """Create ArviZ InferenceData with posterior and log_likelihood groups.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_chains, n_samples, n_params)
    log_likelihood : np.ndarray
        Shape (n_chains, n_samples, n_obs)
    param_names : list
        Parameter names
    model_name : str
        Model name for labeling

    Returns
    -------
    az.InferenceData
    """
    n_chains, n_samples, n_params = samples.shape
    n_obs = log_likelihood.shape[2]

    # Create posterior dataset
    posterior_dict = {
        name: (["chain", "draw"], samples[:, :, i])
        for i, name in enumerate(param_names)
    }
    posterior = xr.Dataset(posterior_dict)

    # Create log_likelihood dataset
    # ArviZ expects log_likelihood with observation dimensions
    log_likelihood_ds = xr.Dataset({
        "obs": (["chain", "draw", "obs_id"], log_likelihood)
    })

    # Create InferenceData
    idata = az.InferenceData(
        posterior=posterior,
        log_likelihood=log_likelihood_ds,
    )

    return idata


def main():
    print("=" * 80)
    print("PSIS-LOO Model Comparison: Ordered vs Random Enzyme Mechanisms")
    print("=" * 80)
    sys.stdout.flush()

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()

    # Load MCMC results
    print("\nLoading MCMC results...")
    sys.stdout.flush()

    ordered_results = np.load(
        PROJECT_ROOT / "results" / "enzyme_simulation_instationary" / "mcmc_results.npz",
        allow_pickle=True,
    )
    random_results = np.load(
        PROJECT_ROOT / "results" / "enzyme_simulation_random_instationary" / "mcmc_results.npz",
        allow_pickle=True,
    )

    ordered_samples = jnp.array(ordered_results["samples"])
    random_samples = jnp.array(random_results["samples"])

    print(f"  Ordered model: {ordered_samples.shape} (chains, samples, params)")
    print(f"  Random model: {random_samples.shape} (chains, samples, params)")
    sys.stdout.flush()

    # Generate reference data (using ordered model)
    print("\nGenerating reference data...")
    sys.stdout.flush()

    ordered_params = compute_wegscheider_constraints(ORDERED_DEFAULT_PARAMS.copy())
    ref_data_list = generate_reference_data_multi_condition(
        ordered_params,
        ORDERED_DEFAULT_INITIAL,
        EXPERIMENTAL_CONDITIONS,
        OBSERVATION_TIMES,
    )

    print(f"  Generated data for {len(ref_data_list)} conditions")
    print(f"  Observations per condition: {N_OBS_PER_CONDITION}")
    print(f"  Total observations: {N_CONDITIONS * N_OBS_PER_CONDITION}")
    sys.stdout.flush()

    # Initial conditions
    ordered_initial_array = jnp.array([
        ORDERED_DEFAULT_INITIAL["A"],
        ORDERED_DEFAULT_INITIAL["Q"],
        ORDERED_DEFAULT_INITIAL["P"],
        ORDERED_DEFAULT_INITIAL["EA"],
        ORDERED_DEFAULT_INITIAL["EPQ"],
        ORDERED_DEFAULT_INITIAL["EP"],
    ])

    random_initial_array = jnp.array([
        RANDOM_DEFAULT_INITIAL["A"],
        RANDOM_DEFAULT_INITIAL["Q"],
        RANDOM_DEFAULT_INITIAL["P"],
        RANDOM_DEFAULT_INITIAL["EA"],
        RANDOM_DEFAULT_INITIAL["EPQ"],
        RANDOM_DEFAULT_INITIAL["EP"],
        RANDOM_DEFAULT_INITIAL["EQ"],
    ])

    # Compute pointwise log-likelihoods
    print("\n" + "-" * 40)
    print("Computing pointwise log-likelihoods")
    print("-" * 40)
    sys.stdout.flush()

    ll_ordered = compute_all_pointwise_ll_ordered(
        ordered_samples,
        ref_data_list,
        EXPERIMENTAL_CONDITIONS,
        OBSERVATION_TIMES,
        ordered_initial_array,
    )
    print(f"  Ordered log-likelihoods shape: {ll_ordered.shape}")
    sys.stdout.flush()

    ll_random = compute_all_pointwise_ll_random(
        random_samples,
        ref_data_list,
        EXPERIMENTAL_CONDITIONS,
        OBSERVATION_TIMES,
        random_initial_array,
    )
    print(f"  Random log-likelihoods shape: {ll_random.shape}")
    sys.stdout.flush()

    # Sanity check: compare sum of pointwise ll to stored log_probs
    print("\nSanity check: sum of pointwise ll vs stored log_probs")
    stored_ll_ordered = ordered_results["log_probs"]
    computed_ll_ordered = np.sum(ll_ordered, axis=2)
    # Note: stored log_probs includes prior, so we expect some difference
    print(f"  Ordered - stored log_probs range: [{stored_ll_ordered.min():.1f}, {stored_ll_ordered.max():.1f}]")
    print(f"  Ordered - computed ll sum range: [{computed_ll_ordered.min():.1f}, {computed_ll_ordered.max():.1f}]")

    stored_ll_random = random_results["log_probs"]
    computed_ll_random = np.sum(ll_random, axis=2)
    print(f"  Random - stored log_probs range: [{stored_ll_random.min():.1f}, {stored_ll_random.max():.1f}]")
    print(f"  Random - computed ll sum range: [{computed_ll_random.min():.1f}, {computed_ll_random.max():.1f}]")
    sys.stdout.flush()

    # Create ArviZ InferenceData objects
    print("\nCreating ArviZ InferenceData...")
    sys.stdout.flush()

    idata_ordered = create_arviz_idata(
        np.array(ordered_samples),
        np.array(ll_ordered),
        list(MCMC_PARAM_NAMES),
        "ordered",
    )

    idata_random = create_arviz_idata(
        np.array(random_samples),
        np.array(ll_random),
        list(MCMC_PARAM_NAMES_RANDOM),
        "random",
    )

    # Save InferenceData (as JSON since h5py may not be available)
    print("\nSaving InferenceData objects...")
    sys.stdout.flush()

    try:
        idata_ordered.to_netcdf(RESULTS_DIR / "ordered_inference_data.nc")
        idata_random.to_netcdf(RESULTS_DIR / "random_inference_data.nc")
        print(f"  Saved to {RESULTS_DIR}")
    except ImportError:
        print("  Note: h5py not available, skipping NetCDF save")
        # Save log-likelihoods as numpy arrays instead
        np.savez(
            RESULTS_DIR / "ordered_log_likelihood.npz",
            log_likelihood=np.array(ll_ordered),
            samples=np.array(ordered_samples),
            param_names=list(MCMC_PARAM_NAMES),
        )
        np.savez(
            RESULTS_DIR / "random_log_likelihood.npz",
            log_likelihood=np.array(ll_random),
            samples=np.array(random_samples),
            param_names=list(MCMC_PARAM_NAMES_RANDOM),
        )
        print(f"  Saved log-likelihoods to .npz files in {RESULTS_DIR}")
    sys.stdout.flush()

    # Compute LOO
    print("\n" + "-" * 40)
    print("Computing PSIS-LOO")
    print("-" * 40)
    sys.stdout.flush()

    print("\nOrdered model LOO...")
    sys.stdout.flush()
    loo_ordered = az.loo(idata_ordered, pointwise=True)
    print(loo_ordered)
    sys.stdout.flush()

    print("\nRandom model LOO...")
    sys.stdout.flush()
    loo_random = az.loo(idata_random, pointwise=True)
    print(loo_random)
    sys.stdout.flush()

    # Model comparison
    print("\n" + "-" * 40)
    print("Model Comparison")
    print("-" * 40)
    sys.stdout.flush()

    comparison = az.compare({
        "ordered": idata_ordered,
        "random": idata_random,
    })
    print(comparison)
    sys.stdout.flush()

    # Save comparison results
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"\nSaved comparison to {RESULTS_DIR / 'model_comparison.csv'}")
    sys.stdout.flush()

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    sys.stdout.flush()

    # Pareto k diagnostic plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    pareto_k_ordered = loo_ordered.pareto_k.values
    # Filter out inf values for plotting
    pareto_k_ordered_finite = pareto_k_ordered[np.isfinite(pareto_k_ordered)]
    if len(pareto_k_ordered_finite) > 0:
        ax.hist(pareto_k_ordered_finite, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0.7, color="red", linestyle="--", label="k=0.7 threshold")
        ax.axvline(1.0, color="darkred", linestyle="--", label="k=1.0 threshold")
    ax.set_xlabel("Pareto k")
    ax.set_ylabel("Count")
    n_bad_ordered = np.sum(pareto_k_ordered > 0.7) if np.any(np.isfinite(pareto_k_ordered)) else len(pareto_k_ordered)
    ax.set_title(f"Ordered Model\n(k > 0.7: {n_bad_ordered}/{len(pareto_k_ordered)})")
    ax.legend()

    ax = axes[1]
    pareto_k_random = loo_random.pareto_k.values
    # Filter out inf values for plotting
    pareto_k_random_finite = pareto_k_random[np.isfinite(pareto_k_random)]
    if len(pareto_k_random_finite) > 0:
        ax.hist(pareto_k_random_finite, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0.7, color="red", linestyle="--", label="k=0.7 threshold")
        ax.axvline(1.0, color="darkred", linestyle="--", label="k=1.0 threshold")
    else:
        # All Pareto k are inf - add text annotation
        ax.text(0.5, 0.5, f"All {len(pareto_k_random)} Pareto k = inf\n(PSIS failed)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel("Pareto k")
    ax.set_ylabel("Count")
    n_bad_random = np.sum(pareto_k_random > 0.7) if np.any(np.isfinite(pareto_k_random)) else len(pareto_k_random)
    ax.set_title(f"Random Model\n(k > 0.7: {n_bad_random}/{len(pareto_k_random)})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pareto_k_diagnostic.png", dpi=150)
    plt.close()
    print(f"  Saved Pareto k diagnostic to {RESULTS_DIR / 'pareto_k_diagnostic.png'}")
    sys.stdout.flush()

    # Model comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    az.plot_compare(comparison, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved model comparison plot to {RESULTS_DIR / 'model_comparison.png'}")
    sys.stdout.flush()

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nOrdered model:")
    print(f"  ELPD: {loo_ordered.elpd_loo:.1f} ± {loo_ordered.se:.1f}")
    print(f"  p_loo (effective params): {loo_ordered.p_loo:.1f}")
    print(f"  Pareto k > 0.7: {np.sum(pareto_k_ordered > 0.7)}/{len(pareto_k_ordered)}")

    print(f"\nRandom model:")
    print(f"  ELPD: {loo_random.elpd_loo:.1f} ± {loo_random.se:.1f}")
    print(f"  p_loo (effective params): {loo_random.p_loo:.1f}")
    print(f"  Pareto k > 0.7: {np.sum(pareto_k_random > 0.7)}/{len(pareto_k_random)}")

    print(f"\nModel comparison (higher ELPD is better):")
    print(comparison.to_string())

    # =========================================================================
    # Posterior Predictive Plots for Condition 1
    # =========================================================================
    print("\n" + "-" * 40)
    print("Computing Posterior Predictive (Condition 1)")
    print("-" * 40)
    sys.stdout.flush()

    # Flatten samples across chains: (2, 400, params) -> (800, params)
    ordered_flat = np.array(ordered_samples).reshape(-1, ordered_samples.shape[-1])
    random_flat = np.array(random_samples).reshape(-1, random_samples.shape[-1])

    print(f"  Ordered samples: {ordered_flat.shape}")
    print(f"  Random samples: {random_flat.shape}")
    sys.stdout.flush()

    # Get condition 1
    condition_1 = EXPERIMENTAL_CONDITIONS[0]
    Q_e_1 = condition_1.get("Q_e", ORDERED_DEFAULT_PARAMS["Q_e"])
    P_e_1 = condition_1.get("P_e", ORDERED_DEFAULT_PARAMS["P_e"])
    print(f"  Condition 1: A_e={condition_1['A_e']:.2f}, E_T={condition_1['E_T']:.2f}, Q_e={Q_e_1:.2f}, P_e={P_e_1:.2f}")
    sys.stdout.flush()

    # Compute posterior predictive for ordered model
    print("\n  Ordered model:")
    ordered_pred = compute_posterior_predictive_ordered(
        jnp.array(ordered_flat),
        condition_1,
        OBSERVATION_TIMES,
        ordered_initial_array,
        max_samples=200,
    )

    # Compute posterior predictive for random model
    print("\n  Random model:")
    random_pred = compute_posterior_predictive_random(
        jnp.array(random_flat),
        condition_1,
        OBSERVATION_TIMES,
        random_initial_array,
        max_samples=200,
    )

    # Get true values from reference data (condition 1)
    true_flux = ref_data_list[0]["fluxes"]["v_net"]
    true_concs = np.array([
        ref_data_list[0]["concentrations"]["A"],
        ref_data_list[0]["concentrations"]["Q"],
        ref_data_list[0]["concentrations"]["P"],
    ])
    true_mids = ref_data_list[0]["timecourse_mids"]

    print("\n  True values for condition 1:")
    print(f"    Net flux: {true_flux:.4f}")
    print(f"    Concentrations [A, Q, P]: [{true_concs[0]:.4f}, {true_concs[1]:.4f}, {true_concs[2]:.4f}]")
    sys.stdout.flush()

    # Generate posterior predictive plots
    print("\nGenerating posterior predictive plots...")
    sys.stdout.flush()

    plot_flux_conc_comparison(
        ordered_pred,
        random_pred,
        true_flux,
        true_concs,
        RESULTS_DIR / "posterior_flux_conc.png",
    )

    plot_mid_timecourse_comparison(
        ordered_pred,
        random_pred,
        true_mids,
        OBSERVATION_TIMES,
        RESULTS_DIR / "posterior_mid_timecourse.png",
    )

    print("\n" + "=" * 80)
    print("PSIS-LOO Comparison Complete!")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

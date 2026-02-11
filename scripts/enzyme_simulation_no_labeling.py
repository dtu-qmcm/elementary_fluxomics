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
from jax.scipy.stats import norm
import diffrax
import blackjax
from jax_tqdm import scan_tqdm

try:
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
        simulate_ode_to_steady_state,
        simulate_enzyme_to_steady_state_jax,
        get_steady_state,
        compute_forward_backward_fluxes,
        params_array_to_dict,
        params_dict_to_array,
    )
except ModuleNotFoundError:
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
        simulate_ode_to_steady_state,
        simulate_enzyme_to_steady_state_jax,
        get_steady_state,
        compute_forward_backward_fluxes,
        params_array_to_dict,
        params_dict_to_array,
    )


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "enzyme_simulation_no_labeling"



# Constants, default params, and shared functions imported from enzyme_utils


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def compute_log_likelihood(
    obs_fluxes,
    sim_fluxes,
    obs_concentrations,
    sim_concentrations,
    flux_std_rel=0.01,
    conc_std=0.01,
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
    flux_std_rel : float
        Relative standard deviation for flux (0.1% of |obs flux|)
    conc_std : float
        Standard deviation for concentration lognormal distribution (default 0.001)

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

    # 1. Fluxes: Normal distribution (std = 0.1% of |obs flux|)
    for flux_id in obs_fluxes:
        if flux_id in sim_fluxes:
            obs = obs_fluxes[flux_id]
            sim = sim_fluxes[flux_id]
            flux_std = flux_std_rel * jnp.abs(obs)
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
    sol = simulate_enzyme_to_steady_state_jax(params, initial_array)

    # Check if solver failed
    solver_failed = jnp.any(jnp.isnan(sol.ys))

    # Extract steady-state (SaveAt(t1=True) gives shape (1, 6) or (6,))
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys

    # Extract measurable concentrations only (A, Q, P)
    A, Q, P = y_ss[0], y_ss[1], y_ss[2]
    sim_concs = jnp.array([A, Q, P])

    # Compute net flux (exchange flux at steady state)
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1% of |ref_flux|)
    flux_std = 0.01 * jnp.abs(ref_flux)
    ll_flux = norm.logpdf(ref_flux, loc=sim_flux, scale=flux_std)

    # Log-likelihood for concentrations (lognormal distribution, std=0.01)
    conc_std = 0.01
    ll_concs = jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_concs, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_concs, 1e-10)),
        scale=conc_std
    ))

    ll_condition = ll_flux + ll_concs

    # Return -inf if solver failed
    return jnp.where(solver_failed, -jnp.inf, ll_condition)


def compute_log_posterior_jax(
    log_param_array,
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
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (num_conditions, 4) [[A_e, E_T, Q_e, P_e], ...]
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

    # Use vmap to compute likelihood for all conditions in parallel
    condition_lls = jax.vmap(
        lambda A_e, E_T, Q_e, P_e, rf, rc: compute_condition_likelihood(
            A_e, E_T, rf, rc, param_array, Q_e, P_e, initial_array
        )
    )(
        fixed_params_varying[:, 0],  # A_e
        fixed_params_varying[:, 1],  # E_T
        fixed_params_varying[:, 2],  # Q_e
        fixed_params_varying[:, 3],  # P_e
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
    n_warmup=1000,
    n_samples=1000,
    n_chains=2,
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
        Number of samples to collect per chain
    n_chains : int
        Number of parallel chains to run
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
    print(f"Running MCMC with blackjax NUTS ({num_conditions} conditions, {n_chains} chains, no isotope labeling)")
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

    # Generic prior: all true parameter values within 2 standard deviations
    # Range in log-space: -2.3 (k_ex=0.1) to 5.5 (k2r=250)
    # Center at log(5)≈1.6, std=2.0 covers range within 2 std devs
    param_array_ref = params_dict_to_array(ref_params)
    n_params = len(param_array_ref)
    prior_mean = jnp.ones(n_params) * 1.6  # log(5) ≈ 1.6
    prior_std = 4.0

    # Log-posterior function (JAX-compatible, takes log-transformed params)
    def logdensity_fn(log_param_array):
        return compute_log_posterior_jax(
            log_param_array,
            fixed_params_varying,
            initial_array,
            ref_flux_array,
            ref_concs_arrays,
            prior_mean,
            prior_std,
        )

    # Initialize at true values (in log-space)
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
        "samples_flat": samples_flat,  # Flattened
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

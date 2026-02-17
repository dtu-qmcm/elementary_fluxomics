#!/usr/bin/env python3
"""Identifiable-only MCMC: steady-state EMU labeling analysis.

This script is a variant of scripts/mcmc/enzyme_simulation.py that fixes the 4
unidentifiable parameters (k3f, k3r, k4f, k4r) to their literature values and
only samples the 7 identifiable parameters.

Unidentifiable parameters (fixed to DEFAULT_PARAMS values):
  k3f=125.0, k3r=12.5, k4f=200.0, k4r=20.0

Identifiable parameters (sampled):
  k2r, k5f, k5r, k_A_ex, k_Q_ex, k_P_ex, k_pq_f

Based on FIM eigenvalue analysis of the instationary mode.
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
    from scripts.emu_jax import (
        build_emu_network,
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
        compute_wegscheider_constraints,
        compute_wegscheider_constraints_jax,
        compute_E,
        compute_rates,
        ode_system,
        simulate_enzyme_to_steady_state_jax,
        simulate_ode_to_steady_state,
        get_steady_state,
        compute_forward_backward_fluxes,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from emu_jax import (
        build_emu_network,
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
        compute_wegscheider_constraints,
        compute_wegscheider_constraints_jax,
        compute_E,
        compute_rates,
        ode_system,
        simulate_enzyme_to_steady_state_jax,
        simulate_ode_to_steady_state,
        get_steady_state,
        compute_forward_backward_fluxes,
    )


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REACTIONS_FILE = PROJECT_ROOT / "models" / "simple_network" / "reactions.tsv"
RESULTS_DIR = PROJECT_ROOT / "results" / "mcmc_identifiable_steady_state"


# ============================================================================
# Identifiable parameter subset
# ============================================================================

MCMC_PARAM_NAMES_IDENTIFIABLE = [
    "k2r", "k5f", "k5r", "k_A_ex", "k_Q_ex", "k_P_ex", "k_pq_f"
]

FIXED_PARAMS_UNIDENTIFIABLE = {
    "k3f": DEFAULT_PARAMS["k3f"],   # 125.0
    "k3r": DEFAULT_PARAMS["k3r"],   # 12.5
    "k4f": DEFAULT_PARAMS["k4f"],   # 200.0
    "k4r": DEFAULT_PARAMS["k4r"],   # 20.0
}


def params_dict_to_array_identifiable(params):
    """Extract identifiable MCMC parameters from dictionary to array."""
    return jnp.array([params[name] for name in MCMC_PARAM_NAMES_IDENTIFIABLE])


# Global cache for EMU network
_PURE_JAX_EMU_NETWORK = None


def get_pure_jax_emu_network():
    """Get or initialize the pure JAX EMU network structure."""
    global _PURE_JAX_EMU_NETWORK

    if _PURE_JAX_EMU_NETWORK is None:
        network = build_emu_network(
            reactions_file=REACTIONS_FILE,
            target_emus={"A": ["12"], "EA": ["12"], "EPQ": ["12"],
                         "EP": ["1"], "Q": ["1"], "P": ["1"]},
            labeling_substrate="A_ext",
            labeling_pattern=["10"],
            labeling_percentage=[1.0],
            labeling_purity=[1.0],
        )
        precomputed = precompute_matrix_structure(network)
        _PURE_JAX_EMU_NETWORK = (network, precomputed)

    return _PURE_JAX_EMU_NETWORK


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def compute_fluxes_array_pure_jax(concs, params, network):
    """Compute flux array for pure JAX EMU simulation."""
    A, Q, P = concs["A"], concs["Q"], concs["P"]
    EA, EPQ, EP, E = concs["EA"], concs["EPQ"], concs["EP"], concs["E"]

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


def compute_mids_pure_jax_emu_ss(concs_dict, params, network, precomputed=None):
    """Compute MIDs using pure JAX steady-state EMU simulation."""
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)
    return simulate_ss_mdvs_pure_jax(fluxes_array, network, precomputed)


def generate_reference_data_single_condition(params, initial):
    """Generate reference observations from a parameter set for a single condition."""
    params = compute_wegscheider_constraints(params)

    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"]
    ])

    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    A, Q, P = float(enzyme_y_ss[0]), float(enzyme_y_ss[1]), float(enzyme_y_ss[2])
    EA, EPQ, EP = float(enzyme_y_ss[3]), float(enzyme_y_ss[4]), float(enzyme_y_ss[5])
    E = params["E_T"] - EA - EPQ - EP

    concs = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)

    net_rates = {
        "v_A_ex": v_A_ex,
        "v2": A * E * params["k2f"] - EA * params["k2r"],
        "v3": EA * params["k3f"] - EPQ * params["k3r"],
        "v4": EPQ * params["k4f"] - Q * EP * params["k4r"],
        "v5": EP * params["k5f"] - E * P * params["k5r"],
        "v_Q_ex": params["k_Q_ex"] * (params["Q_e"] - Q),
        "v_P_ex": params["k_P_ex"] * (params["P_e"] - P),
        "v_pq": P * params["k_pq_f"] - Q * params["k_pq_r"],
        "v_net": v_A_ex,
    }

    network, precomputed = get_pure_jax_emu_network()
    ss_mids_jax = compute_mids_pure_jax_emu_ss(concs, params, network, precomputed)

    ss_mids_np = {
        "A_12": np.array(ss_mids_jax["A_12"]),
        "Q_1": np.array(ss_mids_jax["Q_1"]),
        "P_1": np.array(ss_mids_jax["P_1"]),
    }

    return {
        "fluxes": net_rates,
        "concentrations": concs,
        "ss_mids": ss_mids_np,
        "enzyme_steady_state": np.array(enzyme_y_ss),
    }


def generate_reference_data_multi_condition(base_params, initial, conditions):
    """Generate reference data for multiple experimental conditions."""
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


def compute_condition_likelihood_emu(
    A_e, E_T, ref_flux, ref_concs, ref_mids,
    param_array, Q_e, P_e, initial_array,
    emu_network, precomputed,
):
    """Compute log-likelihood using pure JAX steady-state EMU simulation.

    Parameters
    ----------
    param_array : jnp.array
        Rate constants (7,): identifiable parameters only
    """
    # Build parameter dict: 7 sampled + 4 fixed
    params = {
        "k2r": param_array[0],
        "k3f": FIXED_PARAMS_UNIDENTIFIABLE["k3f"],
        "k3r": FIXED_PARAMS_UNIDENTIFIABLE["k3r"],
        "k4f": FIXED_PARAMS_UNIDENTIFIABLE["k4f"],
        "k4r": FIXED_PARAMS_UNIDENTIFIABLE["k4r"],
        "k5f": param_array[1],
        "k5r": param_array[2],
        "k_A_ex": param_array[3],
        "k_Q_ex": param_array[4],
        "k_P_ex": param_array[5],
        "k_pq_f": param_array[6],
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

    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    A, Q, P = enzyme_y_ss[0], enzyme_y_ss[1], enzyme_y_ss[2]
    EA, EPQ, EP = enzyme_y_ss[3], enzyme_y_ss[4], enzyme_y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    sim_concs = jnp.array([A, Q, P])
    sim_flux = params["k_A_ex"] * (A_e - A)

    flux_std = 0.01 * jnp.abs(ref_flux)
    ll_flux = norm.logpdf(ref_flux, loc=sim_flux, scale=flux_std)

    conc_std = 0.01
    ll_concs = jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_concs, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_concs, 1e-10)),
        scale=conc_std
    ))

    # Step 2: Pure JAX steady-state EMU simulation
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}
    sim_mids_dict = compute_mids_pure_jax_emu_ss(concs_dict, params, emu_network, precomputed)

    sim_mids_array = jnp.concatenate([
        sim_mids_dict["A_12"],
        sim_mids_dict["Q_1"],
        sim_mids_dict["P_1"],
    ])

    mid_std = 0.01
    mid_multiplier = 5
    ll_mids = mid_multiplier * jnp.sum(norm.logpdf(
        jnp.log(jnp.maximum(ref_mids, 1e-10)),
        loc=jnp.log(jnp.maximum(sim_mids_array, 1e-10)),
        scale=mid_std
    ))

    ll_condition = ll_flux + ll_concs + ll_mids

    return jnp.where(enzyme_failed, -jnp.inf, ll_condition)


def compute_log_posterior_jax_emu(
    log_param_array, fixed_params_varying, initial_array,
    ref_flux_array, ref_concs_arrays, ref_mids_arrays,
    prior_mean, prior_std, emu_network, precomputed,
):
    """JAX-compatible log-posterior using pure JAX steady-state EMU simulation.

    Parameters
    ----------
    log_param_array : jnp.array
        Log-transformed MCMC parameters (7,): identifiable only
    """
    param_array = jnp.exp(log_param_array)

    log_prior = jnp.sum(
        -0.5 * ((log_param_array - prior_mean) / prior_std) ** 2
        - jnp.log(prior_std * jnp.sqrt(2 * jnp.pi))
    )

    condition_lls = jax.vmap(
        lambda A_e, E_T, Q_e, P_e, rf, rc, rm: compute_condition_likelihood_emu(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e, initial_array,
            emu_network, precomputed
        )
    )(
        fixed_params_varying[:, 0],
        fixed_params_varying[:, 1],
        fixed_params_varying[:, 2],
        fixed_params_varying[:, 3],
        ref_flux_array,
        ref_concs_arrays,
        ref_mids_arrays,
    )

    total_log_likelihood = jnp.sum(condition_lls)
    return log_prior + total_log_likelihood


def run_mcmc(
    ref_params, initial, conditions=None,
    n_warmup=200, n_samples=200, n_chains=2, seed=42,
):
    """Run MCMC sampling (identifiable params only, steady-state labeling)."""
    if conditions is None:
        conditions = EXPERIMENTAL_CONDITIONS

    num_conditions = len(conditions)

    print("=" * 80)
    print(f"Running IDENTIFIABLE-ONLY MCMC - Steady State ({num_conditions} conditions, {n_chains} chains)")
    print(f"Sampled parameters (7): {MCMC_PARAM_NAMES_IDENTIFIABLE}")
    print(f"Fixed parameters (4): {FIXED_PARAMS_UNIDENTIFIABLE}")
    print("=" * 80)
    print(f"\nExperimental conditions:")
    for i, cond in enumerate(conditions):
        print(f"  Condition {i+1}: E_T={cond['E_T']:.4f}, A_e={cond['A_e']:.4f}, Q_e={cond['Q_e']:.4f}, P_e={cond['P_e']:.4f}")
    sys.stdout.flush()

    fixed_params_varying = jnp.array([
        [cond["A_e"], cond["E_T"], cond["Q_e"], cond["P_e"]] for cond in conditions
    ])

    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"],
    ])

    print("\nGenerating reference data for all conditions...")
    sys.stdout.flush()
    ref_data_list = generate_reference_data_multi_condition(ref_params, initial, conditions)

    ref_flux_array = jnp.array([rd["fluxes"]["v_net"] for rd in ref_data_list])

    ref_concs_arrays = jnp.array([
        [rd["concentrations"]["A"], rd["concentrations"]["Q"], rd["concentrations"]["P"]]
        for rd in ref_data_list
    ])

    ref_mids_arrays = jnp.array([
        np.concatenate([
            np.array(rd["ss_mids"]["A_12"]),
            np.array(rd["ss_mids"]["Q_1"]),
            np.array(rd["ss_mids"]["P_1"]),
        ])
        for rd in ref_data_list
    ])

    # Load EMU network
    emu_network, precomputed = get_pure_jax_emu_network()

    param_array_ref = params_dict_to_array_identifiable(ref_params)
    n_params = len(param_array_ref)
    prior_mean = jnp.ones(n_params) * 1.6
    prior_std = 4.0

    @jax.jit
    def logdensity_fn(log_param_array):
        return compute_log_posterior_jax_emu(
            log_param_array, fixed_params_varying, initial_array,
            ref_flux_array, ref_concs_arrays, ref_mids_arrays,
            prior_mean, prior_std, emu_network, precomputed,
        )

    rng_key = jax.random.PRNGKey(seed)
    initial_position = jnp.log(param_array_ref)

    print(f"\nNumber of MCMC parameters: {n_params} (identifiable only)")
    print(f"Initial position (log-space): {initial_position}")
    print(f"Initial position (natural): {param_array_ref}")
    print(f"Initial log-posterior: {logdensity_fn(initial_position)}")
    sys.stdout.flush()

    print(f"\nRunning {n_warmup} warmup iterations...")
    sys.stdout.flush()

    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, num_steps=n_warmup,
        target_acceptance_rate=0.80, progress_bar=True,
    )

    rng_key, warmup_key = jax.random.split(rng_key)
    state, kernel, *_ = warmup.run(warmup_key, initial_position)

    print(f"Warmup complete.")
    sys.stdout.flush()

    print(f"\nCollecting {n_samples} samples from {n_chains} chains (parallel via vmap)...")
    sys.stdout.flush()

    def run_single_chain(chain_key, initial_state):
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

    chain_keys = jax.random.split(rng_key, n_chains)
    run_chains_parallel = jax.vmap(run_single_chain, in_axes=(0, None))
    log_samples, log_probs = run_chains_parallel(chain_keys, state)

    print(f"  All {n_chains} chains complete (parallel execution)")
    sys.stdout.flush()

    samples = jnp.exp(log_samples)

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
    samples_flat = samples.reshape(-1, samples.shape[-1])

    print("\n" + "=" * 80)
    print("MCMC Results Summary (Identifiable Parameters Only - Steady State)")
    print("=" * 80)

    results = {
        "samples": samples,
        "samples_flat": samples_flat,
        "log_samples": log_samples,
        "log_probs": log_probs,
        "param_names": MCMC_PARAM_NAMES_IDENTIFIABLE,
        "reference": param_array_ref,
        "n_chains": n_chains,
        "n_samples": n_samples,
        "rhats": np.array(rhats),
        "fixed_params": FIXED_PARAMS_UNIDENTIFIABLE,
    }

    print(f"\nParameter estimates (mean +/- std, R-hat, natural scale, {n_chains} chains x {n_samples} samples):")
    for i, name in enumerate(MCMC_PARAM_NAMES_IDENTIFIABLE):
        mean = float(jnp.mean(samples_flat[:, i]))
        std = float(jnp.std(samples_flat[:, i]))
        true_val = float(param_array_ref[i])
        rhat = float(rhats[i])
        print(f"  {name:10s}: {mean:8.4f} +/- {std:6.4f}  (true: {true_val:.4f}, R-hat: {rhat:.3f})")

    print(f"\nFixed (unidentifiable) parameters:")
    for name, val in FIXED_PARAMS_UNIDENTIFIABLE.items():
        print(f"  {name:10s}: {val:.4f} (fixed)")

    sys.stdout.flush()
    return results


def main():
    print("=" * 80)
    print("Identifiable-Only MCMC: Steady-State Isotope Labeling")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()
    params = DEFAULT_PARAMS.copy()
    initial = DEFAULT_INITIAL.copy()

    print("\nApplying Wegscheider constraints...")
    params = compute_wegscheider_constraints(params)
    print(f"  k2f = {params['k2f']:.6f}")
    print(f"  k_pq_r = {params['k_pq_r']:.6f}")
    sys.stdout.flush()

    # Run MCMC
    mcmc_results = run_mcmc(
        ref_params=DEFAULT_PARAMS,
        initial=DEFAULT_INITIAL,
    )

    # Save results
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

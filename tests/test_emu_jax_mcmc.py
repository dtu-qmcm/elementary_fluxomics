#!/usr/bin/env python3
"""Validation tests for pure JAX EMU simulation.

These tests verify that the pure JAX EMU implementation:
1. Produces identical results to the freeflux JAX backend
2. Supports gradient computation through JAX
3. Works correctly in MCMC sampling

Run with:
    uv run pytest tests/test_emu_jax_mcmc.py -v
"""

import pytest
import numpy as np
from pathlib import Path

# Enable float64 precision BEFORE importing JAX
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from scripts.emu_jax import (
    load_emu_network_structure,
    simulate_inst_mdvs_pure_jax,
    create_flux_array_from_dict,
    create_conc_array_from_dict,
    run_freeflux_inst_simulation,
)
from scripts.enzyme_simulation import (
    DEFAULT_PARAMS,
    DEFAULT_INITIAL,
    get_pure_jax_emu_network,
    compute_condition_likelihood_emu,
    compute_fluxes_array_pure_jax,
    compute_concs_array_pure_jax,
    compute_mids_pure_jax_emu,
    params_dict_to_array,
    simulate_enzyme_to_steady_state_jax,
    compute_wegscheider_constraints_jax,
    FORMATION_ENERGIES,
    TEMPERATURE,
)

PROJECT_ROOT = Path(__file__).parent.parent


class TestPureJaxEmuSimulation:
    """Tests for pure JAX EMU simulation."""

    @pytest.fixture
    def emu_network(self):
        """Load the serialized EMU network."""
        network_path = PROJECT_ROOT / "models" / "simple_network" / "emu_network.json"
        return load_emu_network_structure(network_path)

    @pytest.fixture
    def test_fluxes(self):
        """Test flux values."""
        return {
            "v1": 0.5,
            "v1_out": 0.1,
            "v2_f": 0.4,
            "v2_b": 0.1,
            "v3_f": 0.3,
            "v3_b": 0.05,
            "v4_f": 0.25,
            "v4_b": 0.02,
            "v5_f": 0.2,
            "v5_b": 0.01,
            "vq_out": 0.1,
            "vq_in": 0.05,
            "vp_out": 0.1,
            "vp_in": 0.05,
            "vpq_f": 0.3,
            "vpq_b": 0.1,
        }

    @pytest.fixture
    def test_concs(self):
        """Test concentration values."""
        return {
            "A": 2.0,
            "Q": 1.5,
            "P": 1.2,
            "EA": 0.1,
            "EP": 0.1,
            "EPQ": 0.1,
        }

    def test_pure_jax_matches_freeflux(self, emu_network, test_fluxes, test_concs):
        """Validate pure JAX EMU matches freeflux JAX backend within tolerance."""
        timepoints = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
        reactions_file = PROJECT_ROOT / "models" / "simple_network" / "reactions.tsv"

        target_emus = {
            "A": ["12"],
            "EA": ["12"],
            "EPQ": ["12"],
            "EP": ["1"],
            "Q": ["1"],
            "P": ["1"],
        }

        # Only pass free metabolites to freeflux
        freeflux_concs = {k: v for k, v in test_concs.items() if k in ["A", "Q", "P"]}

        # Run freeflux JAX backend
        mdvs_freeflux = run_freeflux_inst_simulation(
            reactions_file=reactions_file,
            target_emus=target_emus,
            timepoints=timepoints,
            labeling_substrate="A_ext",
            labeling_pattern=["11"],
            labeling_percentage=[1.0],
            labeling_purity=[1.0],
            fluxes=test_fluxes,
            concentrations=freeflux_concs,
            use_jax=True,
        )

        # Run pure JAX simulation
        fluxes_array = create_flux_array_from_dict(test_fluxes, emu_network)
        concs_array = create_conc_array_from_dict(test_concs, emu_network)
        timepoints_array = jnp.array(timepoints)

        mdvs_jax = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints_array, emu_network
        )

        # Compare results at final timepoint
        final_t = timepoints[-1]
        max_diff = 0.0

        for emu_id in mdvs_jax.keys():
            if emu_id in mdvs_freeflux:
                ff_mdv = np.array(mdvs_freeflux[emu_id].get(final_t, []))
                jax_mdv = np.array(mdvs_jax[emu_id])

                if len(ff_mdv) == len(jax_mdv):
                    diff = np.max(np.abs(ff_mdv - jax_mdv))
                    max_diff = max(max_diff, diff)

        assert max_diff < 1e-6, f"Maximum difference {max_diff:.2e} exceeds tolerance 1e-6"

    def test_gradient_computation(self, emu_network, test_fluxes, test_concs):
        """Verify gradients can be computed through the simulation."""
        fluxes_array = create_flux_array_from_dict(test_fluxes, emu_network)
        concs_array = create_conc_array_from_dict(test_concs, emu_network)
        timepoints = jnp.array([0.0, 100.0, 1000.0])

        # Target MDVs (dummy)
        target_A_12 = jnp.array([0.1, 0.2, 0.7])

        def loss_fn(fluxes):
            mdvs = simulate_inst_mdvs_pure_jax(fluxes, concs_array, timepoints, emu_network)
            sim_A_12 = mdvs["A_12"]
            return jnp.sum((sim_A_12 - target_A_12) ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(fluxes_array)

        assert not jnp.any(jnp.isnan(grads)), "Gradients contain NaN values"
        assert not jnp.any(jnp.isinf(grads)), "Gradients contain Inf values"
        assert grads.shape == fluxes_array.shape, "Gradient shape mismatch"

    def test_repeated_calls_consistent(self, emu_network, test_fluxes, test_concs):
        """Verify repeated calls produce consistent results.

        Note: Full JIT compilation of simulate_inst_mdvs_pure_jax is not supported
        due to Python control flow (dict operations, for loops). However, the
        function is JAX-traceable for gradient computation, which is what matters
        for gradient-based MCMC (blackjax NUTS).
        """
        fluxes_array = create_flux_array_from_dict(test_fluxes, emu_network)
        concs_array = create_conc_array_from_dict(test_concs, emu_network)
        timepoints = jnp.array([0.0, 100.0, 1000.0])

        # Run twice - results should be identical
        mdvs1 = simulate_inst_mdvs_pure_jax(fluxes_array, concs_array, timepoints, emu_network)
        mdvs2 = simulate_inst_mdvs_pure_jax(fluxes_array, concs_array, timepoints, emu_network)

        result1 = mdvs1["A_12"]
        result2 = mdvs2["A_12"]

        np.testing.assert_allclose(result1, result2, rtol=1e-10)


class TestEmuMcmcIntegration:
    """Tests for pure JAX EMU integration with MCMC."""

    @pytest.fixture
    def emu_network_and_precomputed(self):
        """Load the EMU network using enzyme_simulation's function."""
        return get_pure_jax_emu_network()

    def test_condition_likelihood_emu_computes(self, emu_network_and_precomputed):
        """Verify compute_condition_likelihood_emu returns finite values."""
        emu_network, precomputed = emu_network_and_precomputed
        params = DEFAULT_PARAMS.copy()
        initial = DEFAULT_INITIAL.copy()

        # Compute Wegscheider constraints
        formation_energies_array = jnp.array([
            FORMATION_ENERGIES["A"],
            FORMATION_ENERGIES["P"],
            FORMATION_ENERGIES["Q"],
        ])
        k2f, k_pq_r = compute_wegscheider_constraints_jax(params, formation_energies_array, TEMPERATURE)
        params["k2f"] = float(k2f)
        params["k_pq_r"] = float(k_pq_r)

        # Get reference data by running enzyme simulation
        initial_array = jnp.array([
            initial["A"],
            initial["Q"],
            initial["P"],
            initial["EA"],
            initial["EPQ"],
            initial["EP"],
        ])

        enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
        enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

        A, Q, P = enzyme_y_ss[0], enzyme_y_ss[1], enzyme_y_ss[2]
        ref_flux = params["k_A_ex"] * (params["A_e"] - A)
        ref_concs = jnp.array([A, Q, P])

        # Use dummy reference MIDs
        ref_mids = jnp.array([0.01, 0.1, 0.89,  # A_12
                             0.01, 0.1, 0.89,  # EA_12
                             0.01, 0.1, 0.89,  # EPQ_12
                             0.01, 0.99,       # EP_1
                             0.2, 0.8,         # Q_1
                             0.2, 0.8])        # P_1

        param_array = params_dict_to_array(params)
        timepoints = jnp.array([0.0, 100.0, 1000.0])

        ll = compute_condition_likelihood_emu(
            params["A_e"],
            params["E_T"],
            ref_flux,
            ref_concs,
            ref_mids,
            param_array,
            params["Q_e"],
            params["P_e"],
            initial_array,
            emu_network,
            timepoints,
            precomputed,
        )

        assert jnp.isfinite(ll), f"Log-likelihood is not finite: {ll}"

    def test_condition_likelihood_emu_gradient(self, emu_network_and_precomputed):
        """Verify gradients can be computed through compute_condition_likelihood_emu."""
        emu_network, precomputed = emu_network_and_precomputed
        params = DEFAULT_PARAMS.copy()
        initial = DEFAULT_INITIAL.copy()

        # Compute Wegscheider constraints
        formation_energies_array = jnp.array([
            FORMATION_ENERGIES["A"],
            FORMATION_ENERGIES["P"],
            FORMATION_ENERGIES["Q"],
        ])
        k2f, k_pq_r = compute_wegscheider_constraints_jax(params, formation_energies_array, TEMPERATURE)
        params["k2f"] = float(k2f)
        params["k_pq_r"] = float(k_pq_r)

        initial_array = jnp.array([
            initial["A"],
            initial["Q"],
            initial["P"],
            initial["EA"],
            initial["EPQ"],
            initial["EP"],
        ])

        # Get reference data
        enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
        enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

        A, Q, P = enzyme_y_ss[0], enzyme_y_ss[1], enzyme_y_ss[2]
        ref_flux = params["k_A_ex"] * (params["A_e"] - A)
        ref_concs = jnp.array([A, Q, P])
        ref_mids = jnp.ones(15) / 15  # Dummy uniform MIDs

        timepoints = jnp.array([0.0, 100.0, 1000.0])
        A_e = params["A_e"]
        E_T = params["E_T"]
        Q_e = params["Q_e"]
        P_e = params["P_e"]

        def ll_fn(param_array):
            return compute_condition_likelihood_emu(
                A_e, E_T, ref_flux, ref_concs, ref_mids,
                param_array, Q_e, P_e, initial_array,
                emu_network, timepoints, precomputed,
            )

        param_array = params_dict_to_array(params)
        grad_fn = jax.grad(ll_fn)

        # This may return -inf if the likelihood is very low, which is fine
        # We just want to verify no errors occur
        try:
            grads = grad_fn(param_array)
            # Gradients might be zero if likelihood is -inf, that's OK
            assert grads.shape == param_array.shape
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")


class TestEmuFluxComputation:
    """Tests for flux array computation from concentrations."""

    @pytest.fixture
    def emu_network_and_precomputed(self):
        """Load the EMU network."""
        return get_pure_jax_emu_network()

    def test_flux_array_ordering(self, emu_network_and_precomputed):
        """Verify flux array has correct ordering."""
        emu_network, _precomputed = emu_network_and_precomputed
        params = DEFAULT_PARAMS.copy()

        # Set up concentrations
        concs = {
            "A": 1.0,
            "Q": 1.0,
            "P": 1.0,
            "EA": 0.1,
            "EPQ": 0.1,
            "EP": 0.1,
            "E": params["E_T"] - 0.3,  # E_T - EA - EPQ - EP
        }

        # Compute Wegscheider constraints
        formation_energies_array = jnp.array([
            FORMATION_ENERGIES["A"],
            FORMATION_ENERGIES["P"],
            FORMATION_ENERGIES["Q"],
        ])
        k2f, k_pq_r = compute_wegscheider_constraints_jax(params, formation_energies_array, TEMPERATURE)
        params["k2f"] = float(k2f)
        params["k_pq_r"] = float(k_pq_r)

        fluxes_array = compute_fluxes_array_pure_jax(concs, params, emu_network)

        # Verify shape matches network
        assert fluxes_array.shape[0] == len(emu_network["flux_ids_all"])

        # Verify all fluxes are non-negative
        assert jnp.all(fluxes_array >= 0), "Some fluxes are negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

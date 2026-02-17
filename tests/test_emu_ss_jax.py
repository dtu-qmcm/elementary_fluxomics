#!/usr/bin/env python3
"""Tests for steady-state EMU JAX implementation.

This module contains tests to verify that the JAX steady-state implementation
produces results matching the freeflux reference implementation.

Tests:
1. Pure JAX steady-state matches freeflux native
2. Gradients can be computed through steady-state simulation
3. MDVs are properly normalized (sum to 1)
4. Instationary at long time converges to steady-state

Run with: uv run pytest tests/test_emu_ss_jax.py -v
"""

import sys
from pathlib import Path

# Add scripts to path before importing JAX-related modules
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import emu_jax first to ensure JAX config is set before jax.numpy import
from emu_jax import (
    simulate_ss_mdvs_pure_jax,
    simulate_inst_mdvs_pure_jax,
    run_freeflux_ss_simulation,
    build_emu_network,
    create_flux_array_from_dict,
    create_conc_array_from_dict,
)

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ============================================================================
# Test Configuration
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
REACTIONS_FILE = PROJECT_DIR / "models" / "simple_network" / "reactions.tsv"


@pytest.fixture
def network():
    """Build the EMU network structure at runtime."""
    return build_emu_network(
        reactions_file=REACTIONS_FILE,
        target_emus={"A": ["12"], "EA": ["12"], "EPQ": ["12"],
                     "EP": ["1"], "Q": ["1"], "P": ["1"]},
        labeling_substrate="A_ext",
        labeling_pattern=["10"],
        labeling_percentage=[1.0],
        labeling_purity=[1.0],
    )


@pytest.fixture
def test_fluxes():
    """Test flux values for the simple network."""
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
def test_concs():
    """Test concentration values for the simple network."""
    return {
        "A": 2.0,
        "Q": 1.5,
        "P": 1.2,
        "EA": 0.1,
        "EP": 0.1,
        "EPQ": 0.1,
    }


# ============================================================================
# Test: Steady-State Pure JAX vs Freeflux
# ============================================================================

class TestSteadyStatePureJaxVsFreeflux:
    """Tests comparing steady-state pure JAX vs freeflux native."""

    def test_ss_pure_jax_matches_freeflux(self, network, test_fluxes):
        """Verify pure JAX steady-state matches freeflux native.

        Note: freeflux native uses natural abundance (~1.1% ^13C) for all
        substrate MDVs, while pure JAX uses pure unlabeled [1, 0] for
        non-labeled substrates AND recomputes labeled substrate MDVs without
        natural abundance at unlabeled positions. Tolerance is relaxed to
        accommodate this intentional difference (~0.7% max).
        """
        target_emus = {"A": ["12"], "Q": ["1"], "P": ["1"]}

        # Run freeflux native
        mdvs_freeflux = run_freeflux_ss_simulation(
            reactions_file=REACTIONS_FILE,
            target_emus=target_emus,
            labeling_substrate="A_ext",
            labeling_pattern=["10"],
            labeling_percentage=[1.0],
            labeling_purity=[1.0],
            fluxes=test_fluxes,
            use_jax=False,
        )

        # Run pure JAX
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        mdvs_jax = simulate_ss_mdvs_pure_jax(fluxes_array, network)

        # Compare with relaxed tolerance for natural abundance vs pure unlabeled.
        # A_ext_12 with pattern "10": freeflux gives [0, 0.989, 0.011] due to
        # natural abundance at unlabeled atom 2, pure JAX gives [0, 1, 0].
        # This ~1.1% difference propagates to downstream metabolite MDVs.
        for emu_id in mdvs_jax:
            if emu_id in mdvs_freeflux:
                np.testing.assert_allclose(
                    mdvs_freeflux[emu_id],
                    np.array(mdvs_jax[emu_id]),
                    rtol=5e-2,
                    atol=2e-2,
                    err_msg=f"Mismatch for {emu_id}",
                )

    def test_ss_with_different_flux_values(self, network):
        """Test steady-state simulation with different flux configurations.

        Note: relaxed tolerance due to natural abundance vs pure unlabeled
        substrate MDVs (see test_ss_pure_jax_matches_freeflux docstring).
        """
        target_emus = {"A": ["12"], "Q": ["1"], "P": ["1"]}

        # Higher flux configuration
        high_fluxes = {
            "v1": 1.0,
            "v1_out": 0.2,
            "v2_f": 0.8,
            "v2_b": 0.2,
            "v3_f": 0.6,
            "v3_b": 0.1,
            "v4_f": 0.5,
            "v4_b": 0.05,
            "v5_f": 0.4,
            "v5_b": 0.02,
            "vq_out": 0.2,
            "vq_in": 0.1,
            "vp_out": 0.2,
            "vp_in": 0.1,
            "vpq_f": 0.5,
            "vpq_b": 0.2,
        }

        # Run both implementations
        mdvs_freeflux = run_freeflux_ss_simulation(
            reactions_file=REACTIONS_FILE,
            target_emus=target_emus,
            labeling_substrate="A_ext",
            labeling_pattern=["10"],
            labeling_percentage=[1.0],
            labeling_purity=[1.0],
            fluxes=high_fluxes,
            use_jax=False,
        )

        fluxes_array = create_flux_array_from_dict(high_fluxes, network)
        mdvs_jax = simulate_ss_mdvs_pure_jax(fluxes_array, network)

        # Compare with relaxed tolerance (see test_ss_pure_jax_matches_freeflux)
        for emu_id in mdvs_jax:
            if emu_id in mdvs_freeflux:
                np.testing.assert_allclose(
                    mdvs_freeflux[emu_id],
                    np.array(mdvs_jax[emu_id]),
                    rtol=5e-2,
                    atol=2e-2,
                    err_msg=f"Mismatch for {emu_id}",
                )


# ============================================================================
# Test: Gradient Computation
# ============================================================================

class TestSteadyStateGradients:
    """Tests for gradient computation through steady-state simulation."""

    def test_ss_gradient_computation(self, network, test_fluxes):
        """Verify gradients can be computed through steady-state simulation."""
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        target_mdv = jnp.array([0.1, 0.2, 0.7])

        def loss_fn(fluxes):
            mdvs = simulate_ss_mdvs_pure_jax(fluxes, network)
            return jnp.sum((mdvs["A_12"] - target_mdv) ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(fluxes_array)

        assert not jnp.any(jnp.isnan(grads)), "Gradients contain NaN"
        assert not jnp.any(jnp.isinf(grads)), "Gradients contain Inf"
        assert grads.shape == fluxes_array.shape, "Gradient shape mismatch"

    def test_ss_gradient_values_reasonable(self, network, test_fluxes):
        """Test that gradient values are reasonable (not all zeros)."""
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        target_mdv = jnp.array([0.1, 0.2, 0.7])

        def loss_fn(fluxes):
            mdvs = simulate_ss_mdvs_pure_jax(fluxes, network)
            return jnp.sum((mdvs["A_12"] - target_mdv) ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(fluxes_array)

        # At least some gradients should be non-zero
        assert jnp.any(jnp.abs(grads) > 1e-10), "All gradients are zero"

    def test_ss_jit_compilation(self, network, test_fluxes):
        """Test that steady-state simulation can be JIT compiled."""
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)

        # JIT compile the simulation
        @jax.jit
        def jitted_simulate(fluxes):
            mdvs = simulate_ss_mdvs_pure_jax(fluxes, network)
            # Return sum of one MDV to get a scalar
            return jnp.sum(mdvs["A_12"])

        # Should run without error
        result = jitted_simulate(fluxes_array)
        assert jnp.isfinite(result), "JIT result is not finite"


# ============================================================================
# Test: MDV Normalization
# ============================================================================

class TestMdvNormalization:
    """Tests for MDV normalization (sum to 1)."""

    def test_ss_mdv_normalization(self, network, test_fluxes):
        """Verify steady-state MDVs sum to 1."""
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        mdvs = simulate_ss_mdvs_pure_jax(fluxes_array, network)

        for emu_id, mdv in mdvs.items():
            mdv_sum = jnp.sum(mdv)
            assert jnp.abs(mdv_sum - 1.0) < 1e-6, f"{emu_id} MDV sum = {mdv_sum}"

    def test_ss_mdv_non_negative(self, network, test_fluxes):
        """Verify steady-state MDVs are non-negative."""
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        mdvs = simulate_ss_mdvs_pure_jax(fluxes_array, network)

        for emu_id, mdv in mdvs.items():
            # Allow small negative values due to numerical precision
            assert jnp.all(mdv >= -1e-6), f"{emu_id} has negative MDV values: {mdv}"


# ============================================================================
# Test: Steady-State vs Instationary Convergence
# ============================================================================

class TestSteadyStateVsInstationaryConvergence:
    """Tests verifying instationary converges to steady-state."""

    def test_ss_vs_inst_convergence(self, network, test_fluxes, test_concs):
        """Verify instationary simulation converges to steady-state."""
        # Run steady-state
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        mdvs_ss = simulate_ss_mdvs_pure_jax(fluxes_array, network)

        # Run instationary to very long time to ensure convergence
        concs_array = create_conc_array_from_dict(test_concs, network)
        timepoints = jnp.array([0.0, 10000.0])  # Very long time for convergence

        mdvs_inst = simulate_inst_mdvs_pure_jax(
            fluxes_array, concs_array, timepoints, network
        )

        # Compare - instationary at long time should match steady-state
        # Use relaxed tolerance since:
        # 1. INST may not fully converge even at t=10000 for very small MDV values
        # 2. atol=1e-3 handles near-zero M+0 values (e.g., A_1 ≈ [0, 1] with
        #    pattern "10") where rtol would be too strict
        for emu_id in mdvs_ss:
            if emu_id in mdvs_inst:
                np.testing.assert_allclose(
                    np.array(mdvs_ss[emu_id]),
                    np.array(mdvs_inst[emu_id]),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"INST at t=10000 doesn't match SS for {emu_id}",
                )

    def test_ss_faster_than_inst(self, network, test_fluxes, test_concs):
        """Verify steady-state is faster than instationary (sanity check)."""
        import time

        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        concs_array = create_conc_array_from_dict(test_concs, network)
        timepoints = jnp.array([0.0, 100.0, 1000.0])

        # Time steady-state
        start = time.perf_counter()
        for _ in range(10):
            simulate_ss_mdvs_pure_jax(fluxes_array, network)
        ss_time = time.perf_counter() - start

        # Time instationary
        start = time.perf_counter()
        for _ in range(10):
            simulate_inst_mdvs_pure_jax(fluxes_array, concs_array, timepoints, network)
        inst_time = time.perf_counter() - start

        # Steady-state should be faster (or at least not much slower)
        # Allow some tolerance since JIT compilation affects first run
        assert ss_time < inst_time * 2, f"SS time {ss_time:.3f} not faster than INST {inst_time:.3f}"


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ss_produces_valid_mdvs(self, network, test_fluxes):
        """Test that steady-state produces valid MDVs."""
        fluxes_array = create_flux_array_from_dict(test_fluxes, network)
        mdvs = simulate_ss_mdvs_pure_jax(fluxes_array, network)

        # Should produce valid MDVs
        assert len(mdvs) > 0, "No MDVs produced"
        for emu_id, mdv in mdvs.items():
            assert jnp.all(jnp.isfinite(mdv)), f"Non-finite MDV for {emu_id}"

    def test_ss_with_small_fluxes(self, network):
        """Test steady-state with very small flux values."""
        small_fluxes = {
            "v1": 0.001,
            "v1_out": 0.0002,
            "v2_f": 0.0008,
            "v2_b": 0.0002,
            "v3_f": 0.0006,
            "v3_b": 0.0001,
            "v4_f": 0.0005,
            "v4_b": 0.00005,
            "v5_f": 0.0004,
            "v5_b": 0.00002,
            "vq_out": 0.0002,
            "vq_in": 0.0001,
            "vp_out": 0.0002,
            "vp_in": 0.0001,
            "vpq_f": 0.0006,
            "vpq_b": 0.0002,
        }

        fluxes_array = create_flux_array_from_dict(small_fluxes, network)
        mdvs = simulate_ss_mdvs_pure_jax(fluxes_array, network)

        # Should still produce valid MDVs
        for emu_id, mdv in mdvs.items():
            assert jnp.all(jnp.isfinite(mdv)), f"Non-finite MDV for {emu_id}"
            mdv_sum = jnp.sum(mdv)
            assert jnp.abs(mdv_sum - 1.0) < 1e-4, f"{emu_id} MDV sum = {mdv_sum}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

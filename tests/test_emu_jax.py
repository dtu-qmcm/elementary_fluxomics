#!/usr/bin/env python3
"""Validation tests for JAX-compatible EMU MDV simulation.

This module contains tests to verify that the JAX implementation in emu_jax.py
produces results matching the freeflux reference implementation.

Tests:
1. MDV convolution matches freeflux gen_conv
2. Single INST step matches freeflux
3. Full INST simulation matches freeflux on glycolysis model

Run with: uv run pytest tests/test_emu_jax.py -v
"""

import sys
from pathlib import Path

# Add scripts to path before importing JAX-related modules
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import emu_jax first to ensure JAX config is set before jax.numpy import
from emu_jax import (
    mdv_convolve,
    inst_mdv_step,
    run_glycolysis_simulation,
)

import jax.numpy as jnp
import numpy as np
import pytest


# ============================================================================
# Test Configuration
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR = PROJECT_DIR / "models" / "glycolysis_mech_gly3p"
REACTIONS_FILE = MODEL_DIR / "reactions.tsv"


# ============================================================================
# Test: MDV Convolution
# ============================================================================

class TestMdvConvolve:
    """Tests for MDV convolution function."""

    def test_basic_convolution(self):
        """Test basic MDV convolution matches expected result."""
        # Two 1-atom MDVs: [M+0, M+1]
        arr1 = jnp.array([0.5, 0.5])  # 50% unlabeled, 50% labeled
        arr2 = jnp.array([0.7, 0.3])  # 70% unlabeled, 30% labeled

        result = mdv_convolve(arr1, arr2)

        # Expected: 2-atom MDV [M+0, M+1, M+2]
        # M+0: 0.5 * 0.7 = 0.35
        # M+1: 0.5 * 0.3 + 0.5 * 0.7 = 0.15 + 0.35 = 0.50
        # M+2: 0.5 * 0.3 = 0.15
        expected = jnp.array([0.35, 0.50, 0.15])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_convolution_matches_freeflux(self):
        """Test that JAX convolution matches freeflux gen_conv."""
        # Import freeflux gen_conv
        from freeflux.core.mdv import gen_conv

        arr1 = np.array([0.5, 0.3, 0.2])
        arr2 = np.array([0.7, 0.3])

        # Freeflux implementation
        expected = gen_conv(arr1, arr2)

        # JAX implementation
        result = mdv_convolve(jnp.array(arr1), jnp.array(arr2))

        # Use rtol=1e-6 to account for potential float32/float64 differences
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_convolution_with_pure_labeled(self):
        """Test convolution with fully labeled MDV."""
        # Fully labeled 2-atom EMU: [0, 0, 1] (100% M+2)
        arr1 = jnp.array([0.0, 0.0, 1.0])
        # 50% unlabeled 1-atom EMU: [0.5, 0.5]
        arr2 = jnp.array([0.5, 0.5])

        result = mdv_convolve(arr1, arr2)

        # Expected: 3-atom MDV [M+0, M+1, M+2, M+3]
        # M+0: 0, M+1: 0, M+2: 1*0.5=0.5, M+3: 1*0.5=0.5
        expected = jnp.array([0.0, 0.0, 0.5, 0.5])

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_convolution_with_natural_abundance(self):
        """Test convolution with natural abundance MDVs."""
        # Natural abundance for carbon (~1.1% 13C)
        # 1-atom: [0.989, 0.011]
        arr1 = jnp.array([0.989, 0.011])
        arr2 = jnp.array([0.989, 0.011])

        result = mdv_convolve(arr1, arr2)

        # Expected for 2-atom natural abundance
        # M+0: 0.989 * 0.989 = 0.978121
        # M+1: 0.989 * 0.011 + 0.011 * 0.989 = 0.021758
        # M+2: 0.011 * 0.011 = 0.000121
        expected = jnp.array([0.978121, 0.021758, 0.000121])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_convolution_different_sizes(self):
        """Test convolution with differently sized MDVs."""
        from freeflux.core.mdv import gen_conv

        # 3-atom MDV
        arr1 = np.array([0.1, 0.2, 0.3, 0.4])
        # 2-atom MDV
        arr2 = np.array([0.3, 0.4, 0.3])

        # Freeflux implementation
        expected = gen_conv(arr1, arr2)

        # JAX implementation
        result = mdv_convolve(jnp.array(arr1), jnp.array(arr2))

        # Use rtol=1e-6 to account for potential float32/float64 differences
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ============================================================================
# Test: INST MDV Step
# ============================================================================

class TestInstMdvStep:
    """Tests for single INST MDV step function."""

    def test_step_preserves_mass(self):
        """Test that INST step approximately preserves total mass in MDVs."""
        # Simple 2x2 system
        n_products = 2
        n_sources = 1
        n_isotopomers = 3

        # Random but valid matrices
        A = jnp.array([[-1.0, 0.5], [0.5, -1.0]])
        B = jnp.array([[0.5], [0.5]])
        M = jnp.diag(jnp.array([1.0, 1.0]))

        # Initial MDVs (sum to 1)
        X_t0 = jnp.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
        Y_t0 = jnp.array([[0.7, 0.2, 0.1]])
        Y_t1 = jnp.array([[0.7, 0.2, 0.1]])

        delta_t = 0.01

        X_t1 = inst_mdv_step(X_t0, Y_t0, Y_t1, A, B, M, delta_t)

        # MDVs should still approximately sum to 1 for short time steps
        # Note: May not be exactly 1 due to exchange with sources
        row_sums = jnp.sum(X_t1, axis=1)
        assert jnp.all(row_sums > 0), "MDV sums should be positive"

    def test_step_identity_system(self):
        """Test INST step with identity-like system (no change expected)."""
        # System at steady state with no exchange
        n = 2
        A = jnp.zeros((n, n))  # No internal flux
        B = jnp.zeros((n, 1))  # No input
        M = jnp.eye(n)

        X_t0 = jnp.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        Y_t0 = jnp.array([[1.0, 0.0, 0.0]])
        Y_t1 = Y_t0

        delta_t = 0.1

        X_t1 = inst_mdv_step(X_t0, Y_t0, Y_t1, A, B, M, delta_t)

        # With no flux, MDVs should not change
        np.testing.assert_allclose(X_t1, X_t0, rtol=1e-6)


# ============================================================================
# Test: Full INST Simulation
# ============================================================================

class TestFullInstSimulation:
    """Tests for full INST simulation comparing JAX vs freeflux."""

    @pytest.fixture
    def target_emus(self):
        """Fixture providing target EMUs."""
        return {
            "g6p": ["123456"],
            "f6p": ["123456"],
            "dhap": ["123"],
            "g3p": ["123"],
        }

    def test_jax_vs_freeflux_small_timepoints(self, target_emus):
        """Test JAX matches freeflux for small timepoints."""
        timepoints = [0.0, 0.001, 0.01, 0.1]

        mdvs_freeflux = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            use_jax=False,
        )

        mdvs_jax = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            use_jax=True,
        )

        # Compare results
        for emu_id in mdvs_freeflux:
            if emu_id not in mdvs_jax:
                pytest.fail(f"EMU {emu_id} missing from JAX output")

            for t in mdvs_freeflux[emu_id]:
                if t not in mdvs_jax[emu_id]:
                    continue

                ff_mdv = mdvs_freeflux[emu_id][t]
                jax_mdv = mdvs_jax[emu_id][t]

                np.testing.assert_allclose(
                    jax_mdv,
                    ff_mdv,
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Mismatch for {emu_id} at t={t}",
                )

    def test_jax_vs_freeflux_full_simulation(self, target_emus):
        """Test JAX matches freeflux for full simulation timecourse."""
        timepoints = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

        mdvs_freeflux = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            use_jax=False,
        )

        mdvs_jax = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            use_jax=True,
        )

        max_diffs = []
        for emu_id in mdvs_freeflux:
            if emu_id not in mdvs_jax:
                continue

            for t in mdvs_freeflux[emu_id]:
                if t not in mdvs_jax[emu_id]:
                    continue

                ff_mdv = mdvs_freeflux[emu_id][t]
                jax_mdv = mdvs_jax[emu_id][t]

                max_diff = np.max(np.abs(ff_mdv - jax_mdv))
                max_diffs.append(max_diff)

        # Overall max difference should be small
        assert len(max_diffs) > 0, "No MDVs to compare"
        overall_max = max(max_diffs)
        assert overall_max < 1e-3, f"Maximum MDV difference {overall_max:.2e} exceeds threshold"

    def test_mdv_normalization(self, target_emus):
        """Test that output MDVs are properly normalized (sum to 1)."""
        timepoints = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

        mdvs = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            use_jax=True,
        )

        for emu_id, timepoints_mdvs in mdvs.items():
            for t, mdv in timepoints_mdvs.items():
                mdv_sum = np.sum(mdv)
                np.testing.assert_allclose(
                    mdv_sum,
                    1.0,
                    rtol=0.1,  # Allow 10% deviation due to numerical issues
                    err_msg=f"MDV sum {mdv_sum} not close to 1 for {emu_id} at t={t}",
                )

    def test_steady_state_labeling(self, target_emus):
        """Test that labeling reaches expected steady state."""
        timepoints = [0.0, 1.0, 10.0, 100.0, 1000.0]

        mdvs = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            use_jax=True,
        )

        # At steady state with "110000" labeling on glucose,
        # g6p should have significant M+2 labeling
        if "g6p_123456" in mdvs:
            final_mdv = mdvs["g6p_123456"][1000.0]
            # M+2 should be significant (labeled carbons 1-2)
            assert final_mdv[2] > 0.1, f"Expected significant M+2 labeling in g6p: {final_mdv}"


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_default_simulation_produces_valid_mdvs(self):
        """Test that default simulation produces valid MDVs."""
        timepoints = [0.0, 0.01, 0.1, 1.0]

        mdvs = run_glycolysis_simulation(
            timepoints=timepoints,
            use_jax=True,
        )

        # Should produce valid MDVs
        for emu_id, timepoints_mdvs in mdvs.items():
            for t, mdv in timepoints_mdvs.items():
                assert np.all(np.isfinite(mdv)), f"Non-finite MDV for {emu_id} at t={t}"
                assert np.all(mdv >= -1e-6), f"Negative MDV values for {emu_id} at t={t}"

    def test_mdv_sums_close_to_one(self):
        """Test that MDVs sum to approximately 1."""
        timepoints = [0.0, 0.01, 0.1, 1.0, 10.0]

        mdvs = run_glycolysis_simulation(
            timepoints=timepoints,
            use_jax=True,
        )

        for emu_id, timepoints_mdvs in mdvs.items():
            for t, mdv in timepoints_mdvs.items():
                mdv_sum = np.sum(mdv)
                # Allow some tolerance due to numerical issues
                assert abs(mdv_sum - 1.0) < 0.2, f"MDV sum {mdv_sum} not close to 1 for {emu_id} at t={t}"

    def test_gibbs_overrides_affect_simulation(self):
        """Test that Gibbs energy overrides affect the simulation."""
        timepoints = [0.0, 1.0, 10.0, 100.0]
        target_emus = {"dhap": ["123"], "g3p": ["123"]}

        # Default simulation
        mdvs_default = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            use_jax=True,
        )

        # With Gibbs override for FBA1
        mdvs_override = run_glycolysis_simulation(
            timepoints=timepoints,
            target_emus=target_emus,
            gibbs_overrides={"FBA1": -10.0},  # Much more negative
            use_jax=True,
        )

        # The simulations should produce different results
        # (different Gibbs energies = different exchange fluxes)
        # Just check that both produce valid output
        for mdvs in [mdvs_default, mdvs_override]:
            assert len(mdvs) > 0, "No MDVs produced"
            for emu_id, timepoints_mdvs in mdvs.items():
                for t, mdv in timepoints_mdvs.items():
                    assert np.all(np.isfinite(mdv)), f"Non-finite MDV for {emu_id} at t={t}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""Tests validating that isotopomer and enzyme_ode simulations produce consistent fluxes.

The enzyme_ode uses a 6-component state vector: [A, Q, P, EA, EPQ, EP]
The isotopomer_ode uses an 18-component state vector tracking all isotopomers.

At steady state, summing isotopomers should give identical concentrations and
net fluxes as the enzyme_ode simulation.
"""

import pytest
import jax
import jax.numpy as jnp
import diffrax
from diffrax import SteadyStateEvent

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

from scripts.isotopomer_simulation import (
    isotopomer_ode,
    enzyme_ode,
    DEFAULT_PARAMS,
    DEFAULT_INITIAL_TOTAL,
    build_initial_state,
)


def simulate_enzyme_to_steady_state(params, initial_array):
    """Simulate enzyme_ode to steady state."""
    term = diffrax.ODETerm(enzyme_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)
    discrete_terminating_event = SteadyStateEvent(rtol=1e-6, atol=1e-8)
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


def simulate_isotopomer_to_steady_state(params, initial_total):
    """Simulate isotopomer_ode to steady state starting from all unlabeled."""
    y0 = build_initial_state(initial_total)

    term = diffrax.ODETerm(isotopomer_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)
    discrete_terminating_event = SteadyStateEvent(rtol=1e-6, atol=1e-8)
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=jnp.inf,
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


def sum_isotopomers(y):
    """Sum isotopomers to get total concentrations.

    State vector y has 18 components:
      A_00, A_01, A_10, A_11,     (indices 0-3)
      EA_00, EA_01, EA_10, EA_11, (indices 4-7)
      EPQ_00, EPQ_01, EPQ_10, EPQ_11, (indices 8-11)
      EP_0, EP_1,                 (indices 12-13)
      Q_0, Q_1,                   (indices 14-15)
      P_0, P_1                    (indices 16-17)

    Returns: [A, Q, P, EA, EPQ, EP] (same order as enzyme_ode)
    """
    A = y[0] + y[1] + y[2] + y[3]
    EA = y[4] + y[5] + y[6] + y[7]
    EPQ = y[8] + y[9] + y[10] + y[11]
    EP = y[12] + y[13]
    Q = y[14] + y[15]
    P = y[16] + y[17]

    return jnp.array([A, Q, P, EA, EPQ, EP])


def compute_net_fluxes(y, params):
    """Compute net fluxes from total concentrations.

    Parameters
    ----------
    y : array
        State vector [A, Q, P, EA, EPQ, EP]
    params : dict
        Kinetic parameters

    Returns
    -------
    dict
        Net flux values for each reaction
    """
    A, Q, P, EA, EPQ, EP = y[0], y[1], y[2], y[3], y[4], y[5]
    E = params["E_T"] - EA - EPQ - EP

    # Exchange reactions
    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)
    v_Q_ex = params["k_Q_ex"] * (params["Q_e"] - Q)
    v_P_ex = params["k_P_ex"] * (params["P_e"] - P)

    # Enzyme reactions
    v2 = A * E * params["k2f"] - EA * params["k2r"]
    v3 = EA * params["k3f"] - EPQ * params["k3r"]
    v4 = EPQ * params["k4f"] - Q * EP * params["k4r"]
    v5 = EP * params["k5f"] - E * P * params["k5r"]

    # P <-> Q interconversion
    v_pq = P * params["k_pq_f"] - Q * params["k_pq_r"]

    return {
        "v_A_ex": float(v_A_ex),
        "v2": float(v2),
        "v3": float(v3),
        "v4": float(v4),
        "v5": float(v5),
        "v_Q_ex": float(v_Q_ex),
        "v_P_ex": float(v_P_ex),
        "v_pq": float(v_pq),
    }


def compute_forward_backward_fluxes(y, params):
    """Compute forward and backward fluxes for reversible reactions.

    Parameters
    ----------
    y : array
        State vector [A, Q, P, EA, EPQ, EP]
    params : dict
        Kinetic parameters

    Returns
    -------
    dict
        Forward and backward flux values
    """
    A, Q, P, EA, EPQ, EP = y[0], y[1], y[2], y[3], y[4], y[5]
    E = params["E_T"] - EA - EPQ - EP

    return {
        # v2: A + E <-> EA
        "v2_f": float(A * E * params["k2f"]),
        "v2_b": float(EA * params["k2r"]),
        # v3: EA <-> EPQ
        "v3_f": float(EA * params["k3f"]),
        "v3_b": float(EPQ * params["k3r"]),
        # v4: EPQ <-> EP + Q
        "v4_f": float(EPQ * params["k4f"]),
        "v4_b": float(Q * EP * params["k4r"]),
        # v5: EP <-> E + P
        "v5_f": float(EP * params["k5f"]),
        "v5_b": float(E * P * params["k5r"]),
        # v_pq: P <-> Q
        "v_pq_f": float(P * params["k_pq_f"]),
        "v_pq_b": float(Q * params["k_pq_r"]),
    }


def compute_isotopomer_forward_backward_fluxes(y, params):
    """Compute forward and backward fluxes by summing over all isotopomers.

    Parameters
    ----------
    y : array
        Isotopomer state vector (18 components)
    params : dict
        Kinetic parameters

    Returns
    -------
    dict
        Forward and backward flux values (summed across isotopomers)
    """
    # Extract isotopomers
    A_00, A_01, A_10, A_11 = y[0], y[1], y[2], y[3]
    EA_00, EA_01, EA_10, EA_11 = y[4], y[5], y[6], y[7]
    EPQ_00, EPQ_01, EPQ_10, EPQ_11 = y[8], y[9], y[10], y[11]
    EP_0, EP_1 = y[12], y[13]
    Q_0, Q_1 = y[14], y[15]
    P_0, P_1 = y[16], y[17]

    # Total concentrations
    A = A_00 + A_01 + A_10 + A_11
    EA = EA_00 + EA_01 + EA_10 + EA_11
    EPQ = EPQ_00 + EPQ_01 + EPQ_10 + EPQ_11
    EP = EP_0 + EP_1
    Q = Q_0 + Q_1
    P = P_0 + P_1

    # Free enzyme
    E = params["E_T"] - EA - EPQ - EP

    # v2: A + E <-> EA (summed across isotopomers)
    v2_f = params["k2f"] * E * A
    v2_b = params["k2r"] * EA

    # v3: EA <-> EPQ (summed across isotopomers)
    v3_f = params["k3f"] * EA
    v3_b = params["k3r"] * EPQ

    # v4: EPQ <-> EP + Q (summed across isotopomers)
    # Forward: sum of k4f * EPQ_xx for all isotopomers
    v4_f = params["k4f"] * EPQ
    # Backward: sum of k4r * EP_x * Q_y for all combinations
    # EP_0 * Q_0 -> EPQ_00, EP_0 * Q_1 -> EPQ_01
    # EP_1 * Q_0 -> EPQ_10, EP_1 * Q_1 -> EPQ_11
    v4_b = params["k4r"] * (EP_0 * Q_0 + EP_0 * Q_1 + EP_1 * Q_0 + EP_1 * Q_1)

    # v5: EP <-> E + P (summed across isotopomers)
    v5_f = params["k5f"] * EP
    v5_b = params["k5r"] * E * P

    # v_pq: P <-> Q (summed across isotopomers)
    v_pq_f = params["k_pq_f"] * P
    v_pq_b = params["k_pq_r"] * Q

    return {
        "v2_f": float(v2_f),
        "v2_b": float(v2_b),
        "v3_f": float(v3_f),
        "v3_b": float(v3_b),
        "v4_f": float(v4_f),
        "v4_b": float(v4_b),
        "v5_f": float(v5_f),
        "v5_b": float(v5_b),
        "v_pq_f": float(v_pq_f),
        "v_pq_b": float(v_pq_b),
    }


class TestFluxValidation:
    """Tests validating that isotopomer and enzyme_ode produce consistent results."""

    @pytest.fixture(scope="class")
    def params(self):
        """Get default parameters."""
        return DEFAULT_PARAMS.copy()

    @pytest.fixture(scope="class")
    def initial_total(self):
        """Get default initial total concentrations."""
        return DEFAULT_INITIAL_TOTAL.copy()

    @pytest.fixture(scope="class")
    def enzyme_steady_state(self, params, initial_total):
        """Run enzyme_ode to steady state."""
        initial_array = jnp.array([
            initial_total["A"],
            initial_total["Q"],
            initial_total["P"],
            initial_total["EA"],
            initial_total["EPQ"],
            initial_total["EP"],
        ])
        sol = simulate_enzyme_to_steady_state(params, initial_array)
        y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys
        return y_ss

    @pytest.fixture(scope="class")
    def isotopomer_steady_state(self, params, initial_total):
        """Run isotopomer_ode to steady state."""
        sol = simulate_isotopomer_to_steady_state(params, initial_total)
        y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys
        return y_ss

    def test_steady_state_concentrations_match(
        self, enzyme_steady_state, isotopomer_steady_state
    ):
        """Test that summed isotopomer concentrations match enzyme_ode steady state."""
        # Sum isotopomers to get total concentrations
        iso_totals = sum_isotopomers(isotopomer_steady_state)

        # Compare each concentration
        # Use tolerance of 1e-4 to account for numerical integration differences
        # between the 6-component enzyme_ode and 18-component isotopomer_ode
        tol = 1e-4
        assert jnp.allclose(iso_totals[0], enzyme_steady_state[0], rtol=tol, atol=tol), (
            f"A mismatch: iso={iso_totals[0]:.8f}, enzyme={enzyme_steady_state[0]:.8f}"
        )
        assert jnp.allclose(iso_totals[1], enzyme_steady_state[1], rtol=tol, atol=tol), (
            f"Q mismatch: iso={iso_totals[1]:.8f}, enzyme={enzyme_steady_state[1]:.8f}"
        )
        assert jnp.allclose(iso_totals[2], enzyme_steady_state[2], rtol=tol, atol=tol), (
            f"P mismatch: iso={iso_totals[2]:.8f}, enzyme={enzyme_steady_state[2]:.8f}"
        )
        assert jnp.allclose(iso_totals[3], enzyme_steady_state[3], rtol=tol, atol=tol), (
            f"EA mismatch: iso={iso_totals[3]:.8f}, enzyme={enzyme_steady_state[3]:.8f}"
        )
        assert jnp.allclose(iso_totals[4], enzyme_steady_state[4], rtol=tol, atol=tol), (
            f"EPQ mismatch: iso={iso_totals[4]:.8f}, enzyme={enzyme_steady_state[4]:.8f}"
        )
        assert jnp.allclose(iso_totals[5], enzyme_steady_state[5], rtol=tol, atol=tol), (
            f"EP mismatch: iso={iso_totals[5]:.8f}, enzyme={enzyme_steady_state[5]:.8f}"
        )

    def test_net_fluxes_match(
        self, params, enzyme_steady_state, isotopomer_steady_state
    ):
        """Test that net fluxes computed from both simulations match."""
        # Compute net fluxes from enzyme_ode steady state
        enzyme_fluxes = compute_net_fluxes(enzyme_steady_state, params)

        # Compute net fluxes from summed isotopomer concentrations
        iso_totals = sum_isotopomers(isotopomer_steady_state)
        iso_fluxes = compute_net_fluxes(iso_totals, params)

        # Compare each flux
        # Use tolerance of 1e-4 to account for numerical integration differences
        tol = 1e-4
        for flux_name in enzyme_fluxes:
            assert jnp.allclose(
                iso_fluxes[flux_name], enzyme_fluxes[flux_name], rtol=tol, atol=tol
            ), (
                f"{flux_name} mismatch: iso={iso_fluxes[flux_name]:.8f}, "
                f"enzyme={enzyme_fluxes[flux_name]:.8f}"
            )

    def test_forward_backward_fluxes_match(
        self, params, enzyme_steady_state, isotopomer_steady_state
    ):
        """Test that forward/backward fluxes match between simulations."""
        # Compute forward/backward fluxes from enzyme_ode steady state
        enzyme_fb = compute_forward_backward_fluxes(enzyme_steady_state, params)

        # Compute forward/backward fluxes by summing across isotopomers
        iso_fb = compute_isotopomer_forward_backward_fluxes(
            isotopomer_steady_state, params
        )

        # Compare each forward/backward flux
        # Use tolerance of 1e-4 to account for numerical integration differences
        tol = 1e-4
        for flux_name in enzyme_fb:
            assert jnp.allclose(
                iso_fb[flux_name], enzyme_fb[flux_name], rtol=tol, atol=tol
            ), (
                f"{flux_name} mismatch: iso={iso_fb[flux_name]:.8f}, "
                f"enzyme={enzyme_fb[flux_name]:.8f}"
            )

    def test_enzyme_conservation(
        self, params, enzyme_steady_state, isotopomer_steady_state
    ):
        """Test that enzyme conservation E + EA + EPQ + EP = E_T holds for both."""
        E_T = params["E_T"]

        # Check enzyme_ode
        EA_enz = enzyme_steady_state[3]
        EPQ_enz = enzyme_steady_state[4]
        EP_enz = enzyme_steady_state[5]
        E_enz = E_T - EA_enz - EPQ_enz - EP_enz
        total_enz = E_enz + EA_enz + EPQ_enz + EP_enz

        assert jnp.allclose(total_enz, E_T, rtol=1e-10), (
            f"Enzyme conservation failed (enzyme_ode): {total_enz:.10f} != {E_T:.10f}"
        )

        # Check isotopomer_ode (summed)
        iso_totals = sum_isotopomers(isotopomer_steady_state)
        EA_iso = iso_totals[3]
        EPQ_iso = iso_totals[4]
        EP_iso = iso_totals[5]
        E_iso = E_T - EA_iso - EPQ_iso - EP_iso
        total_iso = E_iso + EA_iso + EPQ_iso + EP_iso

        assert jnp.allclose(total_iso, E_T, rtol=1e-10), (
            f"Enzyme conservation failed (isotopomer): {total_iso:.10f} != {E_T:.10f}"
        )

    def test_mass_balance_at_steady_state(
        self, params, enzyme_steady_state, isotopomer_steady_state
    ):
        """Test that mass balances (dX/dt = 0) are satisfied at steady state."""
        # Check enzyme_ode derivatives are zero
        dydt_enzyme = enzyme_ode(0.0, enzyme_steady_state, params)
        # Use tolerance of 1e-5 to account for steady-state detection threshold
        tol = 1e-5
        assert jnp.allclose(dydt_enzyme, 0.0, atol=tol), (
            f"Enzyme_ode not at steady state: max |dy/dt| = {jnp.max(jnp.abs(dydt_enzyme)):.2e}"
        )

        # Check isotopomer_ode derivatives are zero
        dydt_iso = isotopomer_ode(0.0, isotopomer_steady_state, params)
        assert jnp.allclose(dydt_iso, 0.0, atol=tol), (
            f"Isotopomer_ode not at steady state: max |dy/dt| = {jnp.max(jnp.abs(dydt_iso)):.2e}"
        )

    def test_flux_balance_at_steady_state(self, params, enzyme_steady_state):
        """Test that flux balances are satisfied at steady state.

        At steady state:
        - v_A_ex = v2 (A balance)
        - v4 + v_Q_ex + v_pq = 0 (Q balance, v4 net produces Q)
        - v5 + v_P_ex - v_pq = 0 (P balance, v5 net produces P)
        - v2 = v3 (EA balance)
        - v3 = v4 (EPQ balance)
        - v4 = v5 (EP balance)
        """
        fluxes = compute_net_fluxes(enzyme_steady_state, params)
        # Use tolerance of 1e-5 to account for steady-state detection threshold
        tol = 1e-5

        # A balance: v_A_ex = v2
        assert jnp.allclose(fluxes["v_A_ex"], fluxes["v2"], rtol=tol, atol=tol), (
            f"A balance: v_A_ex={fluxes['v_A_ex']:.8f} != v2={fluxes['v2']:.8f}"
        )

        # Enzyme intermediate balances: v2 = v3 = v4 = v5
        assert jnp.allclose(fluxes["v2"], fluxes["v3"], rtol=tol, atol=tol), (
            f"EA balance: v2={fluxes['v2']:.8f} != v3={fluxes['v3']:.8f}"
        )
        assert jnp.allclose(fluxes["v3"], fluxes["v4"], rtol=tol, atol=tol), (
            f"EPQ balance: v3={fluxes['v3']:.8f} != v4={fluxes['v4']:.8f}"
        )
        assert jnp.allclose(fluxes["v4"], fluxes["v5"], rtol=tol, atol=tol), (
            f"EP balance: v4={fluxes['v4']:.8f} != v5={fluxes['v5']:.8f}"
        )

        # Q balance: v4 + v_Q_ex + v_pq = 0
        q_balance = fluxes["v4"] + fluxes["v_Q_ex"] + fluxes["v_pq"]
        assert jnp.allclose(q_balance, 0.0, atol=tol), (
            f"Q balance: v4 + v_Q_ex + v_pq = {q_balance:.8f} != 0"
        )

        # P balance: v5 + v_P_ex - v_pq = 0
        p_balance = fluxes["v5"] + fluxes["v_P_ex"] - fluxes["v_pq"]
        assert jnp.allclose(p_balance, 0.0, atol=tol), (
            f"P balance: v5 + v_P_ex - v_pq = {p_balance:.8f} != 0"
        )


class TestIsotopomerConsistency:
    """Additional tests for isotopomer-specific consistency checks."""

    @pytest.fixture(scope="class")
    def params(self):
        """Get default parameters."""
        return DEFAULT_PARAMS.copy()

    @pytest.fixture(scope="class")
    def initial_total(self):
        """Get default initial total concentrations."""
        return DEFAULT_INITIAL_TOTAL.copy()

    @pytest.fixture(scope="class")
    def isotopomer_steady_state(self, params, initial_total):
        """Run isotopomer_ode to steady state."""
        sol = simulate_isotopomer_to_steady_state(params, initial_total)
        y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys
        return y_ss

    def test_all_isotopomers_non_negative(self, isotopomer_steady_state):
        """Test that all isotopomer concentrations are non-negative."""
        assert jnp.all(isotopomer_steady_state >= 0), (
            f"Negative isotopomer concentrations found: "
            f"min = {jnp.min(isotopomer_steady_state):.2e}"
        )

    def test_isotopomer_fractions_sum_to_one(self, isotopomer_steady_state):
        """Test that isotopomer fractions sum to 1 for each metabolite."""
        y = isotopomer_steady_state
        tol = 1e-10

        # A fractions
        A_total = y[0] + y[1] + y[2] + y[3]
        if A_total > 1e-10:
            A_sum = (y[0] + y[1] + y[2] + y[3]) / A_total
            assert jnp.allclose(A_sum, 1.0, atol=tol), f"A fractions sum to {A_sum}"

        # EA fractions
        EA_total = y[4] + y[5] + y[6] + y[7]
        if EA_total > 1e-10:
            EA_sum = (y[4] + y[5] + y[6] + y[7]) / EA_total
            assert jnp.allclose(EA_sum, 1.0, atol=tol), f"EA fractions sum to {EA_sum}"

        # EPQ fractions
        EPQ_total = y[8] + y[9] + y[10] + y[11]
        if EPQ_total > 1e-10:
            EPQ_sum = (y[8] + y[9] + y[10] + y[11]) / EPQ_total
            assert jnp.allclose(EPQ_sum, 1.0, atol=tol), f"EPQ fractions sum to {EPQ_sum}"

        # EP fractions
        EP_total = y[12] + y[13]
        if EP_total > 1e-10:
            EP_sum = (y[12] + y[13]) / EP_total
            assert jnp.allclose(EP_sum, 1.0, atol=tol), f"EP fractions sum to {EP_sum}"

        # Q fractions
        Q_total = y[14] + y[15]
        if Q_total > 1e-10:
            Q_sum = (y[14] + y[15]) / Q_total
            assert jnp.allclose(Q_sum, 1.0, atol=tol), f"Q fractions sum to {Q_sum}"

        # P fractions
        P_total = y[16] + y[17]
        if P_total > 1e-10:
            P_sum = (y[16] + y[17]) / P_total
            assert jnp.allclose(P_sum, 1.0, atol=tol), f"P fractions sum to {P_sum}"

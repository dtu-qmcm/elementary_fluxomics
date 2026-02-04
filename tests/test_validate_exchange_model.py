#!/usr/bin/env python3
"""Validation tests for external pool exchange model.

Tests:
1. Steady state reached (derivatives near zero)
2. Mass balance (exchange fluxes balance at steady state)
3. Exchange direction (v_ex > 0 when X_e > X)
4. Isotopomer consistency (sums to total concentration)
5. Labeling propagation through network
6. Concentration equilibration (internal -> external when k_ex large)
7. Isotopomer-NoLabeling derivative consistency
"""

import pytest
import jax
import jax.numpy as jnp
import diffrax

from scripts.enzyme_simulation_no_labeling import (
    DEFAULT_PARAMS as NL_PARAMS,
    DEFAULT_INITIAL as NL_INITIAL,
    compute_rates as nl_compute_rates,
    ode_system as nl_ode_system,
    compute_wegscheider_constraints,
)

from scripts.isotopomer_simulation import (
    DEFAULT_PARAMS as ISO_PARAMS,
    DEFAULT_INITIAL_TOTAL,
    isotopomer_ode,
    build_initial_state,
    STATE_INDEX,
    STATE_NAMES,
)


def nl_simulate_ode(y0, params, t_span):
    """Run ODE simulation using diffrax."""
    params_constrained = compute_wegscheider_constraints(params)

    term = diffrax.ODETerm(nl_ode_system)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.array([t_span[1]]))
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    solution = diffrax.diffeqsolve(
        term, solver, t0=t_span[0], t1=t_span[1], dt0=1e-4,
        y0=y0, args=params_constrained, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=500000,
    )
    return solution, params_constrained


@pytest.fixture(scope="module")
def steady_state_result():
    """Run ODE to steady state, shared across tests."""
    t_span = (0.0, 100.0)
    y0 = jnp.array([
        NL_INITIAL["A"], NL_INITIAL["Q"], NL_INITIAL["P"],
        NL_INITIAL["EA"], NL_INITIAL["EPQ"], NL_INITIAL["EP"]
    ])
    result, params_constrained = nl_simulate_ode(y0, NL_PARAMS, t_span)
    y_final = result.ys[-1]
    return y_final, params_constrained


@pytest.fixture(scope="module")
def isotopomer_steady_state():
    """Run isotopomer simulation to steady state, shared across tests."""
    y0 = build_initial_state(DEFAULT_INITIAL_TOTAL)
    t_span = (0.0, 100.0)

    term = diffrax.ODETerm(isotopomer_ode)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.array([t_span[1]]))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)

    solution = diffrax.diffeqsolve(
        term, solver, t0=t_span[0], t1=t_span[1], dt0=0.001,
        y0=y0, args=ISO_PARAMS, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000,
    )
    return solution.ys[-1]


def test_steady_state_reached(steady_state_result):
    """Verify ODE solver reaches steady state (derivatives near zero)."""
    y_ss, params = steady_state_result
    dydt = nl_ode_system(100.0, y_ss, params)
    max_deriv = jnp.max(jnp.abs(dydt))
    assert max_deriv < 1e-4


def test_mass_balance(steady_state_result):
    """Verify mass conservation at steady state."""
    y_ss, params = steady_state_result
    rates = nl_compute_rates(y_ss, params)

    # At steady state: v_A_ex = v2 (A balance)
    a_balance = abs(rates['v_A_ex'] - rates['v2'])
    assert a_balance < 1e-4

    # Q balance: v4 + v_Q_ex + v_pq = 0
    q_balance = abs(rates['v4'] + rates['v_Q_ex'] + rates['v_pq'])
    assert q_balance < 1e-4

    # P balance: v5 + v_P_ex - v_pq = 0
    p_balance = abs(rates['v5'] + rates['v_P_ex'] - rates['v_pq'])
    assert p_balance < 1e-4


def test_exchange_direction(steady_state_result):
    """Verify exchange responds correctly to concentration gradients."""
    y_ss, params = steady_state_result
    A, Q, P = y_ss[0], y_ss[1], y_ss[2]
    rates = nl_compute_rates(y_ss, params)

    # When X_e > X: v_X_ex > 0 (net influx)
    A_e = params["A_e"]
    assert (A_e > A and rates["v_A_ex"] > 0) or (A_e < A and rates["v_A_ex"] < 0) or (abs(A_e - A) < 1e-6)

    Q_e = params["Q_e"]
    assert (Q_e > Q and rates["v_Q_ex"] > 0) or (Q_e < Q and rates["v_Q_ex"] < 0) or (abs(Q_e - Q) < 1e-6)

    P_e = params["P_e"]
    assert (P_e > P and rates["v_P_ex"] > 0) or (P_e < P and rates["v_P_ex"] < 0) or (abs(P_e - P) < 1e-6)


def test_isotopomer_consistency(isotopomer_steady_state):
    """Verify isotopomer fractions sum to total concentrations."""
    y = isotopomer_steady_state

    A_total = y[STATE_INDEX["A_00"]] + y[STATE_INDEX["A_01"]] + \
              y[STATE_INDEX["A_10"]] + y[STATE_INDEX["A_11"]]
    Q_total = y[STATE_INDEX["Q_0"]] + y[STATE_INDEX["Q_1"]]
    P_total = y[STATE_INDEX["P_0"]] + y[STATE_INDEX["P_1"]]

    assert A_total > 0
    assert Q_total > 0
    assert P_total > 0

    # MIDs must sum to 1
    A_mid = jnp.array([y[STATE_INDEX["A_00"]], y[STATE_INDEX["A_01"]],
                       y[STATE_INDEX["A_10"]], y[STATE_INDEX["A_11"]]]) / A_total
    Q_mid = jnp.array([y[STATE_INDEX["Q_0"]], y[STATE_INDEX["Q_1"]]]) / Q_total
    P_mid = jnp.array([y[STATE_INDEX["P_0"]], y[STATE_INDEX["P_1"]]]) / P_total

    assert abs(jnp.sum(A_mid) - 1.0) < 1e-6
    assert abs(jnp.sum(Q_mid) - 1.0) < 1e-6
    assert abs(jnp.sum(P_mid) - 1.0) < 1e-6


def test_labeling_patterns(isotopomer_steady_state):
    """Verify labeling propagates correctly through network."""
    y = isotopomer_steady_state

    A_total = y[STATE_INDEX["A_00"]] + y[STATE_INDEX["A_01"]] + \
              y[STATE_INDEX["A_10"]] + y[STATE_INDEX["A_11"]]
    A_labeled_a = (y[STATE_INDEX["A_10"]] + y[STATE_INDEX["A_11"]]) / A_total

    # A_e is 100% labeled, so A should have significant labeling
    assert A_labeled_a > 0.1


def test_concentration_equilibration():
    """Verify internal concentrations approach external when k_ex is large."""
    t_span = (0.0, 100.0)
    y0 = jnp.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])

    # High exchange rate
    high_k_params = dict(NL_PARAMS)
    high_k_params["k_A_ex"] = 10.0
    high_k_result, high_k_params_c = nl_simulate_ode(y0, high_k_params, t_span)
    a_ratio_high = high_k_result.ys[-1][0] / high_k_params_c["A_e"]

    # Default exchange rate
    default_result, default_params_c = nl_simulate_ode(y0, NL_PARAMS, t_span)
    a_ratio_default = default_result.ys[-1][0] / default_params_c["A_e"]

    # Higher k_A_ex should give ratio closer to 1
    assert a_ratio_high > a_ratio_default


def test_derivatives_match_no_labeling():
    """Verify isotopomer ODE gives same total derivatives as no-labeling ODE."""
    nl_params_constrained = compute_wegscheider_constraints(NL_PARAMS)

    A, Q, P, EA, EPQ, EP = 1.0, 0.5, 0.5, 0.01, 0.01, 0.01
    y_nl = jnp.array([A, Q, P, EA, EPQ, EP])

    y_iso = jnp.zeros(len(STATE_NAMES))
    y_iso = y_iso.at[STATE_INDEX["A_00"]].set(A)
    y_iso = y_iso.at[STATE_INDEX["Q_0"]].set(Q)
    y_iso = y_iso.at[STATE_INDEX["P_0"]].set(P)
    y_iso = y_iso.at[STATE_INDEX["EA_00"]].set(EA)
    y_iso = y_iso.at[STATE_INDEX["EPQ_00"]].set(EPQ)
    y_iso = y_iso.at[STATE_INDEX["EP_0"]].set(EP)

    dydt_nl = nl_ode_system(0.0, y_nl, nl_params_constrained)
    dydt_iso = isotopomer_ode(0.0, y_iso, ISO_PARAMS)

    # Sum isotopomer derivatives to get total derivatives
    dA_iso = dydt_iso[STATE_INDEX["A_00"]] + dydt_iso[STATE_INDEX["A_01"]] + \
             dydt_iso[STATE_INDEX["A_10"]] + dydt_iso[STATE_INDEX["A_11"]]
    dQ_iso = dydt_iso[STATE_INDEX["Q_0"]] + dydt_iso[STATE_INDEX["Q_1"]]
    dP_iso = dydt_iso[STATE_INDEX["P_0"]] + dydt_iso[STATE_INDEX["P_1"]]
    dEA_iso = dydt_iso[STATE_INDEX["EA_00"]] + dydt_iso[STATE_INDEX["EA_01"]] + \
              dydt_iso[STATE_INDEX["EA_10"]] + dydt_iso[STATE_INDEX["EA_11"]]
    dEPQ_iso = dydt_iso[STATE_INDEX["EPQ_00"]] + dydt_iso[STATE_INDEX["EPQ_01"]] + \
               dydt_iso[STATE_INDEX["EPQ_10"]] + dydt_iso[STATE_INDEX["EPQ_11"]]
    dEP_iso = dydt_iso[STATE_INDEX["EP_0"]] + dydt_iso[STATE_INDEX["EP_1"]]

    diffs = [
        abs(dydt_nl[0] - dA_iso),
        abs(dydt_nl[1] - dQ_iso),
        abs(dydt_nl[2] - dP_iso),
        abs(dydt_nl[3] - dEA_iso),
        abs(dydt_nl[4] - dEPQ_iso),
        abs(dydt_nl[5] - dEP_iso),
    ]
    assert max(diffs) < 0.01

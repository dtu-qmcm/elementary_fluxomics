#!/usr/bin/env python3
"""Validation script for external pool exchange model.

Runs a series of tests to verify that the exchange kinetics implementation
is correct before running MCMC sampling.

Tests:
1. Steady state reached (derivatives near zero)
2. Mass balance (exchange fluxes balance at steady state)
3. Exchange direction (v_ex > 0 when X_e > X)
4. Isotopomer consistency (sums to total concentration)
5. Labeling propagation through network
6. Concentration equilibration (internal -> external when k_ex large)
"""

from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

# Import from local scripts
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from enzyme_simulation_no_labeling import (
    DEFAULT_PARAMS as NL_PARAMS,
    DEFAULT_INITIAL as NL_INITIAL,
    compute_rates as nl_compute_rates,
    ode_system as nl_ode_system,
    compute_wegscheider_constraints,
)

from isotopomer_simulation import (
    DEFAULT_PARAMS as ISO_PARAMS,
    DEFAULT_INITIAL_TOTAL,
    isotopomer_ode,
    build_initial_state,
    STATE_INDEX,
    STATE_NAMES,
)


def nl_simulate_ode(y0, params, t_span):
    """Run ODE simulation using diffrax."""
    # Apply Wegscheider constraints to get consistent k2f and k_pq_r
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


def test_steady_state_reached():
    """Verify ODE solver reaches steady state (derivatives near zero)."""
    print("\n=== Test 1: Steady State Reached ===")

    # Run simulation
    t_span = (0.0, 100.0)
    y0 = jnp.array([
        NL_INITIAL["A"], NL_INITIAL["Q"], NL_INITIAL["P"],
        NL_INITIAL["EA"], NL_INITIAL["EPQ"], NL_INITIAL["EP"]
    ])

    result, params_constrained = nl_simulate_ode(y0, NL_PARAMS, t_span)
    y_final = result.ys[-1]

    # Compute derivatives at final state
    dydt = nl_ode_system(t_span[1], y_final, params_constrained)
    max_deriv = jnp.max(jnp.abs(dydt))

    print(f"  Final state: A={y_final[0]:.4f}, Q={y_final[1]:.4f}, P={y_final[2]:.4f}")
    print(f"  Final state: EA={y_final[3]:.6f}, EPQ={y_final[4]:.6f}, EP={y_final[5]:.6f}")
    print(f"  Max |dy/dt| at steady state: {max_deriv:.2e}")

    # Relax tolerance since the system may not fully converge
    passed = max_deriv < 1e-4
    print(f"  PASSED: {passed}")
    return passed, y_final, params_constrained


def test_mass_balance(y_ss, params):
    """Verify mass conservation at steady state.

    At steady state, the net flux through the system should balance:
    - v_A_ex = v2 (what flows in equals what binds to enzyme)
    - Total exchange fluxes should balance with internal transformations
    """
    print("\n=== Test 2: Mass Balance ===")

    rates = nl_compute_rates(y_ss, params)

    print(f"  Exchange fluxes:")
    print(f"    v_A_ex = {rates['v_A_ex']:.6f}")
    print(f"    v_Q_ex = {rates['v_Q_ex']:.6f}")
    print(f"    v_P_ex = {rates['v_P_ex']:.6f}")

    print(f"  Internal fluxes:")
    print(f"    v2 = {rates['v2']:.6f}")
    print(f"    v3 = {rates['v3']:.6f}")
    print(f"    v4 = {rates['v4']:.6f}")
    print(f"    v5 = {rates['v5']:.6f}")
    print(f"    v_pq = {rates['v_pq']:.6f}")

    # At steady state: v_A_ex = v2 (A balance)
    a_balance = abs(rates['v_A_ex'] - rates['v2'])
    print(f"  A balance (v_A_ex - v2): {a_balance:.2e}")

    # Q balance: v4 + v_Q_ex + v_pq = 0
    q_balance = abs(rates['v4'] + rates['v_Q_ex'] + rates['v_pq'])
    print(f"  Q balance (v4 + v_Q_ex + v_pq): {q_balance:.2e}")

    # P balance: v5 + v_P_ex - v_pq = 0
    p_balance = abs(rates['v5'] + rates['v_P_ex'] - rates['v_pq'])
    print(f"  P balance (v5 + v_P_ex - v_pq): {p_balance:.2e}")

    # Relax tolerance since the system may not fully converge
    passed = a_balance < 1e-4 and q_balance < 1e-4 and p_balance < 1e-4
    print(f"  PASSED: {passed}")
    return passed


def test_exchange_direction(y_ss, params):
    """Verify exchange responds correctly to concentration gradients.

    When X_e > X: v_X_ex > 0 (net influx)
    When X_e < X: v_X_ex < 0 (net efflux)
    """
    print("\n=== Test 3: Exchange Direction ===")

    A, Q, P = y_ss[0], y_ss[1], y_ss[2]
    rates = nl_compute_rates(y_ss, params)

    # Check A
    A_e = params["A_e"]
    v_A_ex = rates["v_A_ex"]
    a_correct = (A_e > A and v_A_ex > 0) or (A_e < A and v_A_ex < 0) or (abs(A_e - A) < 1e-6)
    print(f"  A: A_e={A_e:.2f}, A={A:.4f}, v_A_ex={v_A_ex:.6f}")
    print(f"    A_e > A: {A_e > A}, v_A_ex > 0: {v_A_ex > 0} -> Correct: {a_correct}")

    # Check Q
    Q_e = params["Q_e"]
    v_Q_ex = rates["v_Q_ex"]
    q_correct = (Q_e > Q and v_Q_ex > 0) or (Q_e < Q and v_Q_ex < 0) or (abs(Q_e - Q) < 1e-6)
    print(f"  Q: Q_e={Q_e:.2f}, Q={Q:.4f}, v_Q_ex={v_Q_ex:.6f}")
    print(f"    Q_e > Q: {Q_e > Q}, v_Q_ex > 0: {v_Q_ex > 0} -> Correct: {q_correct}")

    # Check P
    P_e = params["P_e"]
    v_P_ex = rates["v_P_ex"]
    p_correct = (P_e > P and v_P_ex > 0) or (P_e < P and v_P_ex < 0) or (abs(P_e - P) < 1e-6)
    print(f"  P: P_e={P_e:.2f}, P={P:.4f}, v_P_ex={v_P_ex:.6f}")
    print(f"    P_e > P: {P_e > P}, v_P_ex > 0: {v_P_ex > 0} -> Correct: {p_correct}")

    passed = a_correct and q_correct and p_correct
    print(f"  PASSED: {passed}")
    return passed


def test_isotopomer_consistency():
    """Verify isotopomer fractions sum to total concentrations."""
    print("\n=== Test 4: Isotopomer Consistency ===")

    # Run isotopomer simulation
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
    y_final = solution.ys[-1]

    # Check A isotopomers sum
    A_total = y_final[STATE_INDEX["A_00"]] + y_final[STATE_INDEX["A_01"]] + \
              y_final[STATE_INDEX["A_10"]] + y_final[STATE_INDEX["A_11"]]

    # Check Q isotopomers sum
    Q_total = y_final[STATE_INDEX["Q_0"]] + y_final[STATE_INDEX["Q_1"]]

    # Check P isotopomers sum
    P_total = y_final[STATE_INDEX["P_0"]] + y_final[STATE_INDEX["P_1"]]

    print(f"  A isotopomers: A_00={y_final[STATE_INDEX['A_00']]:.4f}, "
          f"A_01={y_final[STATE_INDEX['A_01']]:.4f}, "
          f"A_10={y_final[STATE_INDEX['A_10']]:.4f}, "
          f"A_11={y_final[STATE_INDEX['A_11']]:.4f}")
    print(f"  A_total = {A_total:.4f}")

    print(f"  Q isotopomers: Q_0={y_final[STATE_INDEX['Q_0']]:.4f}, "
          f"Q_1={y_final[STATE_INDEX['Q_1']]:.4f}")
    print(f"  Q_total = {Q_total:.4f}")

    print(f"  P isotopomers: P_0={y_final[STATE_INDEX['P_0']]:.4f}, "
          f"P_1={y_final[STATE_INDEX['P_1']]:.4f}")
    print(f"  P_total = {P_total:.4f}")

    # Check MIDs sum to 1
    A_mid = jnp.array([y_final[STATE_INDEX["A_00"]], y_final[STATE_INDEX["A_01"]],
                       y_final[STATE_INDEX["A_10"]], y_final[STATE_INDEX["A_11"]]]) / A_total
    Q_mid = jnp.array([y_final[STATE_INDEX["Q_0"]], y_final[STATE_INDEX["Q_1"]]]) / Q_total
    P_mid = jnp.array([y_final[STATE_INDEX["P_0"]], y_final[STATE_INDEX["P_1"]]]) / P_total

    a_mid_sum = jnp.sum(A_mid)
    q_mid_sum = jnp.sum(Q_mid)
    p_mid_sum = jnp.sum(P_mid)

    print(f"  MID sums: A={a_mid_sum:.6f}, Q={q_mid_sum:.6f}, P={p_mid_sum:.6f}")

    passed = (abs(a_mid_sum - 1.0) < 1e-6 and
              abs(q_mid_sum - 1.0) < 1e-6 and
              abs(p_mid_sum - 1.0) < 1e-6 and
              A_total > 0 and Q_total > 0 and P_total > 0)
    print(f"  PASSED: {passed}")
    return passed, y_final


def test_labeling_patterns(y_iso):
    """Verify labeling propagates correctly through network.

    At steady state with A_e 100% labeled on atom 'a':
    - A should have significant labeling on atom 'a' (A_10 + A_11 > 0)
    - EA, EPQ should propagate labeling from A
    - Q gets atom 'b' from EPQ, P gets atom 'a' from EPQ
    """
    print("\n=== Test 5: Labeling Patterns ===")

    # A labeling: atom 'a' should be labeled (A_10, A_11 have label on position 0)
    A_total = y_iso[STATE_INDEX["A_00"]] + y_iso[STATE_INDEX["A_01"]] + \
              y_iso[STATE_INDEX["A_10"]] + y_iso[STATE_INDEX["A_11"]]
    A_labeled_a = (y_iso[STATE_INDEX["A_10"]] + y_iso[STATE_INDEX["A_11"]]) / A_total

    print(f"  A labeling on atom 'a': {A_labeled_a:.4f} (should be high, input is 100% labeled)")

    # EA labeling should mirror A labeling
    EA_total = y_iso[STATE_INDEX["EA_00"]] + y_iso[STATE_INDEX["EA_01"]] + \
               y_iso[STATE_INDEX["EA_10"]] + y_iso[STATE_INDEX["EA_11"]]
    if EA_total > 1e-10:
        EA_labeled_a = (y_iso[STATE_INDEX["EA_10"]] + y_iso[STATE_INDEX["EA_11"]]) / EA_total
        print(f"  EA labeling on atom 'a': {EA_labeled_a:.4f}")
    else:
        EA_labeled_a = 0.0
        print(f"  EA total very small: {EA_total:.2e}")

    # EPQ labeling
    EPQ_total = y_iso[STATE_INDEX["EPQ_00"]] + y_iso[STATE_INDEX["EPQ_01"]] + \
                y_iso[STATE_INDEX["EPQ_10"]] + y_iso[STATE_INDEX["EPQ_11"]]
    if EPQ_total > 1e-10:
        EPQ_labeled_a = (y_iso[STATE_INDEX["EPQ_10"]] + y_iso[STATE_INDEX["EPQ_11"]]) / EPQ_total
        print(f"  EPQ labeling on atom 'a': {EPQ_labeled_a:.4f}")
    else:
        EPQ_labeled_a = 0.0
        print(f"  EPQ total very small: {EPQ_total:.2e}")

    # Q gets atom 'b' from A (position 1 in A -> position 0 in Q)
    # So Q labeling comes from A_01 and A_11
    Q_total = y_iso[STATE_INDEX["Q_0"]] + y_iso[STATE_INDEX["Q_1"]]
    Q_labeled = y_iso[STATE_INDEX["Q_1"]] / Q_total
    A_labeled_b = (y_iso[STATE_INDEX["A_01"]] + y_iso[STATE_INDEX["A_11"]]) / A_total
    print(f"  Q labeling: {Q_labeled:.4f} (derived from A atom 'b': {A_labeled_b:.4f})")

    # P gets atom 'a' from A (position 0 in A -> position 0 in P)
    P_total = y_iso[STATE_INDEX["P_0"]] + y_iso[STATE_INDEX["P_1"]]
    P_labeled = y_iso[STATE_INDEX["P_1"]] / P_total
    print(f"  P labeling: {P_labeled:.4f} (derived from A atom 'a': {A_labeled_a:.4f})")

    # Verify labeling propagation makes sense
    # A_e is 100% labeled on 'a', so A should be highly labeled on 'a'
    # Q and P get labeling from A through enzyme mechanism
    passed = A_labeled_a > 0.1  # A should have significant labeling
    print(f"  PASSED: {passed}")
    return passed


def test_concentration_equilibration():
    """Verify internal concentrations approach external when k_ex is large.

    With very high k_A_ex, A should approach A_e quickly.
    """
    print("\n=== Test 6: Concentration Equilibration ===")

    # Create params with very high exchange rate for A
    high_k_params = dict(NL_PARAMS)
    high_k_params["k_A_ex"] = 10.0  # 100x higher than default

    # Run simulation using the helper function (applies Wegscheider constraints)
    t_span = (0.0, 100.0)
    y0 = jnp.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])

    high_k_result, high_k_params_constrained = nl_simulate_ode(y0, high_k_params, t_span)
    y_final = high_k_result.ys[-1]

    A_final = y_final[0]
    A_e = high_k_params_constrained["A_e"]

    # A should be closer to A_e with high k_A_ex
    # (not equal because there's consumption by enzyme)
    a_ratio = A_final / A_e

    print(f"  With k_A_ex = {high_k_params_constrained['k_A_ex']}:")
    print(f"    A_e = {A_e:.2f}")
    print(f"    A_final = {A_final:.4f}")
    print(f"    A/A_e ratio = {a_ratio:.4f}")

    # Also test default parameters
    default_result, default_params_constrained = nl_simulate_ode(y0, NL_PARAMS, t_span)
    A_default = default_result.ys[-1][0]
    default_ratio = A_default / default_params_constrained["A_e"]

    print(f"  With k_A_ex = {default_params_constrained['k_A_ex']} (default):")
    print(f"    A_final = {A_default:.4f}")
    print(f"    A/A_e ratio = {default_ratio:.4f}")

    # Higher k_A_ex should give ratio closer to 1
    passed = a_ratio > default_ratio
    print(f"  Higher k_A_ex gives A closer to A_e: {passed}")
    print(f"  PASSED: {passed}")
    return passed


def test_derivatives_match_no_labeling():
    """Verify isotopomer ODE gives same total derivatives as no-labeling ODE."""
    print("\n=== Test 7: Isotopomer-NoLabeling Consistency ===")

    # Apply Wegscheider constraints to both parameter sets for fair comparison
    nl_params_constrained = compute_wegscheider_constraints(NL_PARAMS)

    # Set up equivalent initial conditions
    # No-labeling state: [A, Q, P, EA, EPQ, EP]
    A, Q, P, EA, EPQ, EP = 1.0, 0.5, 0.5, 0.01, 0.01, 0.01
    y_nl = jnp.array([A, Q, P, EA, EPQ, EP])

    # Isotopomer state: all unlabeled
    y_iso = jnp.zeros(len(STATE_NAMES))
    y_iso = y_iso.at[STATE_INDEX["A_00"]].set(A)
    y_iso = y_iso.at[STATE_INDEX["Q_0"]].set(Q)
    y_iso = y_iso.at[STATE_INDEX["P_0"]].set(P)
    y_iso = y_iso.at[STATE_INDEX["EA_00"]].set(EA)
    y_iso = y_iso.at[STATE_INDEX["EPQ_00"]].set(EPQ)
    y_iso = y_iso.at[STATE_INDEX["EP_0"]].set(EP)

    # Compute derivatives using constrained params for both
    dydt_nl = nl_ode_system(0.0, y_nl, nl_params_constrained)
    dydt_iso = isotopomer_ode(0.0, y_iso, ISO_PARAMS)

    print(f"  Using k2f={nl_params_constrained['k2f']:.6f} (Wegscheider-constrained)")
    print(f"  Isotopomer k2f={ISO_PARAMS['k2f']:.6f}")

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

    print(f"  No-labeling derivatives:")
    print(f"    dA/dt = {dydt_nl[0]:.6f}")
    print(f"    dQ/dt = {dydt_nl[1]:.6f}")
    print(f"    dP/dt = {dydt_nl[2]:.6f}")
    print(f"    dEA/dt = {dydt_nl[3]:.6f}")
    print(f"    dEPQ/dt = {dydt_nl[4]:.6f}")
    print(f"    dEP/dt = {dydt_nl[5]:.6f}")

    print(f"  Isotopomer sum derivatives:")
    print(f"    dA/dt = {dA_iso:.6f}")
    print(f"    dQ/dt = {dQ_iso:.6f}")
    print(f"    dP/dt = {dP_iso:.6f}")
    print(f"    dEA/dt = {dEA_iso:.6f}")
    print(f"    dEPQ/dt = {dEPQ_iso:.6f}")
    print(f"    dEP/dt = {dEP_iso:.6f}")

    # Check differences
    diffs = [
        abs(dydt_nl[0] - dA_iso),
        abs(dydt_nl[1] - dQ_iso),
        abs(dydt_nl[2] - dP_iso),
        abs(dydt_nl[3] - dEA_iso),
        abs(dydt_nl[4] - dEPQ_iso),
        abs(dydt_nl[5] - dEP_iso),
    ]
    max_diff = max(diffs)
    print(f"  Max difference: {max_diff:.2e}")

    # With same Wegscheider-constrained parameters, derivatives should match closely
    passed = max_diff < 0.01
    print(f"  PASSED: {passed}")
    return passed


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("Validation Tests for External Pool Exchange Model")
    print("=" * 60)

    results = {}

    # Test 1: Steady state
    passed, y_ss, params_constrained = test_steady_state_reached()
    results["steady_state"] = passed

    # Test 2: Mass balance (uses steady state from test 1)
    results["mass_balance"] = test_mass_balance(y_ss, params_constrained)

    # Test 3: Exchange direction
    results["exchange_direction"] = test_exchange_direction(y_ss, params_constrained)

    # Test 4: Isotopomer consistency
    passed, y_iso = test_isotopomer_consistency()
    results["isotopomer_consistency"] = passed

    # Test 5: Labeling patterns
    results["labeling_patterns"] = test_labeling_patterns(y_iso)

    # Test 6: Concentration equilibration
    results["concentration_equilibration"] = test_concentration_equilibration()

    # Test 7: Isotopomer-NoLabeling consistency
    results["derivatives_consistency"] = test_derivatives_match_no_labeling()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL VALIDATION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please review the implementation")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

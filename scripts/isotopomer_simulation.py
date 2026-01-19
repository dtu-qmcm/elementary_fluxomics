#!/usr/bin/env python3
"""Isotopomer simulation using separate labelled/unlabelled species.

Simulates the enzyme mechanism by tracking each isotopomer as a separate
species. Reaction rates use the same kinetic parameters - isotope labels
don't affect kinetics. MID fractions are computed by normalizing.

Labeling pattern:
  - A_e (external A): 100% labeled on both atoms (isotopomer A_11)
  - Q_e, P_e (external Q, P): 100% unlabeled

This module provides two ODE functions:
  - enzyme_ode: Total concentrations only (no isotopomers) for steady-state
  - isotopomer_ode: Full isotopomer tracking for labeling dynamics
"""

from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

try:
    from scripts.isotopomer_network import STATE_NAMES, STATE_INDEX
except ModuleNotFoundError:
    from isotopomer_network import STATE_NAMES, STATE_INDEX


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "isotopomer_simulation"

# Default kinetic parameters (same as enzyme_simulation.py)
DEFAULT_PARAMS = {
    # External pool concentrations (fixed boundary conditions)
    "A_e": 5.0,          # external A concentration
    "Q_e": 1.0,          # external Q concentration
    "P_e": 1.0,          # external P concentration
    # Exchange rate constants (symmetric mass-action: v = k * (X_e - X))
    "k_A_ex": 0.1,       # A exchange rate constant
    "k_Q_ex": 0.1,       # Q exchange rate constant
    "k_P_ex": 0.1,       # P exchange rate constant
    # Enzyme kinetic rate constants
    "k2f": 0.560202,     # Computed from Wegscheider constraint
    "k2r": 250.0,
    "k3f": 125.0,
    "k3r": 12.5,
    "k4f": 200.0,
    "k4r": 20.0,
    "k5f": 150.0,
    "k5r": 15.0,
    "k_pq_f": 50.0,
    "k_pq_r": 14.906082,  # Computed from Wegscheider constraint
    "E_T": 1.0,
}

# Initial total concentrations (all unlabelled at t=0)
DEFAULT_INITIAL_TOTAL = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
}


def build_initial_state(total_concs):
    """Build initial isotopomer state from total concentrations.

    All metabolites start completely unlabelled.
    """
    y0 = jnp.zeros(len(STATE_NAMES))

    # A starts unlabelled (A_00)
    y0 = y0.at[STATE_INDEX["A_00"]].set(total_concs["A"])

    # EA starts unlabelled (EA_00)
    y0 = y0.at[STATE_INDEX["EA_00"]].set(total_concs["EA"])

    # EPQ starts unlabelled (EPQ_00)
    y0 = y0.at[STATE_INDEX["EPQ_00"]].set(total_concs["EPQ"])

    # EP starts unlabelled (EP_0)
    y0 = y0.at[STATE_INDEX["EP_0"]].set(total_concs["EP"])

    # Q starts unlabelled (Q_0)
    y0 = y0.at[STATE_INDEX["Q_0"]].set(total_concs["Q"])

    # P starts unlabelled (P_0)
    y0 = y0.at[STATE_INDEX["P_0"]].set(total_concs["P"])

    return y0


def enzyme_ode(t, y, params):
    """ODE system for total concentrations (no isotope labeling).

    This function solves for steady-state fluxes and concentrations without
    tracking individual isotopomers. Use this to find metabolic steady state,
    then use isotopomer_ode for labeling dynamics.

    State variables: [A, Q, P, EA, EPQ, EP]

    Mass balances:
    - A: exchange with external pool, consumed by enzyme binding
    - Q: produced by v4, exchange with external pool, interconversion with P
    - P: produced by v5, exchange with external pool, interconversion with Q
    - EA, EPQ, EP: enzyme intermediates

    Enzyme conservation: E = E_T - EA - EPQ - EP
    """
    A, Q, P, EA, EPQ, EP = y[0], y[1], y[2], y[3], y[4], y[5]

    # Free enzyme from conservation
    E = params["E_T"] - EA - EPQ - EP

    # Extract parameters
    A_e = params["A_e"]
    Q_e = params["Q_e"]
    P_e = params["P_e"]
    k_A_ex = params["k_A_ex"]
    k_Q_ex = params["k_Q_ex"]
    k_P_ex = params["k_P_ex"]
    k2f, k2r = params["k2f"], params["k2r"]
    k3f, k3r = params["k3f"], params["k3r"]
    k4f, k4r = params["k4f"], params["k4r"]
    k5f, k5r = params["k5f"], params["k5r"]
    k_pq_f, k_pq_r = params["k_pq_f"], params["k_pq_r"]

    # Exchange reactions with external pools
    v_A_ex = k_A_ex * (A_e - A)
    v_Q_ex = k_Q_ex * (Q_e - Q)
    v_P_ex = k_P_ex * (P_e - P)

    # Enzyme-catalyzed reactions
    v2 = A * E * k2f - EA * k2r
    v3 = EA * k3f - EPQ * k3r
    v4 = EPQ * k4f - Q * EP * k4r
    v5 = EP * k5f - E * P * k5r

    # P <-> Q interconversion
    v_pq = P * k_pq_f - Q * k_pq_r

    # Mass balances
    dA_dt = v_A_ex - v2
    dQ_dt = v4 + v_Q_ex + v_pq
    dP_dt = v5 + v_P_ex - v_pq
    dEA_dt = v2 - v3
    dEPQ_dt = v3 - v4
    dEP_dt = v4 - v5

    return jnp.array([dA_dt, dQ_dt, dP_dt, dEA_dt, dEPQ_dt, dEP_dt])


def isotopomer_ode(t, y, params):
    """ODE system for isotopomer dynamics with external pool exchange kinetics.

    State vector y has 18 components:
      A_00, A_01, A_10, A_11,     (indices 0-3)
      EA_00, EA_01, EA_10, EA_11, (indices 4-7)
      EPQ_00, EPQ_01, EPQ_10, EPQ_11, (indices 8-11)
      EP_0, EP_1,                 (indices 12-13)
      Q_0, Q_1,                   (indices 14-15)
      P_0, P_1                    (indices 16-17)

    Exchange reactions use symmetric mass-action kinetics:
      v_X_ex = k_X_ex * (X_e - X)
    where X_e is the fixed external concentration.

    External labeling patterns:
      - A_e: 100% labeled on both atoms (isotopomer A_11)
      - Q_e: 100% unlabeled (isotopomer Q_0)
      - P_e: 100% unlabeled (isotopomer P_0)
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

    # Free enzyme from conservation
    E = params["E_T"] - EA - EPQ - EP

    # Extract parameters
    A_e = params["A_e"]
    Q_e = params["Q_e"]
    P_e = params["P_e"]
    k_A_ex = params["k_A_ex"]
    k_Q_ex = params["k_Q_ex"]
    k_P_ex = params["k_P_ex"]
    k2f, k2r = params["k2f"], params["k2r"]
    k3f, k3r = params["k3f"], params["k3r"]
    k4f, k4r = params["k4f"], params["k4r"]
    k5f, k5r = params["k5f"], params["k5r"]
    k_pq_f, k_pq_r = params["k_pq_f"], params["k_pq_r"]

    # =========================================================================
    # v_A_ex: A_e <-> A (exchange with external A pool)
    # A_e is 100% labeled on both atoms (isotopomer A_11)
    # Net flux: v_A_ex = k_A_ex * (A_e - A)
    # - When v_A_ex > 0 (net influx): add to A_11 only
    # - When v_A_ex < 0 (net efflux): remove proportionally from all isotopomers
    # =========================================================================
    v_A_ex = k_A_ex * (A_e - A)
    # Avoid division by zero
    A_safe = jnp.maximum(A, 1e-10)
    # Influx goes to A_11 (fully labeled), efflux is proportional
    dA_00_v1 = jnp.where(v_A_ex >= 0, 0.0, v_A_ex * A_00 / A_safe)
    dA_01_v1 = jnp.where(v_A_ex >= 0, 0.0, v_A_ex * A_01 / A_safe)
    dA_10_v1 = jnp.where(v_A_ex >= 0, 0.0, v_A_ex * A_10 / A_safe)
    dA_11_v1 = jnp.where(v_A_ex >= 0, v_A_ex, v_A_ex * A_11 / A_safe)

    # =========================================================================
    # v2: A + E <-> EA (atoms preserved)
    # =========================================================================
    # Forward: A_xx + E -> EA_xx (rate = k2f * A_xx * E)
    # Backward: EA_xx -> A_xx + E (rate = k2r * EA_xx)

    v2f_00 = k2f * A_00 * E
    v2f_01 = k2f * A_01 * E
    v2f_10 = k2f * A_10 * E
    v2f_11 = k2f * A_11 * E

    v2b_00 = k2r * EA_00
    v2b_01 = k2r * EA_01
    v2b_10 = k2r * EA_10
    v2b_11 = k2r * EA_11

    dA_00_v2 = -v2f_00 + v2b_00
    dA_01_v2 = -v2f_01 + v2b_01
    dA_10_v2 = -v2f_10 + v2b_10
    dA_11_v2 = -v2f_11 + v2b_11

    dEA_00_v2 = v2f_00 - v2b_00
    dEA_01_v2 = v2f_01 - v2b_01
    dEA_10_v2 = v2f_10 - v2b_10
    dEA_11_v2 = v2f_11 - v2b_11

    # =========================================================================
    # v3: EA <-> EPQ (atoms preserved)
    # =========================================================================
    v3f_00 = k3f * EA_00
    v3f_01 = k3f * EA_01
    v3f_10 = k3f * EA_10
    v3f_11 = k3f * EA_11

    v3b_00 = k3r * EPQ_00
    v3b_01 = k3r * EPQ_01
    v3b_10 = k3r * EPQ_10
    v3b_11 = k3r * EPQ_11

    dEA_00_v3 = -v3f_00 + v3b_00
    dEA_01_v3 = -v3f_01 + v3b_01
    dEA_10_v3 = -v3f_10 + v3b_10
    dEA_11_v3 = -v3f_11 + v3b_11

    dEPQ_00_v3 = v3f_00 - v3b_00
    dEPQ_01_v3 = v3f_01 - v3b_01
    dEPQ_10_v3 = v3f_10 - v3b_10
    dEPQ_11_v3 = v3f_11 - v3b_11

    # =========================================================================
    # v4: EPQ <-> EP + Q (EPQ_ab -> EP_a + Q_b)
    # =========================================================================
    # Forward: EPQ_ab -> EP_a + Q_b (rate = k4f * EPQ_ab)
    # Backward: EP_a + Q_b -> EPQ_ab (rate = k4r * EP_a * Q_b)

    # Forward rates
    v4f_00 = k4f * EPQ_00  # -> EP_0 + Q_0
    v4f_01 = k4f * EPQ_01  # -> EP_0 + Q_1
    v4f_10 = k4f * EPQ_10  # -> EP_1 + Q_0
    v4f_11 = k4f * EPQ_11  # -> EP_1 + Q_1

    # Backward rates
    v4b_00 = k4r * EP_0 * Q_0  # -> EPQ_00
    v4b_01 = k4r * EP_0 * Q_1  # -> EPQ_01
    v4b_10 = k4r * EP_1 * Q_0  # -> EPQ_10
    v4b_11 = k4r * EP_1 * Q_1  # -> EPQ_11

    dEPQ_00_v4 = -v4f_00 + v4b_00
    dEPQ_01_v4 = -v4f_01 + v4b_01
    dEPQ_10_v4 = -v4f_10 + v4b_10
    dEPQ_11_v4 = -v4f_11 + v4b_11

    # EP_0 produced from EPQ_00, EPQ_01; consumed to make EPQ_00, EPQ_01
    dEP_0_v4 = (v4f_00 + v4f_01) - (v4b_00 + v4b_01)
    # EP_1 produced from EPQ_10, EPQ_11; consumed to make EPQ_10, EPQ_11
    dEP_1_v4 = (v4f_10 + v4f_11) - (v4b_10 + v4b_11)

    # Q_0 produced from EPQ_00, EPQ_10; consumed to make EPQ_00, EPQ_10
    dQ_0_v4 = (v4f_00 + v4f_10) - (v4b_00 + v4b_10)
    # Q_1 produced from EPQ_01, EPQ_11; consumed to make EPQ_01, EPQ_11
    dQ_1_v4 = (v4f_01 + v4f_11) - (v4b_01 + v4b_11)

    # =========================================================================
    # v5: EP <-> E + P (atoms preserved)
    # =========================================================================
    v5f_0 = k5f * EP_0
    v5f_1 = k5f * EP_1

    v5b_0 = k5r * E * P_0
    v5b_1 = k5r * E * P_1

    dEP_0_v5 = -v5f_0 + v5b_0
    dEP_1_v5 = -v5f_1 + v5b_1

    dP_0_v5 = v5f_0 - v5b_0
    dP_1_v5 = v5f_1 - v5b_1

    # =========================================================================
    # v_Q_ex: Q_e <-> Q (exchange with external Q pool)
    # Q_e is 100% unlabeled (isotopomer Q_0)
    # Net flux: v_Q_ex = k_Q_ex * (Q_e - Q)
    # - When v_Q_ex > 0 (net influx): add to Q_0 only
    # - When v_Q_ex < 0 (net efflux): remove proportionally from all isotopomers
    # =========================================================================
    v_Q_ex = k_Q_ex * (Q_e - Q)
    Q_safe = jnp.maximum(Q, 1e-10)
    dQ_0_vqex = jnp.where(v_Q_ex >= 0, v_Q_ex, v_Q_ex * Q_0 / Q_safe)
    dQ_1_vqex = jnp.where(v_Q_ex >= 0, 0.0, v_Q_ex * Q_1 / Q_safe)

    # =========================================================================
    # v_P_ex: P_e <-> P (exchange with external P pool)
    # P_e is 100% unlabeled (isotopomer P_0)
    # Net flux: v_P_ex = k_P_ex * (P_e - P)
    # - When v_P_ex > 0 (net influx): add to P_0 only
    # - When v_P_ex < 0 (net efflux): remove proportionally from all isotopomers
    # =========================================================================
    v_P_ex = k_P_ex * (P_e - P)
    P_safe = jnp.maximum(P, 1e-10)
    dP_0_vpex = jnp.where(v_P_ex >= 0, v_P_ex, v_P_ex * P_0 / P_safe)
    dP_1_vpex = jnp.where(v_P_ex >= 0, 0.0, v_P_ex * P_1 / P_safe)

    # =========================================================================
    # vpq: P <-> Q (atoms preserved)
    # =========================================================================
    vpq_f_0 = k_pq_f * P_0
    vpq_f_1 = k_pq_f * P_1

    vpq_b_0 = k_pq_r * Q_0
    vpq_b_1 = k_pq_r * Q_1

    dP_0_vpq = -vpq_f_0 + vpq_b_0
    dP_1_vpq = -vpq_f_1 + vpq_b_1

    dQ_0_vpq = vpq_f_0 - vpq_b_0
    dQ_1_vpq = vpq_f_1 - vpq_b_1

    # =========================================================================
    # Sum all contributions
    # =========================================================================
    dA_00 = dA_00_v1 + dA_00_v2
    dA_01 = dA_01_v1 + dA_01_v2
    dA_10 = dA_10_v1 + dA_10_v2
    dA_11 = dA_11_v1 + dA_11_v2

    dEA_00 = dEA_00_v2 + dEA_00_v3
    dEA_01 = dEA_01_v2 + dEA_01_v3
    dEA_10 = dEA_10_v2 + dEA_10_v3
    dEA_11 = dEA_11_v2 + dEA_11_v3

    dEPQ_00 = dEPQ_00_v3 + dEPQ_00_v4
    dEPQ_01 = dEPQ_01_v3 + dEPQ_01_v4
    dEPQ_10 = dEPQ_10_v3 + dEPQ_10_v4
    dEPQ_11 = dEPQ_11_v3 + dEPQ_11_v4

    dEP_0 = dEP_0_v4 + dEP_0_v5
    dEP_1 = dEP_1_v4 + dEP_1_v5

    dQ_0 = dQ_0_v4 + dQ_0_vqex + dQ_0_vpq
    dQ_1 = dQ_1_v4 + dQ_1_vqex + dQ_1_vpq

    dP_0 = dP_0_v5 + dP_0_vpex + dP_0_vpq
    dP_1 = dP_1_v5 + dP_1_vpex + dP_1_vpq

    return jnp.array([
        dA_00, dA_01, dA_10, dA_11,
        dEA_00, dEA_01, dEA_10, dEA_11,
        dEPQ_00, dEPQ_01, dEPQ_10, dEPQ_11,
        dEP_0, dEP_1,
        dQ_0, dQ_1,
        dP_0, dP_1,
    ])


def simulate_isotopomers(params, initial_total, t_span=(0, 100), n_points=100):
    """Simulate the isotopomer ODE system."""
    y0 = build_initial_state(initial_total)

    t_eval = jnp.linspace(t_span[0], t_span[1], n_points)

    term = diffrax.ODETerm(isotopomer_ode)
    solver = diffrax.Kvaerno3()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    saveat = diffrax.SaveAt(ts=t_eval)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=1e-4,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1000000,
    )

    return sol


def compute_mids(y):
    """Compute MIDs from isotopomer concentrations.

    Returns dict with MID for each metabolite.
    """
    # A: 2 atoms, MID has 3 components (M+0, M+1, M+2)
    A_00, A_01, A_10, A_11 = y[0], y[1], y[2], y[3]
    A_total = A_00 + A_01 + A_10 + A_11
    A_mid = jnp.array([A_00, A_01 + A_10, A_11]) / jnp.maximum(A_total, 1e-10)

    # EA: 2 atoms
    EA_00, EA_01, EA_10, EA_11 = y[4], y[5], y[6], y[7]
    EA_total = EA_00 + EA_01 + EA_10 + EA_11
    EA_mid = jnp.array([EA_00, EA_01 + EA_10, EA_11]) / jnp.maximum(EA_total, 1e-10)

    # EPQ: 2 atoms
    EPQ_00, EPQ_01, EPQ_10, EPQ_11 = y[8], y[9], y[10], y[11]
    EPQ_total = EPQ_00 + EPQ_01 + EPQ_10 + EPQ_11
    EPQ_mid = jnp.array([EPQ_00, EPQ_01 + EPQ_10, EPQ_11]) / jnp.maximum(EPQ_total, 1e-10)

    # EP: 1 atom, MID has 2 components (M+0, M+1)
    EP_0, EP_1 = y[12], y[13]
    EP_total = EP_0 + EP_1
    EP_mid = jnp.array([EP_0, EP_1]) / jnp.maximum(EP_total, 1e-10)

    # Q: 1 atom
    Q_0, Q_1 = y[14], y[15]
    Q_total = Q_0 + Q_1
    Q_mid = jnp.array([Q_0, Q_1]) / jnp.maximum(Q_total, 1e-10)

    # P: 1 atom
    P_0, P_1 = y[16], y[17]
    P_total = P_0 + P_1
    P_mid = jnp.array([P_0, P_1]) / jnp.maximum(P_total, 1e-10)

    return {
        "A_12": A_mid,
        "EA_12": EA_mid,
        "EPQ_12": EPQ_mid,
        "EP_1": EP_mid,
        "Q_1": Q_mid,
        "P_1": P_mid,
    }


def main():
    print("=" * 70)
    print("Isotopomer Simulation")
    print("=" * 70)
    sys.stdout.flush()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    params = DEFAULT_PARAMS.copy()
    initial_total = DEFAULT_INITIAL_TOTAL.copy()

    print("\nSimulating isotopomer dynamics...")
    sys.stdout.flush()

    sol = simulate_isotopomers(params, initial_total, t_span=(0, 100), n_points=100)

    if sol.result != diffrax.RESULTS.successful:
        print(f"ERROR: ODE integration failed: {sol.result}")
        sys.exit(1)

    print("Simulation complete.")
    sys.stdout.flush()

    # Extract steady-state MIDs
    y_ss = sol.ys[-1]
    ss_mids = compute_mids(y_ss)

    print("\n" + "=" * 70)
    print("Steady-State MIDs (Isotopomer Simulation)")
    print("=" * 70)
    for met, mid in ss_mids.items():
        print(f"  {met}: {np.array(mid)}")

    # Compare with freeflux (run enzyme_simulation to get reference)
    print("\n" + "=" * 70)
    print("Time-Course MIDs at Selected Timepoints")
    print("=" * 70)

    display_indices = [0, 25, 50, 75, -1]
    for idx in display_indices:
        t = float(sol.ts[idx])
        y = sol.ys[idx]
        mids = compute_mids(y)
        print(f"\nt = {t:.2f}s:")
        for met, mid in mids.items():
            print(f"  {met}: {np.array(mid)}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()

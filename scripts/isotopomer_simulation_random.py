#!/usr/bin/env python3
"""Isotopomer simulation for the random uni-bi enzyme mechanism.

This module provides ODE functions for the random mechanism where either
product (P or Q) can release first from the EPQ complex.

Random mechanism:
                    +-> EP + Q -> E + P + Q  (path 1: Q released first, v4a/v5a)
E + A <-> EA <-> EPQ -|
                    +-> EQ + P -> E + Q + P  (path 2: P released first, v4b/v5b)

Mass balances:
- dA/dt   = v1 - v1_out - v2
- dEA/dt  = v2 - v3
- dEPQ/dt = v3 - v4a - v4b              # Both pathways consume EPQ
- dEP/dt  = v4a - v5a                   # Path 1: Q released first
- dEQ/dt  = v4b - v5b                   # Path 2: P released first
- dQ/dt   = v4a + v5b + vq_ex + vpq     # Q from v4a and v5b
- dP/dt   = v4b + v5a + vp_ex - vpq     # P from v4b and v5a

Enzyme conservation: E = E_T - EA - EPQ - EP - EQ

Labeling:
  - A_e (external A): 100% labeled on both atoms (isotopomer A_11)
  - Q_e, P_e (external Q, P): 100% unlabeled
"""

from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

try:
    from scripts.isotopomer_network_random import (
        STATE_NAMES_RANDOM,
        STATE_INDEX_RANDOM,
        N_ISOTOPOMERS_RANDOM,
    )
except ModuleNotFoundError:
    from isotopomer_network_random import (
        STATE_NAMES_RANDOM,
        STATE_INDEX_RANDOM,
        N_ISOTOPOMERS_RANDOM,
    )


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "isotopomer_simulation_random"

# Default kinetic parameters for the random mechanism
DEFAULT_PARAMS_RANDOM = {
    # External pool concentrations (fixed boundary conditions)
    "A_e": 5.0,          # external A concentration
    "Q_e": 1.0,          # external Q concentration
    "P_e": 1.0,          # external P concentration
    # Exchange rate constants
    "k_A_ex": 0.1,       # A exchange rate constant
    "k_Q_ex": 0.1,       # Q exchange rate constant
    "k_P_ex": 0.1,       # P exchange rate constant
    # Enzyme kinetic rate constants
    "k2f": 10.0,         # A + E -> EA forward (computed from Wegscheider)
    "k2r": 250.0,        # EA -> A + E reverse
    "k3f": 125.0,        # EA -> EPQ forward
    "k3r": 12.5,         # EPQ -> EA reverse
    # Path 1: Q releases first (EPQ -> EP + Q -> E + P + Q)
    "k4af": 100.0,       # EPQ -> EP + Q forward
    "k4ar": 10.0,        # EP + Q -> EPQ reverse
    "k5af": 75.0,        # EP -> E + P forward
    "k5ar": 7.5,         # E + P -> EP reverse
    # Path 2: P releases first (EPQ -> EQ + P -> E + Q + P)
    "k4bf": 100.0,       # EPQ -> EQ + P forward
    "k4br": 10.0,        # EQ + P -> EPQ reverse
    "k5bf": 75.0,        # EQ -> E + Q forward
    "k5br": 7.5,         # E + Q -> EQ reverse (computed from Wegscheider loop)
    # P <-> Q interconversion
    "k_pq_f": 50.0,      # P -> Q forward
    "k_pq_r": 15.0,      # Q -> P reverse (computed from Wegscheider)
    "E_T": 1.0,          # total enzyme concentration
}

# Initial total concentrations (all unlabelled at t=0)
DEFAULT_INITIAL_TOTAL_RANDOM = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
    "EQ": 0.0,
}


def build_initial_state_random(total_concs):
    """Build initial isotopomer state from total concentrations.

    All metabolites start completely unlabelled.
    """
    y0 = jnp.zeros(N_ISOTOPOMERS_RANDOM)

    # A starts unlabelled (A_00)
    y0 = y0.at[STATE_INDEX_RANDOM["A_00"]].set(total_concs["A"])

    # EA starts unlabelled (EA_00)
    y0 = y0.at[STATE_INDEX_RANDOM["EA_00"]].set(total_concs["EA"])

    # EPQ starts unlabelled (EPQ_00)
    y0 = y0.at[STATE_INDEX_RANDOM["EPQ_00"]].set(total_concs["EPQ"])

    # EP starts unlabelled (EP_0)
    y0 = y0.at[STATE_INDEX_RANDOM["EP_0"]].set(total_concs["EP"])

    # EQ starts unlabelled (EQ_0)
    y0 = y0.at[STATE_INDEX_RANDOM["EQ_0"]].set(total_concs["EQ"])

    # Q starts unlabelled (Q_0)
    y0 = y0.at[STATE_INDEX_RANDOM["Q_0"]].set(total_concs["Q"])

    # P starts unlabelled (P_0)
    y0 = y0.at[STATE_INDEX_RANDOM["P_0"]].set(total_concs["P"])

    return y0


def enzyme_ode_random(t, y, params):
    """ODE system for total concentrations (no isotope labeling) - random mechanism.

    This function solves for steady-state fluxes and concentrations without
    tracking individual isotopomers. Use this to find metabolic steady state,
    then use isotopomer_ode_random for labeling dynamics.

    State variables: [A, Q, P, EA, EPQ, EP, EQ]

    Mass balances:
    - A: exchange with external pool, consumed by enzyme binding
    - Q: produced by v4a, v5b; exchange with external pool; interconversion with P
    - P: produced by v4b, v5a; exchange with external pool; interconversion with Q
    - EA, EPQ, EP, EQ: enzyme intermediates

    Enzyme conservation: E = E_T - EA - EPQ - EP - EQ
    """
    A, Q, P, EA, EPQ, EP, EQ = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

    # Free enzyme from conservation
    E = params["E_T"] - EA - EPQ - EP - EQ

    # Extract parameters
    A_e = params["A_e"]
    Q_e = params["Q_e"]
    P_e = params["P_e"]
    k_A_ex = params["k_A_ex"]
    k_Q_ex = params["k_Q_ex"]
    k_P_ex = params["k_P_ex"]
    k2f, k2r = params["k2f"], params["k2r"]
    k3f, k3r = params["k3f"], params["k3r"]
    k4af, k4ar = params["k4af"], params["k4ar"]
    k5af, k5ar = params["k5af"], params["k5ar"]
    k4bf, k4br = params["k4bf"], params["k4br"]
    k5bf, k5br = params["k5bf"], params["k5br"]
    k_pq_f, k_pq_r = params["k_pq_f"], params["k_pq_r"]

    # Exchange reactions with external pools
    v_A_ex = k_A_ex * (A_e - A)
    v_Q_ex = k_Q_ex * (Q_e - Q)
    v_P_ex = k_P_ex * (P_e - P)

    # Enzyme-catalyzed reactions
    v2 = A * E * k2f - EA * k2r
    v3 = EA * k3f - EPQ * k3r
    # Path 1: Q releases first
    v4a = EPQ * k4af - Q * EP * k4ar
    v5a = EP * k5af - E * P * k5ar
    # Path 2: P releases first
    v4b = EPQ * k4bf - P * EQ * k4br
    v5b = EQ * k5bf - E * Q * k5br

    # P <-> Q interconversion
    v_pq = P * k_pq_f - Q * k_pq_r

    # Mass balances
    dA_dt = v_A_ex - v2
    dQ_dt = v4a + v5b + v_Q_ex + v_pq    # Q from path 1 (v4a) and path 2 (v5b)
    dP_dt = v4b + v5a + v_P_ex - v_pq    # P from path 2 (v4b) and path 1 (v5a)
    dEA_dt = v2 - v3
    dEPQ_dt = v3 - v4a - v4b              # EPQ consumed by both pathways
    dEP_dt = v4a - v5a                    # Path 1: EP formed from v4a, consumed by v5a
    dEQ_dt = v4b - v5b                    # Path 2: EQ formed from v4b, consumed by v5b

    return jnp.array([dA_dt, dQ_dt, dP_dt, dEA_dt, dEPQ_dt, dEP_dt, dEQ_dt])


def isotopomer_ode_random(t, y, params):
    """ODE system for isotopomer dynamics - random mechanism.

    State vector y has 20 components:
      A_00, A_01, A_10, A_11,       (indices 0-3)
      EA_00, EA_01, EA_10, EA_11,   (indices 4-7)
      EPQ_00, EPQ_01, EPQ_10, EPQ_11, (indices 8-11)
      EP_0, EP_1,                    (indices 12-13)
      EQ_0, EQ_1,                    (indices 14-15) NEW
      Q_0, Q_1,                      (indices 16-17)
      P_0, P_1                       (indices 18-19)

    Atom mappings for the random mechanism:
      - v2: A(ab) <-> EA(ab) (atoms preserved)
      - v3: EA(ab) <-> EPQ(ab) (atoms preserved)
      - v4a: EPQ(ab) <-> EP(a) + Q(b) (split: atom 'a' to EP, atom 'b' to Q)
      - v4b: EPQ(ab) <-> EQ(b) + P(a) (split: atom 'b' to EQ, atom 'a' to P)
      - v5a: EP(a) <-> E + P(a) (atom preserved)
      - v5b: EQ(b) <-> E + Q(b) (atom preserved, but tracked as 'a' in Q since Q has 1 atom)
      - vpq: P(a) <-> Q(a) (atom preserved)

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
    EQ_0, EQ_1 = y[14], y[15]
    Q_0, Q_1 = y[16], y[17]
    P_0, P_1 = y[18], y[19]

    # Total concentrations
    A = A_00 + A_01 + A_10 + A_11
    EA = EA_00 + EA_01 + EA_10 + EA_11
    EPQ = EPQ_00 + EPQ_01 + EPQ_10 + EPQ_11
    EP = EP_0 + EP_1
    EQ = EQ_0 + EQ_1
    Q = Q_0 + Q_1
    P = P_0 + P_1

    # Free enzyme from conservation
    E = params["E_T"] - EA - EPQ - EP - EQ

    # Extract parameters
    A_e = params["A_e"]
    Q_e = params["Q_e"]
    P_e = params["P_e"]
    k_A_ex = params["k_A_ex"]
    k_Q_ex = params["k_Q_ex"]
    k_P_ex = params["k_P_ex"]
    k2f, k2r = params["k2f"], params["k2r"]
    k3f, k3r = params["k3f"], params["k3r"]
    k4af, k4ar = params["k4af"], params["k4ar"]
    k5af, k5ar = params["k5af"], params["k5ar"]
    k4bf, k4br = params["k4bf"], params["k4br"]
    k5bf, k5br = params["k5bf"], params["k5br"]
    k_pq_f, k_pq_r = params["k_pq_f"], params["k_pq_r"]

    # =========================================================================
    # v1: A_ext -> A (input of labeled substrate)
    # A_ext is 100% labeled on both atoms (isotopomer A_11)
    # =========================================================================
    v1_in = k_A_ex * A_e
    v1_out = k_A_ex * A

    A_safe = jnp.maximum(A, 1e-10)
    dA_00_v1 = -v1_out * A_00 / A_safe
    dA_01_v1 = -v1_out * A_01 / A_safe
    dA_10_v1 = -v1_out * A_10 / A_safe
    dA_11_v1 = v1_in - v1_out * A_11 / A_safe

    # =========================================================================
    # v2: A + E <-> EA (atoms preserved)
    # =========================================================================
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
    # v4a: EPQ <-> EP + Q (Path 1: Q releases first)
    # EPQ(ab) -> EP(a) + Q(b)
    # Atom 'a' (position 0) goes to EP, atom 'b' (position 1) goes to Q
    # =========================================================================
    # Forward rates: EPQ_xy -> EP_x + Q_y
    v4af_00 = k4af * EPQ_00  # -> EP_0 + Q_0
    v4af_01 = k4af * EPQ_01  # -> EP_0 + Q_1
    v4af_10 = k4af * EPQ_10  # -> EP_1 + Q_0
    v4af_11 = k4af * EPQ_11  # -> EP_1 + Q_1

    # Backward rates: EP_x + Q_y -> EPQ_xy
    v4ab_00 = k4ar * EP_0 * Q_0  # -> EPQ_00
    v4ab_01 = k4ar * EP_0 * Q_1  # -> EPQ_01
    v4ab_10 = k4ar * EP_1 * Q_0  # -> EPQ_10
    v4ab_11 = k4ar * EP_1 * Q_1  # -> EPQ_11

    dEPQ_00_v4a = -v4af_00 + v4ab_00
    dEPQ_01_v4a = -v4af_01 + v4ab_01
    dEPQ_10_v4a = -v4af_10 + v4ab_10
    dEPQ_11_v4a = -v4af_11 + v4ab_11

    # EP_0 produced from EPQ_00, EPQ_01 (first atom = 0); consumed to make EPQ_00, EPQ_01
    dEP_0_v4a = (v4af_00 + v4af_01) - (v4ab_00 + v4ab_01)
    # EP_1 produced from EPQ_10, EPQ_11 (first atom = 1); consumed to make EPQ_10, EPQ_11
    dEP_1_v4a = (v4af_10 + v4af_11) - (v4ab_10 + v4ab_11)

    # Q_0 produced from EPQ_00, EPQ_10 (second atom = 0); consumed to make EPQ_00, EPQ_10
    dQ_0_v4a = (v4af_00 + v4af_10) - (v4ab_00 + v4ab_10)
    # Q_1 produced from EPQ_01, EPQ_11 (second atom = 1); consumed to make EPQ_01, EPQ_11
    dQ_1_v4a = (v4af_01 + v4af_11) - (v4ab_01 + v4ab_11)

    # =========================================================================
    # v4b: EPQ <-> EQ + P (Path 2: P releases first)
    # EPQ(ab) -> EQ(b) + P(a)
    # Atom 'b' (position 1) goes to EQ, atom 'a' (position 0) goes to P
    # =========================================================================
    # Forward rates: EPQ_xy -> EQ_y + P_x
    v4bf_00 = k4bf * EPQ_00  # -> EQ_0 + P_0
    v4bf_01 = k4bf * EPQ_01  # -> EQ_1 + P_0
    v4bf_10 = k4bf * EPQ_10  # -> EQ_0 + P_1
    v4bf_11 = k4bf * EPQ_11  # -> EQ_1 + P_1

    # Backward rates: EQ_y + P_x -> EPQ_xy
    v4bb_00 = k4br * EQ_0 * P_0  # -> EPQ_00
    v4bb_01 = k4br * EQ_1 * P_0  # -> EPQ_01
    v4bb_10 = k4br * EQ_0 * P_1  # -> EPQ_10
    v4bb_11 = k4br * EQ_1 * P_1  # -> EPQ_11

    dEPQ_00_v4b = -v4bf_00 + v4bb_00
    dEPQ_01_v4b = -v4bf_01 + v4bb_01
    dEPQ_10_v4b = -v4bf_10 + v4bb_10
    dEPQ_11_v4b = -v4bf_11 + v4bb_11

    # EQ_0 produced from EPQ_00, EPQ_10 (second atom = 0); consumed to make EPQ_00, EPQ_10
    dEQ_0_v4b = (v4bf_00 + v4bf_10) - (v4bb_00 + v4bb_10)
    # EQ_1 produced from EPQ_01, EPQ_11 (second atom = 1); consumed to make EPQ_01, EPQ_11
    dEQ_1_v4b = (v4bf_01 + v4bf_11) - (v4bb_01 + v4bb_11)

    # P_0 produced from EPQ_00, EPQ_01 (first atom = 0); consumed to make EPQ_00, EPQ_01
    dP_0_v4b = (v4bf_00 + v4bf_01) - (v4bb_00 + v4bb_01)
    # P_1 produced from EPQ_10, EPQ_11 (first atom = 1); consumed to make EPQ_10, EPQ_11
    dP_1_v4b = (v4bf_10 + v4bf_11) - (v4bb_10 + v4bb_11)

    # =========================================================================
    # v5a: EP <-> E + P (Path 1: release P from EP)
    # Atom preserved: EP_x -> P_x
    # =========================================================================
    v5af_0 = k5af * EP_0
    v5af_1 = k5af * EP_1

    v5ab_0 = k5ar * E * P_0
    v5ab_1 = k5ar * E * P_1

    dEP_0_v5a = -v5af_0 + v5ab_0
    dEP_1_v5a = -v5af_1 + v5ab_1

    dP_0_v5a = v5af_0 - v5ab_0
    dP_1_v5a = v5af_1 - v5ab_1

    # =========================================================================
    # v5b: EQ <-> E + Q (Path 2: release Q from EQ)
    # Atom preserved: EQ_x -> Q_x
    # =========================================================================
    v5bf_0 = k5bf * EQ_0
    v5bf_1 = k5bf * EQ_1

    v5bb_0 = k5br * E * Q_0
    v5bb_1 = k5br * E * Q_1

    dEQ_0_v5b = -v5bf_0 + v5bb_0
    dEQ_1_v5b = -v5bf_1 + v5bb_1

    dQ_0_v5b = v5bf_0 - v5bb_0
    dQ_1_v5b = v5bf_1 - v5bb_1

    # =========================================================================
    # vq_in/vq_out: Q exchange with external Q pool
    # Q_e (external) is 100% unlabeled (isotopomer Q_0)
    # =========================================================================
    vq_in = k_Q_ex * Q_e
    vq_out = k_Q_ex * Q

    Q_safe = jnp.maximum(Q, 1e-10)
    dQ_0_vqex = vq_in - vq_out * Q_0 / Q_safe
    dQ_1_vqex = -vq_out * Q_1 / Q_safe

    # =========================================================================
    # vp_in/vp_out: P exchange with external P pool
    # P_e (external) is 100% unlabeled (isotopomer P_0)
    # =========================================================================
    vp_in = k_P_ex * P_e
    vp_out = k_P_ex * P

    P_safe = jnp.maximum(P, 1e-10)
    dP_0_vpex = vp_in - vp_out * P_0 / P_safe
    dP_1_vpex = -vp_out * P_1 / P_safe

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

    dEPQ_00 = dEPQ_00_v3 + dEPQ_00_v4a + dEPQ_00_v4b
    dEPQ_01 = dEPQ_01_v3 + dEPQ_01_v4a + dEPQ_01_v4b
    dEPQ_10 = dEPQ_10_v3 + dEPQ_10_v4a + dEPQ_10_v4b
    dEPQ_11 = dEPQ_11_v3 + dEPQ_11_v4a + dEPQ_11_v4b

    dEP_0 = dEP_0_v4a + dEP_0_v5a
    dEP_1 = dEP_1_v4a + dEP_1_v5a

    dEQ_0 = dEQ_0_v4b + dEQ_0_v5b
    dEQ_1 = dEQ_1_v4b + dEQ_1_v5b

    dQ_0 = dQ_0_v4a + dQ_0_v5b + dQ_0_vqex + dQ_0_vpq
    dQ_1 = dQ_1_v4a + dQ_1_v5b + dQ_1_vqex + dQ_1_vpq

    dP_0 = dP_0_v4b + dP_0_v5a + dP_0_vpex + dP_0_vpq
    dP_1 = dP_1_v4b + dP_1_v5a + dP_1_vpex + dP_1_vpq

    return jnp.array([
        dA_00, dA_01, dA_10, dA_11,
        dEA_00, dEA_01, dEA_10, dEA_11,
        dEPQ_00, dEPQ_01, dEPQ_10, dEPQ_11,
        dEP_0, dEP_1,
        dEQ_0, dEQ_1,
        dQ_0, dQ_1,
        dP_0, dP_1,
    ])


def simulate_isotopomers_random(params, initial_total, t_span=(0, 100), n_points=100):
    """Simulate the isotopomer ODE system for the random mechanism."""
    y0 = build_initial_state_random(initial_total)

    t_eval = jnp.linspace(t_span[0], t_span[1], n_points)

    term = diffrax.ODETerm(isotopomer_ode_random)
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


def compute_mids_random(y):
    """Compute MIDs from isotopomer concentrations for the random mechanism.

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

    # EQ: 1 atom
    EQ_0, EQ_1 = y[14], y[15]
    EQ_total = EQ_0 + EQ_1
    EQ_mid = jnp.array([EQ_0, EQ_1]) / jnp.maximum(EQ_total, 1e-10)

    # Q: 1 atom
    Q_0, Q_1 = y[16], y[17]
    Q_total = Q_0 + Q_1
    Q_mid = jnp.array([Q_0, Q_1]) / jnp.maximum(Q_total, 1e-10)

    # P: 1 atom
    P_0, P_1 = y[18], y[19]
    P_total = P_0 + P_1
    P_mid = jnp.array([P_0, P_1]) / jnp.maximum(P_total, 1e-10)

    return {
        "A_12": A_mid,
        "EA_12": EA_mid,
        "EPQ_12": EPQ_mid,
        "EP_1": EP_mid,
        "EQ_1": EQ_mid,
        "Q_1": Q_mid,
        "P_1": P_mid,
    }


def main():
    print("=" * 70)
    print("Isotopomer Simulation (Random Mechanism)")
    print("=" * 70)
    sys.stdout.flush()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    params = DEFAULT_PARAMS_RANDOM.copy()
    initial_total = DEFAULT_INITIAL_TOTAL_RANDOM.copy()

    print("\nSimulating isotopomer dynamics...")
    sys.stdout.flush()

    sol = simulate_isotopomers_random(params, initial_total, t_span=(0, 100), n_points=100)

    if sol.result != diffrax.RESULTS.successful:
        print(f"ERROR: ODE integration failed: {sol.result}")
        sys.exit(1)

    print("Simulation complete.")
    sys.stdout.flush()

    # Extract steady-state MIDs
    y_ss = sol.ys[-1]
    ss_mids = compute_mids_random(y_ss)

    print("\n" + "=" * 70)
    print("Steady-State MIDs (Isotopomer Simulation - Random Mechanism)")
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
        mids = compute_mids_random(y)
        print(f"\nt = {t:.2f}s:")
        for met, mid in mids.items():
            print(f"  {met}: {np.array(mid)}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()

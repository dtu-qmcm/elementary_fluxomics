#!/usr/bin/env python3
"""Validate isotopomer ODE against freeflux EMU simulation.

This script compares the steady-state MIDs computed by:
1. Custom isotopomer ODE (18-state system in isotopomer_simulation.py)
2. Freeflux EMU algorithm (via emu_jax.py)

Both methods should produce identical MIDs at steady state for the same
fluxes and concentrations.

Validation procedure:
1. Run enzyme_ode to get metabolic steady-state concentrations
2. Compute forward/backward fluxes from those concentrations
3. Run isotopomer_ode from the steady state (all unlabeled) until isotopic steady state
4. Run freeflux EMU simulation with the same fluxes/concentrations
5. Compare the resulting MIDs
"""

from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

try:
    from scripts.isotopomer_simulation import isotopomer_ode, enzyme_ode
    from scripts.emu_jax import run_freeflux_inst_simulation
except ModuleNotFoundError:
    from isotopomer_simulation import isotopomer_ode, enzyme_ode
    from emu_jax import run_freeflux_inst_simulation

import diffrax
from diffrax import SteadyStateEvent


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
REACTIONS_FILE = PROJECT_ROOT / "models" / "simple_network" / "reactions.tsv"

# Thermodynamic constants
R_GAS = 8.314e-3  # Gas constant in kJ/(mol·K)
TEMPERATURE = 298.15  # Temperature in K

FORMATION_ENERGIES = {
    "A": 0.0,
    "P": 0.5,
    "Q": -2.5,
}


def compute_wegscheider_constraints(params):
    """Compute dependent rate constants from Wegscheider conditions."""
    RT = R_GAS * TEMPERATURE

    delta_G_enzyme = FORMATION_ENERGIES["P"] + FORMATION_ENERGIES["Q"] - FORMATION_ENERGIES["A"]
    K_enzyme = np.exp(-delta_G_enzyme / RT)

    delta_G_pq = FORMATION_ENERGIES["Q"] - FORMATION_ENERGIES["P"]
    K_pq = np.exp(-delta_G_pq / RT)

    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

    k_pq_r = params["k_pq_f"] / K_pq

    updated_params = params.copy()
    updated_params["k2f"] = k2f
    updated_params["k_pq_r"] = k_pq_r

    return updated_params


def simulate_enzyme_to_steady_state(params, initial_array):
    """Run enzyme ODE to metabolic steady state."""
    term = diffrax.ODETerm(enzyme_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-10, atol=1e-12)

    # Tight tolerance for accurate steady-state detection
    discrete_terminating_event = SteadyStateEvent(rtol=1e-8, atol=1e-10)
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


def simulate_isotopomers_to_steady_state(params, initial_total_array):
    """Run isotopomer ODE to isotopic steady state."""
    y0 = jnp.zeros(18)
    y0 = y0.at[0].set(initial_total_array[0])   # A_00
    y0 = y0.at[4].set(initial_total_array[3])   # EA_00
    y0 = y0.at[8].set(initial_total_array[4])   # EPQ_00
    y0 = y0.at[12].set(initial_total_array[5])  # EP_0
    y0 = y0.at[14].set(initial_total_array[1])  # Q_0
    y0 = y0.at[16].set(initial_total_array[2])  # P_0

    term = diffrax.ODETerm(isotopomer_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-10, atol=1e-12)

    # Tight tolerance for accurate isotopic steady-state detection
    discrete_terminating_event = SteadyStateEvent(rtol=1e-8, atol=1e-10)
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


def compute_mids_from_isotopomers(y):
    """Compute MIDs from isotopomer state vector."""
    # A: 2 atoms, isotopomers at indices 0-3 (A_00, A_01, A_10, A_11)
    A_00, A_01, A_10, A_11 = y[0], y[1], y[2], y[3]
    A_total = A_00 + A_01 + A_10 + A_11
    A_mid = np.array([A_00, A_01 + A_10, A_11]) / max(A_total, 1e-10)

    # EA: 2 atoms, isotopomers at indices 4-7
    EA_00, EA_01, EA_10, EA_11 = y[4], y[5], y[6], y[7]
    EA_total = EA_00 + EA_01 + EA_10 + EA_11
    EA_mid = np.array([EA_00, EA_01 + EA_10, EA_11]) / max(EA_total, 1e-10)

    # EPQ: 2 atoms, isotopomers at indices 8-11
    EPQ_00, EPQ_01, EPQ_10, EPQ_11 = y[8], y[9], y[10], y[11]
    EPQ_total = EPQ_00 + EPQ_01 + EPQ_10 + EPQ_11
    EPQ_mid = np.array([EPQ_00, EPQ_01 + EPQ_10, EPQ_11]) / max(EPQ_total, 1e-10)

    # EP: 1 atom, isotopomers at indices 12-13 (EP_0, EP_1)
    EP_0, EP_1 = y[12], y[13]
    EP_total = EP_0 + EP_1
    EP_mid = np.array([EP_0, EP_1]) / max(EP_total, 1e-10)

    # Q: 1 atom, isotopomers at indices 14-15
    Q_0, Q_1 = y[14], y[15]
    Q_total = Q_0 + Q_1
    Q_mid = np.array([Q_0, Q_1]) / max(Q_total, 1e-10)

    # P: 1 atom, isotopomers at indices 16-17
    P_0, P_1 = y[16], y[17]
    P_total = P_0 + P_1
    P_mid = np.array([P_0, P_1]) / max(P_total, 1e-10)

    return {
        "A_12": A_mid,
        "EA_12": EA_mid,
        "EPQ_12": EPQ_mid,
        "EP_1": EP_mid,
        "Q_1": Q_mid,
        "P_1": P_mid,
    }


def compute_fluxes_for_emu(concs, params):
    """Compute forward/backward fluxes for EMU simulation.

    The freeflux reaction network uses one-way fluxes for exchange reactions:
    - v1: A_ext -> A (input of labeled substrate)
    - vq_out/vq_in, vp_out/vp_in: separate in/out fluxes for Q and P

    The isotopomer ODE now uses the same separate flux model:
    - v1_in = k_A_ex * A_e (influx of labeled A)
    - v1_out = k_A_ex * A (efflux, but freeflux doesn't have this reaction)

    Note: The freeflux model has v1 as one-way input only (no A -> A_ext).
    For proper comparison, we use v1 = k_A_ex * A_e for the input flux.
    """
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    E = concs["E"]

    # External exchange fluxes - match the isotopomer ODE's separate flux model
    # v1: A_ext -> A (input of labeled A_11)
    # v1_out: A -> A_sink (output proportional to isotopomer fractions)
    v1 = params["k_A_ex"] * params["A_e"]
    v1_out = params["k_A_ex"] * A

    # For Q and P, we have separate in/out fluxes
    # vq_in = k_Q_ex * Q_e (input of unlabeled Q_0)
    # vq_out = k_Q_ex * Q (output proportional to isotopomer fractions)
    vq_in = params["k_Q_ex"] * params["Q_e"]
    vq_out = params["k_Q_ex"] * Q

    # Similarly for P
    vp_in = params["k_P_ex"] * params["P_e"]
    vp_out = params["k_P_ex"] * P

    # Enzyme reactions
    v2_f = A * E * params["k2f"]
    v2_b = EA * params["k2r"]
    v3_f = EA * params["k3f"]
    v3_b = EPQ * params["k3r"]
    v4_f = EPQ * params["k4f"]
    v4_b = Q * EP * params["k4r"]
    v5_f = EP * params["k5f"]
    v5_b = E * P * params["k5r"]

    # P <-> Q interconversion
    vpq_f = P * params["k_pq_f"]
    vpq_b = Q * params["k_pq_r"]

    return {
        "v1": float(v1),
        "v1_out": float(v1_out),
        "v2_f": float(v2_f),
        "v2_b": float(v2_b),
        "v3_f": float(v3_f),
        "v3_b": float(v3_b),
        "v4_f": float(v4_f),
        "v4_b": float(v4_b),
        "v5_f": float(v5_f),
        "v5_b": float(v5_b),
        "vq_out": float(vq_out),
        "vq_in": float(vq_in),
        "vp_out": float(vp_out),
        "vp_in": float(vp_in),
        "vpq_f": float(vpq_f),
        "vpq_b": float(vpq_b),
    }


def run_emu_simulation(concs, params, timepoints=None):
    """Run freeflux EMU simulation."""
    if timepoints is None:
        # Use longer simulation time for true isotopic steady state
        # t=1000s is not sufficient; t=5000s gives better convergence
        timepoints = [0.0, 5000.0]

    fluxes = compute_fluxes_for_emu(concs, params)

    target_emus = {
        "A": ["12"],
        "EA": ["12"],
        "EPQ": ["12"],
        "EP": ["1"],
        "Q": ["1"],
        "P": ["1"],
    }

    # Only pass concentrations for free metabolites
    emu_concentrations = {
        "A": concs["A"],
        "Q": concs["Q"],
        "P": concs["P"],
    }

    mdvs = run_freeflux_inst_simulation(
        reactions_file=REACTIONS_FILE,
        target_emus=target_emus,
        timepoints=timepoints,
        labeling_substrate="A_ext",
        labeling_pattern=["11"],
        labeling_percentage=[1.0],
        labeling_purity=[1.0],
        fluxes=fluxes,
        concentrations=emu_concentrations,
        use_jax=True,
    )

    return mdvs


def extract_mids_from_emu(mdvs, t=5000.0):
    """Extract MIDs from EMU simulation output."""
    return {
        "A_12": np.array(mdvs.get("A_12", {}).get(t, [0, 0, 0])),
        "EA_12": np.array(mdvs.get("EA_12", {}).get(t, [0, 0, 0])),
        "EPQ_12": np.array(mdvs.get("EPQ_12", {}).get(t, [0, 0, 0])),
        "EP_1": np.array(mdvs.get("EP_1", {}).get(t, [0, 0])),
        "Q_1": np.array(mdvs.get("Q_1", {}).get(t, [0, 0])),
        "P_1": np.array(mdvs.get("P_1", {}).get(t, [0, 0])),
    }


def main():
    print("=" * 80)
    print("Validation: Isotopomer ODE vs Freeflux EMU")
    print("=" * 80)

    # Default parameters
    params = {
        "A_e": 5.0,
        "Q_e": 1.0,
        "P_e": 1.0,
        "k_A_ex": 0.1,
        "k_Q_ex": 0.1,
        "k_P_ex": 0.1,
        "k2f": 10.0,  # Will be overwritten
        "k2r": 250.0,
        "k3f": 125.0,
        "k3r": 12.5,
        "k4f": 200.0,
        "k4r": 20.0,
        "k5f": 150.0,
        "k5r": 15.0,
        "k_pq_f": 50.0,
        "k_pq_r": 2.0,  # Will be overwritten
        "E_T": 1.0,
    }

    initial = {
        "A": 0.1,
        "Q": 0.1,
        "P": 0.1,
        "EA": 0.0,
        "EPQ": 0.0,
        "EP": 0.0,
    }

    # Apply Wegscheider constraints
    params = compute_wegscheider_constraints(params)
    print(f"\nApplied Wegscheider constraints:")
    print(f"  k2f = {params['k2f']:.6f}")
    print(f"  k_pq_r = {params['k_pq_r']:.6f}")

    # Step 1: Simulate enzyme ODE to metabolic steady state
    print("\n" + "-" * 40)
    print("Step 1: Simulating enzyme ODE to steady state")
    print("-" * 40)

    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"]
    ])

    enzyme_sol = simulate_enzyme_to_steady_state(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Extract concentrations
    A, Q, P = float(enzyme_y_ss[0]), float(enzyme_y_ss[1]), float(enzyme_y_ss[2])
    EA, EPQ, EP = float(enzyme_y_ss[3]), float(enzyme_y_ss[4]), float(enzyme_y_ss[5])
    E = params["E_T"] - EA - EPQ - EP

    concs = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    print(f"\nSteady-state concentrations:")
    for met, val in concs.items():
        print(f"  {met}: {val:.6f}")

    # Print fluxes
    fluxes = compute_fluxes_for_emu(concs, params)
    print(f"\nForward/backward fluxes:")
    for flux_id, val in fluxes.items():
        print(f"  {flux_id}: {val:.6f}")

    # Step 2: Run isotopomer ODE
    print("\n" + "-" * 40)
    print("Step 2: Running isotopomer ODE to isotopic steady state")
    print("-" * 40)

    iso_sol = simulate_isotopomers_to_steady_state(params, enzyme_y_ss)
    iso_y_ss = iso_sol.ys[0] if iso_sol.ys.ndim > 1 else iso_sol.ys

    iso_mids = compute_mids_from_isotopomers(np.array(iso_y_ss))

    print(f"\nIsotopomer ODE MIDs:")
    for emu_id, mid in iso_mids.items():
        print(f"  {emu_id}: {mid}")

    # Step 3: Run freeflux EMU simulation
    print("\n" + "-" * 40)
    print("Step 3: Running freeflux EMU simulation")
    print("-" * 40)

    emu_mdvs = run_emu_simulation(concs, params, timepoints=[0.0, 5000.0])
    emu_mids = extract_mids_from_emu(emu_mdvs, t=5000.0)

    print(f"\nFreeflux EMU MIDs:")
    for emu_id, mid in emu_mids.items():
        print(f"  {emu_id}: {mid}")

    # Step 4: Compare MIDs
    print("\n" + "-" * 40)
    print("Step 4: Comparing MIDs")
    print("-" * 40)

    max_diff = 0.0
    all_close = True

    # Tolerance: 1% difference is acceptable (typical MID measurement error is 1-5%)
    TOLERANCE = 0.01

    print(f"\n{'EMU':<10} {'Max Abs Diff':<15} {'Isotopomer ODE':<35} {'Freeflux EMU':<35}")
    print("-" * 95)

    for emu_id in iso_mids.keys():
        iso_mid = iso_mids[emu_id]
        emu_mid = emu_mids[emu_id]

        diff = np.abs(iso_mid - emu_mid)
        max_diff_emu = np.max(diff)
        max_diff = max(max_diff, max_diff_emu)

        if max_diff_emu > TOLERANCE:
            all_close = False

        iso_str = np.array2string(iso_mid, precision=6, suppress_small=True)
        emu_str = np.array2string(emu_mid, precision=6, suppress_small=True)

        print(f"{emu_id:<10} {max_diff_emu:<15.2e} {iso_str:<35} {emu_str:<35}")

    print("\n" + "=" * 80)
    print("Validation Result")
    print("=" * 80)
    print(f"\nMaximum absolute difference: {max_diff:.2e}")

    if all_close:
        print("\nVALIDATION PASSED: Isotopomer ODE and freeflux EMU produce equivalent MIDs")
    else:
        print("\nVALIDATION FAILED: Significant differences detected between methods")
        print("\nPossible causes:")
        print("  - Different atom mapping conventions")
        print("  - Different boundary conditions for labeled substrate")
        print("  - Numerical precision issues")

    return all_close


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Enzyme kinetics ODE simulation with isotopomer labeling analysis.

This script simulates an enzyme-catalyzed reaction mechanism with external pool
exchange kinetics:

dA/dt = v_A_ex - v2        (exchange with external A pool, binding to enzyme)
dQ/dt = v4 + v_Q_ex + v_pq (produced by enzyme, exchange with external Q pool)
dP/dt = v5 + v_P_ex - v_pq (produced by enzyme, exchange with external P pool)
dEA/dt = v2 - v3
dEPQ/dt = v3 - v4
dEP/dt = v4 - v5

Exchange reactions use symmetric mass-action kinetics:
  v_X_ex = k_X_ex * (X_e - X)   where X_e is the fixed external concentration

Enzyme conservation: E + EA + EPQ + EP = E_T

Atom mapping:
  A(ab) <-> EPQ(ab)
  EPQ(ab) <-> EP(a) + Q(b)
  P(a) <-> Q(a)  (interconversion)

Labeling:
  - A_e (external A): 100% labeled on both atoms (isotopomer A_11)
  - Q_e, P_e (external Q, P): 100% unlabeled

Inference approach:
  1. Run enzyme_ode to get metabolic steady-state fluxes and concentrations
  2. Run isotopomer_ode from that steady state (all unlabeled) to get MIDs
  3. Use fluxes/concentrations from step 1 and MIDs from step 2 in log-likelihood
"""

from pathlib import Path
import sys

# Enable float64 precision BEFORE importing JAX
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import diffrax
from diffrax import SteadyStateEvent
import blackjax

try:
    from scripts.isotopomer_simulation import isotopomer_ode, enzyme_ode
    from scripts.emu_jax import (
        run_freeflux_inst_simulation,
        run_freeflux_ss_simulation,
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
        simulate_ss_mdvs_pure_jax,
        create_flux_array_from_dict,
        create_conc_array_from_dict,
        precompute_matrix_structure,
    )
except ModuleNotFoundError:
    from isotopomer_simulation import isotopomer_ode, enzyme_ode
    from emu_jax import (
        run_freeflux_inst_simulation,
        run_freeflux_ss_simulation,
        load_emu_network_structure,
        simulate_inst_mdvs_pure_jax,
        simulate_ss_mdvs_pure_jax,
        create_flux_array_from_dict,
        create_conc_array_from_dict,
        precompute_matrix_structure,
    )
from freeflux import Model


# Global cache for EMU network (initialized once)
_EMU_NETWORK_CACHE = None
_PURE_JAX_EMU_NETWORK = None


def get_emu_network():
    """Get or initialize the EMU network cache (legacy, uses freeflux lambdas).

    This pre-computes the EMU network structure using freeflux,
    which is then used for JAX-traceable simulation.

    Note: For fully JAX-traceable MCMC, use get_pure_jax_emu_network() instead.
    """
    global _EMU_NETWORK_CACHE

    if _EMU_NETWORK_CACHE is None:
        model = Model("simple_network")
        model.read_from_file(str(REACTIONS_FILE))

        # Target EMUs (measurable metabolites only - not enzyme intermediates)
        target_emus = {
            "A": ["12"],
            "Q": ["1"],
            "P": ["1"],
        }

        with model.simulator("inst") as isim:
            isim.set_target_EMUs(target_emus)
            isim.set_timepoints([0.0, 1.0])  # Dummy timepoints for setup
            isim.set_labeling_strategy(
                "A_ext",
                labeling_pattern=["11"],
                percentage=[1.0],
                purity=[1.0],
            )

            # Set dummy fluxes (non-zero to avoid division errors)
            for flux_id in model.totalfluxids:
                isim.set_flux(flux_id, 1.0)

            # Set dummy concentrations
            for metab_id in ["A", "Q", "P"]:
                isim.set_concentration(metab_id, 1.0)

            isim.prepare(n_jobs=1)

            # Extract network structure
            _EMU_NETWORK_CACHE = {
                "matrix_As": model.matrix_As,
                "matrix_Bs": model.matrix_Bs,
                "matrix_Ms": model.matrix_Ms,
                "initial_matrix_Xs": model.initial_matrix_Xs,
                "initial_matrix_Ys": model.initial_matrix_Ys,
                "substrate_MDVs": {str(k): np.array(v) for k, v in model.substrate_MDVs.items()},
                "flux_ids": list(model.totalfluxids),
                "metab_ids": list(model.metabolites),
            }

    return _EMU_NETWORK_CACHE


def get_pure_jax_emu_network():
    """Get or initialize the pure JAX EMU network structure.

    This loads the serialized EMU network from JSON for fully JAX-traceable
    simulation compatible with gradient-based MCMC. Also precomputes the
    matrix structure for efficient JAX tracing.

    Returns
    -------
    tuple
        (network, precomputed) where:
        - network: Serialized EMU network structure
        - precomputed: Precomputed matrix structure arrays for JAX tracing
    """
    global _PURE_JAX_EMU_NETWORK

    if _PURE_JAX_EMU_NETWORK is None:
        emu_network_path = PROJECT_ROOT / "models" / "simple_network" / "emu_network.json"
        network = load_emu_network_structure(emu_network_path)
        precomputed = precompute_matrix_structure(network)
        _PURE_JAX_EMU_NETWORK = (network, precomputed)

    return _PURE_JAX_EMU_NETWORK


def build_emu_matrices_jax(fluxes, concs, emu_network):
    """Build EMU matrices from fluxes and concentrations (JAX-compatible).

    Parameters
    ----------
    fluxes : dict
        Flux values (JAX arrays or floats)
    concs : dict
        Concentration values (JAX arrays or floats)
    emu_network : dict
        Pre-computed EMU network from get_emu_network()

    Returns
    -------
    dict
        EMU size -> {"A", "B", "M"} matrices as JAX arrays
    """
    matrices = {}

    for size in sorted(emu_network["matrix_As"].keys()):
        lambA, flux_ids_A, product_emus = emu_network["matrix_As"][size]
        lambB, flux_ids_B, source_emus = emu_network["matrix_Bs"][size]
        lambM, metab_ids = emu_network["matrix_Ms"][size]

        # Build flux values arrays
        flux_values_A = [float(fluxes.get(fid, 1.0)) for fid in flux_ids_A]
        flux_values_B = [float(fluxes.get(fid, 1.0)) for fid in flux_ids_B]
        conc_values = [float(concs.get(mid, 1.0)) for mid in metab_ids]

        # Evaluate matrices
        A = jnp.array(lambA(*flux_values_A))
        B = jnp.array(lambB(*flux_values_B))
        M = jnp.array(lambM(*conc_values))

        matrices[size] = {
            "A": A,
            "B": B,
            "M": M,
            "product_emus": product_emus,
            "source_emus": source_emus,
        }

    return matrices


def run_emu_simulation_jax(concs, params, timepoints, emu_network):
    """Run EMU simulation using pure JAX (traceable).

    This function is JAX-traceable and can be used inside vmapped/jitted functions.

    Parameters
    ----------
    concs : dict
        Steady-state concentrations from enzyme_ode
    params : dict
        Rate constants
    timepoints : jnp.array
        Simulation timepoints
    emu_network : dict
        Pre-computed EMU network from get_emu_network()

    Returns
    -------
    dict
        EMU MDVs at final timepoint {emu_id: mdv_array}
    """
    # Compute fluxes for EMU simulation
    fluxes = compute_fluxes_for_emu(concs, params)

    # Build matrices with current flux/concentration values
    matrices = build_emu_matrices_jax(fluxes, concs, emu_network)

    # Get initial states
    Xs = {}
    Ys = {}
    for size in sorted(matrices.keys()):
        Xs[size] = jnp.array(emu_network["initial_matrix_Xs"][size])
        Ys[size] = jnp.array(emu_network["initial_matrix_Ys"][size])

    # Get substrate MDVs (A_ext is fully labeled: [0, 0, 1] for size 2)
    substrate_mdvs = {k: jnp.array(v) for k, v in emu_network["substrate_MDVs"].items()}

    # Run simulation for each timepoint
    sim_mdvs = {}
    t_prev = 0.0

    for t_idx in range(len(timepoints)):
        t = timepoints[t_idx]
        if t <= 0:
            continue

        dt = t - t_prev

        # Process EMU sizes in ascending order
        for size in sorted(matrices.keys()):
            mat = matrices[size]
            A = mat["A"]
            B = mat["B"]
            M = mat["M"]

            X_prev = Xs[size]
            Y_prev = Ys[size]

            # Build Y_current (for INST step, Y changes based on smaller EMUs)
            # For simplicity, assume Y is constant (substrate MDVs don't change)
            Y_curr = Y_prev

            # Run INST step
            X_curr = inst_mdv_step(X_prev, Y_prev, Y_curr, A, B, M, dt)

            Xs[size] = X_curr
            Ys[size] = Y_curr

            # Store EMU MDVs
            product_emus = mat["product_emus"]
            for i, emu in enumerate(product_emus):
                emu_id = emu.id if hasattr(emu, "id") else str(emu)
                sim_mdvs[emu_id] = X_curr[i]

        t_prev = t

    return sim_mdvs


def extract_mids_from_jax_output(sim_mdvs):
    """Extract MIDs from JAX EMU simulation output.

    Parameters
    ----------
    sim_mdvs : dict
        EMU MDVs from run_emu_simulation_jax

    Returns
    -------
    dict
        MIDs in format compatible with existing code
    """
    # Only measurable metabolites (not enzyme intermediates)
    return {
        "A_12": sim_mdvs.get("A_12", jnp.array([0, 0, 0])),
        "Q_1": sim_mdvs.get("Q_1", jnp.array([0, 0])),
        "P_1": sim_mdvs.get("P_1", jnp.array([0, 0])),
    }


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
REACTIONS_FILE = PROJECT_ROOT / "models" / "simple_network" / "reactions.tsv"
RESULTS_DIR = PROJECT_ROOT / "results" / "enzyme_simulation"


# Default parameters
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
    "k2f": 10.0,         # A + E -> EA forward (computed from constraints)
    "k2r": 250.0,        # EA -> A + E reverse (25x)
    "k3f": 125.0,        # EA -> EPQ forward (25x)
    "k3r": 12.5,         # EPQ -> EA reverse (25x)
    "k4f": 200.0,        # EPQ -> EP + Q forward (25x)
    "k4r": 20.0,         # EP + Q -> EPQ reverse (25x)
    "k5f": 150.0,        # EP -> E + P forward (25x)
    "k5r": 15.0,         # E + P -> EP reverse (25x)
    "k_pq_f": 50.0,      # P -> Q forward rate constant (25x)
    "k_pq_r": 2.0,       # Q -> P reverse rate constant (computed from constraints)
    "E_T": 1.0,          # total enzyme concentration
}

# Initial conditions (state order: A, Q, P, EA, EPQ, EP)
DEFAULT_INITIAL = {
    "A": 0.1,
    "Q": 0.1,
    "P": 0.1,
    "EA": 0.0,
    "EPQ": 0.0,
    "EP": 0.0,
}

# Experimental conditions for multi-condition fitting
# Fixed seed ensures identical conditions across all scripts
np.random.seed(12345)
EXPERIMENTAL_CONDITIONS = [
    {"E_T": float(E_T), "A_e": float(A_e)}
    for E_T, A_e in zip(
        np.random.uniform(0.5, 2.0, 5),  # E_T varies from 0.5 to 2.0
        np.random.uniform(2.0, 10.0, 5),  # A_e varies from 2.0 to 10.0
    )
]
NUM_CONDITIONS = len(EXPERIMENTAL_CONDITIONS)

# Thermodynamic constants
R_GAS = 8.314e-3  # Gas constant in kJ/(mol·K)
TEMPERATURE = 298.15  # Temperature in K

# Formation energies (chemical potentials) in kJ/mol
# These constrain the Gibbs energies of reactions via:
#   ΔG°(A → P + Q) = μ_P + μ_Q - μ_A = -2 kJ/mol
#   ΔG°(P → Q) = μ_Q - μ_P = -3 kJ/mol
FORMATION_ENERGIES = {
    "A": 0.0,    # Reference state
    "P": 0.5,    # kJ/mol
    "Q": -2.5,   # kJ/mol
}


def create_output_directory():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")
    sys.stdout.flush()
    return RESULTS_DIR


def compute_wegscheider_constraints(params, formation_energies=None, temperature=None):
    """Compute dependent rate constants from Wegscheider conditions.

    The Wegscheider condition ensures thermodynamic consistency by requiring
    that equilibrium constants around reaction cycles satisfy:
        K_cycle = exp(-ΔG°_cycle / RT)

    For the enzyme mechanism (A → P + Q through enzyme intermediates):
        K_enzyme = (k2f · k3f · k4f · k5f) / (k2r · k3r · k4r · k5r) = exp(-ΔG°/RT)
        where ΔG° = μ_P + μ_Q - μ_A

    For the P ⇌ Q interconversion:
        K_pq = k_pq_f / k_pq_r = exp(-ΔG°_pq/RT)
        where ΔG°_pq = μ_Q - μ_P

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants (k2f and k_pq_r will be overwritten)
    formation_energies : dict, optional
        Formation energies for A, P, Q in kJ/mol. Defaults to FORMATION_ENERGIES.
    temperature : float, optional
        Temperature in K. Defaults to TEMPERATURE.

    Returns
    -------
    dict
        Updated parameter dictionary with k2f and k_pq_r computed from constraints
    """
    if formation_energies is None:
        formation_energies = FORMATION_ENERGIES
    if temperature is None:
        temperature = TEMPERATURE

    RT = R_GAS * temperature

    # Gibbs energy for overall enzyme reaction: A → P + Q
    delta_G_enzyme = (
        formation_energies["P"] + formation_energies["Q"] - formation_energies["A"]
    )
    K_enzyme = np.exp(-delta_G_enzyme / RT)

    # Gibbs energy for P → Q interconversion
    delta_G_pq = formation_energies["Q"] - formation_energies["P"]
    K_pq = np.exp(-delta_G_pq / RT)

    # Solve for k2f from Haldane relation (enzyme cycle constraint)
    # K_enzyme = (k2f · k3f · k4f · k5f) / (k2r · k3r · k4r · k5r)
    # k2f = K_enzyme · (k2r · k3r · k4r · k5r) / (k3f · k4f · k5f)
    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

    # Solve for k_pq_r from P↔Q equilibrium constraint
    # K_pq = k_pq_f / k_pq_r
    # k_pq_r = k_pq_f / K_pq
    k_pq_r = params["k_pq_f"] / K_pq

    # Create updated params dict
    updated_params = params.copy()
    updated_params["k2f"] = k2f
    updated_params["k_pq_r"] = k_pq_r

    return updated_params


def compute_wegscheider_constraints_jax(params, formation_energies_array, temperature):
    """JAX-compatible version of Wegscheider constraint computation.

    Parameters
    ----------
    params : dict
        Parameter dictionary (JAX-traced values)
    formation_energies_array : jnp.array
        Formation energies [μ_A, μ_P, μ_Q] in kJ/mol
    temperature : float
        Temperature in K

    Returns
    -------
    tuple
        (k2f, k_pq_r) computed from thermodynamic constraints
    """
    RT = R_GAS * temperature
    mu_A, mu_P, mu_Q = formation_energies_array[0], formation_energies_array[1], formation_energies_array[2]

    # Gibbs energy for overall enzyme reaction: A → P + Q
    delta_G_enzyme = mu_P + mu_Q - mu_A
    K_enzyme = jnp.exp(-delta_G_enzyme / RT)

    # Gibbs energy for P → Q interconversion
    delta_G_pq = mu_Q - mu_P
    K_pq = jnp.exp(-delta_G_pq / RT)

    # Solve for k2f from Haldane relation
    k2f = K_enzyme * (
        params["k2r"] * params["k3r"] * params["k4r"] * params["k5r"]
    ) / (params["k3f"] * params["k4f"] * params["k5f"])

    # Solve for k_pq_r from P↔Q equilibrium
    k_pq_r = params["k_pq_f"] / K_pq

    return k2f, k_pq_r


def compute_E(EA, EPQ, EP, E_T):
    """Compute free enzyme from conservation constraint."""
    return E_T - EA - EPQ - EP


def compute_fluxes_for_emu(concs, params, use_jax_arrays=False):
    """Compute forward/backward fluxes for EMU simulation.

    The exchange reactions are modeled as separate in/out fluxes:
    - v1: A_ext -> A (input of labeled substrate)
    - v1_out: A -> A_sink (output, removes isotopomers proportionally)
    - vq_in/vq_out, vp_in/vp_out: Q and P exchanges

    Parameters
    ----------
    concs : dict
        Steady-state concentrations {A, Q, P, EA, EPQ, EP, E}
    params : dict
        Rate constants
    use_jax_arrays : bool
        If True, keep values as JAX arrays (for tracing)

    Returns
    -------
    dict
        Fluxes in format expected by freeflux/EMU
    """
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    E = concs["E"]

    # External exchange fluxes (separate in/out for proper isotope tracking)
    v1 = params["k_A_ex"] * params["A_e"]  # Input from A_ext (labeled)
    v1_out = params["k_A_ex"] * A  # Output to A_sink
    vq_out = params["k_Q_ex"] * Q  # Output to Q_ext
    vq_in = params["k_Q_ex"] * params["Q_e"]  # Input from Q_ext_in (unlabeled)
    vp_out = params["k_P_ex"] * P  # Output to P_ext
    vp_in = params["k_P_ex"] * params["P_e"]  # Input from P_ext_in (unlabeled)

    # Enzyme reactions (forward/backward)
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

    fluxes = {
        "v1": v1,
        "v1_out": v1_out,
        "v2_f": v2_f,
        "v2_b": v2_b,
        "v3_f": v3_f,
        "v3_b": v3_b,
        "v4_f": v4_f,
        "v4_b": v4_b,
        "v5_f": v5_f,
        "v5_b": v5_b,
        "vq_out": vq_out,
        "vq_in": vq_in,
        "vp_out": vp_out,
        "vp_in": vp_in,
        "vpq_f": vpq_f,
        "vpq_b": vpq_b,
    }

    # Convert to Python floats if not using JAX arrays (for freeflux compatibility)
    if not use_jax_arrays:
        fluxes = {k: float(v) for k, v in fluxes.items()}

    return fluxes


def run_emu_simulation(concs, params, timepoints=None):
    """Run EMU simulation using freeflux backend with JAX computation.

    Parameters
    ----------
    concs : dict
        Steady-state concentrations from enzyme_ode
    params : dict
        Rate constants
    timepoints : list, optional
        Simulation timepoints. Defaults to [0, 1000] for steady-state.

    Returns
    -------
    dict
        EMU MDVs {emu_id: {time: mdv_array}}
    """
    if timepoints is None:
        timepoints = [0.0, 1000.0]  # Long time for steady-state

    # Compute fluxes for EMU simulation
    fluxes = compute_fluxes_for_emu(concs, params)

    # Target EMUs for the simple network
    target_emus = {
        "A": ["12"],      # 2 atoms
        "EA": ["12"],     # 2 atoms
        "EPQ": ["12"],    # 2 atoms
        "EP": ["1"],      # 1 atom
        "Q": ["1"],       # 1 atom
        "P": ["1"],       # 1 atom
    }

    # Only pass concentrations for free metabolites, not enzyme intermediates
    # Enzyme intermediates (EA, EPQ, EP) cause freeflux cleanup issues
    emu_concentrations = {
        "A": concs["A"],
        "Q": concs["Q"],
        "P": concs["P"],
    }

    # Run freeflux INST simulation
    mdvs = run_freeflux_inst_simulation(
        reactions_file=REACTIONS_FILE,
        target_emus=target_emus,
        timepoints=timepoints,
        labeling_substrate="A_ext",
        labeling_pattern=["11"],  # Fully labeled (both atoms)
        labeling_percentage=[1.0],
        labeling_purity=[1.0],
        fluxes=fluxes,
        concentrations=emu_concentrations,
        use_jax=True,
    )

    return mdvs


def extract_mids_from_emu(mdvs, t=1000.0):
    """Extract MIDs from EMU simulation output.

    Parameters
    ----------
    mdvs : dict
        EMU MDVs from run_emu_simulation
    t : float
        Timepoint to extract

    Returns
    -------
    dict
        MIDs in format compatible with existing code
    """
    # Only measurable metabolites (not enzyme intermediates)
    return {
        "A_12": np.array(mdvs.get("A_12", {}).get(t, [0, 0, 0])),
        "Q_1": np.array(mdvs.get("Q_1", {}).get(t, [0, 0])),
        "P_1": np.array(mdvs.get("P_1", {}).get(t, [0, 0])),
    }


def simulate_enzyme_to_steady_state_jax(params, initial_array):
    """JAX-compatible enzyme ODE simulation to steady state.

    Uses enzyme_ode (total concentrations without isotopomers) to find
    metabolic steady state. This gives the fluxes and concentrations for
    the log-likelihood calculation.

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    initial_array : jnp.array
        Initial concentrations [A, Q, P, EA, EPQ, EP]

    Returns
    -------
    diffrax.Solution
        ODE solution at steady state
    """
    term = diffrax.ODETerm(enzyme_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    discrete_terminating_event = SteadyStateEvent(rtol=1e-4, atol=1e-6)
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
    """JAX-compatible isotopomer simulation to steady state.

    Starts from the given total concentrations (all isotopomers unlabeled)
    and runs until isotopic steady state is reached. The boundary condition
    A_e is fully labeled (A_11).

    Parameters
    ----------
    params : dict
        Parameter dictionary with rate constants
    initial_total_array : jnp.array
        Initial total concentrations [A, Q, P, EA, EPQ, EP] from enzyme_ode
        steady state. All isotopomers start unlabeled.

    Returns
    -------
    diffrax.Solution
        ODE solution at steady state
    """
    # Build initial isotopomer state (all unlabelled)
    y0 = jnp.zeros(18)
    # A starts unlabelled (A_00 at index 0)
    y0 = y0.at[0].set(initial_total_array[0])
    # EA starts unlabelled (EA_00 at index 4)
    y0 = y0.at[4].set(initial_total_array[3])
    # EPQ starts unlabelled (EPQ_00 at index 8)
    y0 = y0.at[8].set(initial_total_array[4])
    # EP starts unlabelled (EP_0 at index 12)
    y0 = y0.at[12].set(initial_total_array[5])
    # Q starts unlabelled (Q_0 at index 14)
    y0 = y0.at[14].set(initial_total_array[1])
    # P starts unlabelled (P_0 at index 16)
    y0 = y0.at[16].set(initial_total_array[2])

    term = diffrax.ODETerm(isotopomer_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Steady-state event: terminates when norm(f) < atol + rtol * norm(y)
    discrete_terminating_event = SteadyStateEvent(rtol=1e-4, atol=1e-6)

    # Save only the final state
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=jnp.inf,  # Run until event triggers
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


def compute_mids_jax(y):
    """JAX-compatible MID computation from isotopomer state.

    Parameters
    ----------
    y : jnp.array
        Isotopomer state vector (18 components)

    Returns
    -------
    dict
        MIDs for each metabolite
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


def compute_rates(y, params):
    """Compute all reaction rates from state and parameters.

    Exchange reactions use symmetric mass-action kinetics:
    v_X_ex = k_X_ex * (X_e - X)
    Positive v_X_ex means net influx, negative means net efflux.
    """
    A, Q, P, EA, EPQ, EP = y
    E = compute_E(EA, EPQ, EP, params["E_T"])

    # Exchange reactions with external pools (symmetric mass-action)
    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)
    v_Q_ex = params["k_Q_ex"] * (params["Q_e"] - Q)
    v_P_ex = params["k_P_ex"] * (params["P_e"] - P)

    # Enzyme-catalyzed reactions
    v2 = A * E * params["k2f"] - EA * params["k2r"]
    v3 = EA * params["k3f"] - EPQ * params["k3r"]
    v4 = EPQ * params["k4f"] - Q * EP * params["k4r"]
    v5 = EP * params["k5f"] - E * P * params["k5r"]

    # P <-> Q interconversion
    v_pq = P * params["k_pq_f"] - Q * params["k_pq_r"]

    return {
        "v_A_ex": v_A_ex, "v2": v2, "v3": v3, "v4": v4, "v5": v5,
        "v_Q_ex": v_Q_ex, "v_P_ex": v_P_ex, "v_pq": v_pq
    }


def ode_system(t, y, args):
    """ODE right-hand side for diffrax.

    State variables: [A, Q, P, EA, EPQ, EP]

    Mass balances:
    - A: exchange with external pool, consumed by enzyme binding
    - Q: produced by v4, exchange with external pool, interconversion with P
    - P: produced by v5, exchange with external pool, interconversion with Q
    - EA, EPQ, EP: enzyme intermediates
    """
    params = args
    rates = compute_rates(y, params)

    # A: exchange with external pool, consumed by enzyme binding
    dA_dt = rates["v_A_ex"] - rates["v2"]

    # Q: produced by v4, exchange with external pool, interconversion with P
    dQ_dt = rates["v4"] + rates["v_Q_ex"] + rates["v_pq"]

    # P: produced by v5, exchange with external pool, interconversion with Q
    dP_dt = rates["v5"] + rates["v_P_ex"] - rates["v_pq"]

    # Enzyme intermediates (unchanged)
    dEA_dt = rates["v2"] - rates["v3"]
    dEPQ_dt = rates["v3"] - rates["v4"]
    dEP_dt = rates["v4"] - rates["v5"]

    return jnp.array([dA_dt, dQ_dt, dP_dt, dEA_dt, dEPQ_dt, dEP_dt])


def simulate_ode_to_steady_state(params, initial):
    """Simulate the ODE system to steady state using diffrax event detection.

    Wegscheider thermodynamic constraints are always applied to compute
    k2f and k_pq_r from the formation energies before simulation.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    initial : dict
        Initial conditions

    Returns
    -------
    diffrax.Solution
        ODE solution at steady state
    """
    # Apply thermodynamic constraints to compute dependent rate constants
    params = compute_wegscheider_constraints(params)

    y0 = jnp.array([initial[k] for k in ["A", "Q", "P", "EA", "EPQ", "EP"]])

    term = diffrax.ODETerm(ode_system)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Steady-state event: terminates when norm(f) < atol + rtol * norm(y)
    discrete_terminating_event = SteadyStateEvent(rtol=1e-4, atol=1e-6)

    # Save only the final state
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=jnp.inf,  # Run until event triggers
        dt0=1e-10,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
        discrete_terminating_event=discrete_terminating_event,
    )

    return sol


def get_steady_state(sol, params):
    """Extract steady-state concentrations and fluxes."""
    # With SaveAt(t1=True), sol.ys is (1, 6) - extract the single row
    y_ss = sol.ys[0] if sol.ys.ndim > 1 else sol.ys
    concentrations = {
        "A": float(y_ss[0]),
        "Q": float(y_ss[1]),
        "P": float(y_ss[2]),
        "EA": float(y_ss[3]),
        "EPQ": float(y_ss[4]),
        "EP": float(y_ss[5]),
        "E": float(compute_E(y_ss[3], y_ss[4], y_ss[5], params["E_T"])),
    }

    rates = compute_rates(y_ss, params)

    return concentrations, rates


def compute_forward_backward_fluxes(concs, params):
    """Compute forward and backward fluxes for each reversible reaction."""
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    E = concs["E"]

    fluxes = {
        # Exchange reactions (net flux)
        "v_A_ex": params["k_A_ex"] * (params["A_e"] - A),
        "v_Q_ex": params["k_Q_ex"] * (params["Q_e"] - Q),
        "v_P_ex": params["k_P_ex"] * (params["P_e"] - P),
        # Enzyme reactions (forward and backward)
        "v2_f": A * E * params["k2f"],
        "v2_b": EA * params["k2r"],
        "v3_f": EA * params["k3f"],
        "v3_b": EPQ * params["k3r"],
        "v4_f": EPQ * params["k4f"],
        "v4_b": Q * EP * params["k4r"],
        "v5_f": EP * params["k5f"],
        "v5_b": E * P * params["k5r"],
        # P <-> Q interconversion
        "vpq_f": P * params["k_pq_f"],
        "vpq_b": Q * params["k_pq_r"],
    }

    return fluxes


def compute_log_likelihood(
    obs_fluxes,
    sim_fluxes,
    obs_concentrations,
    sim_concentrations,
    obs_ss_mids,
    sim_ss_mids,
    obs_timeseries_mids,
    sim_timeseries_mids,
    flux_std=0.1,
    conc_std=0.1,
    mid_std=0.01,
):
    """Compute total log-likelihood of observations given simulated values.

    Parameters
    ----------
    obs_fluxes : dict
        Observed flux values {flux_id: value}
    sim_fluxes : dict
        Simulated flux values {flux_id: value}
    obs_concentrations : dict
        Observed concentration values {metabolite_id: value}
    sim_concentrations : dict
        Simulated concentration values {metabolite_id: value}
    obs_ss_mids : dict
        Observed steady-state MIDs {emu_id: array}
    sim_ss_mids : dict
        Simulated steady-state MIDs {emu_id: array}
    obs_timeseries_mids : dict
        Observed timeseries MIDs {emu_id: {time: array}}
    sim_timeseries_mids : dict
        Simulated timeseries MIDs {emu_id: {time: array}}
    flux_std : float
        Standard deviation for flux normal distribution (default 0.1)
    conc_std : float
        Standard deviation for concentration lognormal distribution (default 0.1)
    mid_std : float
        Standard deviation for MID lognormal distribution (default 0.01)

    Returns
    -------
    dict
        Log-likelihoods for each component and total
    """
    log_likelihood = {
        "fluxes": 0.0,
        "concentrations": 0.0,
        "ss_mids": 0.0,
        "timeseries_mids": 0.0,
        "total": 0.0,
    }

    # 1. Fluxes: Normal distribution
    for flux_id in obs_fluxes:
        if flux_id in sim_fluxes:
            obs = obs_fluxes[flux_id]
            sim = sim_fluxes[flux_id]
            # Normal log-likelihood: -0.5 * ((obs - sim) / sigma)^2 - log(sigma * sqrt(2*pi))
            residual = (obs - sim) / flux_std
            log_likelihood["fluxes"] += -0.5 * residual**2 - jnp.log(flux_std * jnp.sqrt(2 * jnp.pi))

    # 2. Concentrations: Lognormal distribution
    # If X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma)
    # We compare log(obs) to log(sim) with given std_dev
    for met_id in obs_concentrations:
        if met_id in sim_concentrations:
            obs = obs_concentrations[met_id]
            sim = sim_concentrations[met_id]
            if obs > 0 and sim > 0:
                log_obs = jnp.log(obs)
                log_sim = jnp.log(sim)
                residual = (log_obs - log_sim) / conc_std
                log_likelihood["concentrations"] += -0.5 * residual**2 - jnp.log(conc_std * jnp.sqrt(2 * jnp.pi))
                # Jacobian term for lognormal: -log(obs)
                log_likelihood["concentrations"] += -jnp.log(obs)

    # 3. Steady-state MIDs: Lognormal distribution
    for emu_id in obs_ss_mids:
        if emu_id in sim_ss_mids:
            obs_mid = jnp.asarray(obs_ss_mids[emu_id])
            sim_mid = jnp.asarray(sim_ss_mids[emu_id])
            # Only include positive values (MID fractions should be > 0)
            for i in range(len(obs_mid)):
                obs_val = obs_mid[i]
                sim_val = sim_mid[i]
                if obs_val > 0 and sim_val > 0:
                    log_obs = jnp.log(obs_val)
                    log_sim = jnp.log(sim_val)
                    residual = (log_obs - log_sim) / mid_std
                    log_likelihood["ss_mids"] += -0.5 * residual**2 - jnp.log(mid_std * jnp.sqrt(2 * jnp.pi))
                    log_likelihood["ss_mids"] += -jnp.log(obs_val)

    # 4. Timeseries MIDs: Lognormal distribution
    for emu_id in obs_timeseries_mids:
        if emu_id in sim_timeseries_mids:
            obs_times = obs_timeseries_mids[emu_id]
            sim_times = sim_timeseries_mids[emu_id]
            for t in obs_times:
                if t in sim_times:
                    obs_mid = jnp.asarray(obs_times[t])
                    sim_mid = jnp.asarray(sim_times[t])
                    for i in range(len(obs_mid)):
                        obs_val = obs_mid[i]
                        sim_val = sim_mid[i]
                        if obs_val > 0 and sim_val > 0:
                            log_obs = jnp.log(obs_val)
                            log_sim = jnp.log(sim_val)
                            residual = (log_obs - log_sim) / mid_std
                            log_likelihood["timeseries_mids"] += -0.5 * residual**2 - jnp.log(mid_std * jnp.sqrt(2 * jnp.pi))
                            log_likelihood["timeseries_mids"] += -jnp.log(obs_val)

    # Total log-likelihood
    log_likelihood["total"] = (
        log_likelihood["fluxes"]
        + log_likelihood["concentrations"]
        + log_likelihood["ss_mids"]
        + log_likelihood["timeseries_mids"]
    )

    return log_likelihood


# Parameter names that will be sampled (independent rate constants only)
# k2f and k_pq_r are computed from Wegscheider thermodynamic constraints
# A_e, Q_e, P_e, E_T are fixed (experimental conditions)
MCMC_PARAM_NAMES = [
    "k2r", "k3f", "k3r", "k4f", "k4r", "k5f", "k5r",
    "k_A_ex", "k_Q_ex", "k_P_ex", "k_pq_f"
]


def params_array_to_dict(param_array, fixed_params):
    """Convert parameter array to full parameter dictionary."""
    params = fixed_params.copy()
    for i, name in enumerate(MCMC_PARAM_NAMES):
        params[name] = param_array[i]
    return params


def params_dict_to_array(params):
    """Extract MCMC parameters from dictionary to array."""
    return jnp.array([params[name] for name in MCMC_PARAM_NAMES])


def generate_reference_data_single_condition(params, initial):
    """Generate reference observations from a parameter set for a single condition.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run isotopomer_ode from that steady state (all unlabeled) to get MIDs

    The fluxes and concentrations come from enzyme_ode (step 1).
    The MIDs come from isotopomer_ode (step 2) with A_e = A_11 (fully labeled).

    Parameters
    ----------
    params : dict
        Full parameter dictionary
    initial : dict
        Initial conditions

    Returns
    -------
    dict
        Reference data containing fluxes, concentrations, ss_mids
    """
    # Apply constraints to get updated params
    params = compute_wegscheider_constraints(params)

    initial_array = jnp.array([
        initial["A"], initial["Q"], initial["P"],
        initial["EA"], initial["EPQ"], initial["EP"]
    ])

    # Step 1: Run enzyme_ode to get metabolic steady state
    enzyme_sol = simulate_enzyme_to_steady_state_jax(params, initial_array)
    enzyme_y_ss = enzyme_sol.ys[0] if enzyme_sol.ys.ndim > 1 else enzyme_sol.ys

    # Extract concentrations from enzyme_ode steady state
    A, Q, P = float(enzyme_y_ss[0]), float(enzyme_y_ss[1]), float(enzyme_y_ss[2])
    EA, EPQ, EP = float(enzyme_y_ss[3]), float(enzyme_y_ss[4]), float(enzyme_y_ss[5])
    E = params["E_T"] - EA - EPQ - EP

    concs = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}

    # Compute net fluxes at steady state from enzyme_ode
    v_A_ex = params["k_A_ex"] * (params["A_e"] - A)
    v_Q_ex = params["k_Q_ex"] * (params["Q_e"] - Q)
    v_P_ex = params["k_P_ex"] * (params["P_e"] - P)

    net_rates = {
        "v_A_ex": v_A_ex,
        "v2": A * E * params["k2f"] - EA * params["k2r"],
        "v3": EA * params["k3f"] - EPQ * params["k3r"],
        "v4": EPQ * params["k4f"] - Q * EP * params["k4r"],
        "v5": EP * params["k5f"] - E * P * params["k5r"],
        "v_Q_ex": v_Q_ex,
        "v_P_ex": v_P_ex,
        "v_pq": P * params["k_pq_f"] - Q * params["k_pq_r"],
        "v_net": v_A_ex,  # Net flux through the system at steady state
    }

    # Step 2: Run steady-state EMU simulation from enzyme_ode steady state
    # Use the same SS solver as the likelihood function for consistency
    network, precomputed = get_pure_jax_emu_network()
    ss_mids_jax = compute_mids_pure_jax_emu_ss(concs, params, network, precomputed)

    # Convert JAX arrays to numpy for compatibility
    ss_mids_np = {
        "A_12": np.array(ss_mids_jax["A_12"]),
        "Q_1": np.array(ss_mids_jax["Q_1"]),
        "P_1": np.array(ss_mids_jax["P_1"]),
    }

    return {
        "fluxes": net_rates,
        "concentrations": concs,
        "ss_mids": ss_mids_np,
        "enzyme_steady_state": np.array(enzyme_y_ss),  # For initializing isotopomer sim
    }


def generate_reference_data_multi_condition(base_params, initial, conditions):
    """Generate reference data for multiple experimental conditions.

    Parameters
    ----------
    base_params : dict
        Base parameter dictionary (A_e, E_T will be overridden per condition)
    initial : dict
        Initial conditions (same across all conditions)
    conditions : list of dict
        Experimental conditions with keys 'A_e', 'E_T'

    Returns
    -------
    list of dict
        Reference datasets, one per condition
    """
    ref_data_list = []
    for cond in conditions:
        params = base_params.copy()
        params["A_e"] = cond["A_e"]
        params["E_T"] = cond["E_T"]
        ref_data = generate_reference_data_single_condition(params, initial)
        ref_data_list.append(ref_data)
    return ref_data_list


def compute_condition_likelihood(
    A_e,
    E_T,
    ref_flux,
    ref_concs,
    ref_mids,
    param_array,
    Q_e,
    P_e,
    initial_array,
):
    """Compute log-likelihood for a single experimental condition.

    Two-step process:
    1. Run enzyme_ode to get steady-state fluxes and concentrations
    2. Run isotopomer_ode from that steady state (all unlabeled) to get MIDs

    Fluxes and concentrations are from enzyme_ode (step 1).
    MIDs are from isotopomer_ode (step 2) with A_e = A_11 (fully labeled).

    Parameters
    ----------
    A_e : float
        External A concentration for this condition
    E_T : float
        Total enzyme concentration for this condition
    ref_flux : float
        Reference net flux value (v_net at steady state)
    ref_concs : jnp.array
        Reference concentration values (3,): [A, Q, P]
    ref_mids : jnp.array
        Reference MID values (7,): measurable metabolites [A_12(3), Q_1(2), P_1(2)]
    param_array : jnp.array
        Rate constants (11,): [k2r, k3f, k3r, k4f, k4r, k5f, k5r, k_A_ex, k_Q_ex, k_P_ex, k_pq_f]
    Q_e : float
        External Q concentration (fixed)
    P_e : float
        External P concentration (fixed)
    initial_array : jnp.array
        Initial conditions (6,)

    Returns
    -------
    float
        Log-likelihood for this condition
    """
    # Build parameter dict for ODE (matching MCMC_PARAM_NAMES order)
    params = {
        "k2r": param_array[0],
        "k3f": param_array[1],
        "k3r": param_array[2],
        "k4f": param_array[3],
        "k4r": param_array[4],
        "k5f": param_array[5],
        "k5r": param_array[6],
        "k_A_ex": param_array[7],
        "k_Q_ex": param_array[8],
        "k_P_ex": param_array[9],
        "k_pq_f": param_array[10],
        "A_e": A_e,
        "Q_e": Q_e,
        "P_e": P_e,
        "E_T": E_T,
    }

    # Compute dependent rate constants from Wegscheider constraints
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

    # Check if enzyme solver failed
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    # Extract concentrations from enzyme_ode steady state (for flux/conc likelihood)
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]

    # Measurable concentrations only (A, Q, P)
    sim_concs = jnp.array([A, Q, P])

    # Compute net flux from enzyme_ode steady state
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1)
    flux_std = 0.1
    flux_residual = (ref_flux - sim_flux) / flux_std
    ll_flux = -0.5 * flux_residual**2 - jnp.log(flux_std * jnp.sqrt(2 * jnp.pi))

    # Log-likelihood for concentrations (lognormal distribution, std=0.1)
    conc_std = 0.1
    log_ref_concs = jnp.log(jnp.maximum(ref_concs, 1e-10))
    log_sim_concs = jnp.log(jnp.maximum(sim_concs, 1e-10))
    conc_residuals = (log_ref_concs - log_sim_concs) / conc_std
    ll_concs = jnp.sum(
        -0.5 * conc_residuals**2
        - jnp.log(conc_std * jnp.sqrt(2 * jnp.pi))
    )

    # Step 2: Run isotopomer_ode from enzyme_ode steady state (all unlabeled)
    iso_sol = simulate_isotopomers_to_steady_state(params, enzyme_y_ss)

    # Check if isotopomer solver failed
    iso_failed = jnp.any(jnp.isnan(iso_sol.ys)) | (iso_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    # Extract steady-state isotopomer state
    iso_y_ss = iso_sol.ys[0] if iso_sol.ys.ndim > 1 else iso_sol.ys

    # Compute simulated MIDs from isotopomer state (measurable metabolites only)
    sim_mids_dict = compute_mids_jax(iso_y_ss)

    sim_mids_array = jnp.concatenate([
        jnp.array(sim_mids_dict["A_12"]),
        jnp.array(sim_mids_dict["Q_1"]),
        jnp.array(sim_mids_dict["P_1"]),
    ])

    # Log-likelihood for MIDs (lognormal distribution, std=0.01)
    mid_std = 0.01
    log_ref_mids = jnp.log(jnp.maximum(ref_mids, 1e-10))
    log_sim_mids = jnp.log(jnp.maximum(sim_mids_array, 1e-10))
    mid_residuals = (log_ref_mids - log_sim_mids) / mid_std
    ll_mids = jnp.sum(
        -0.5 * mid_residuals**2
        - jnp.log(mid_std * jnp.sqrt(2 * jnp.pi))
    )

    ll_condition = ll_flux + ll_concs + ll_mids

    # Return -inf if either solver failed
    solver_failed = enzyme_failed | iso_failed
    return jnp.where(solver_failed, -jnp.inf, ll_condition)


def compute_fluxes_array_pure_jax(concs, params, network):
    """Compute flux array for pure JAX EMU simulation.

    Computes forward/backward fluxes in the order expected by the
    serialized EMU network structure.

    Parameters
    ----------
    concs : dict or arrays
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, E
    params : dict
        Rate constants
    network : dict
        Serialized EMU network structure

    Returns
    -------
    jnp.ndarray
        Flux array in order matching network["flux_ids_all"]
    """
    A = concs["A"]
    Q = concs["Q"]
    P = concs["P"]
    EA = concs["EA"]
    EPQ = concs["EPQ"]
    EP = concs["EP"]
    E = concs["E"]

    # Compute all fluxes (matching reactions.tsv order)
    fluxes_dict = {
        "v1": params["k_A_ex"] * params["A_e"],  # A_ext -> A (input)
        "v1_out": params["k_A_ex"] * A,  # A -> A_sink
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

    # Create array in network order
    return jnp.array([fluxes_dict.get(fid, 0.0) for fid in network["flux_ids_all"]])


def compute_concs_array_pure_jax(concs, network):
    """Compute concentration array for pure JAX EMU simulation.

    Parameters
    ----------
    concs : dict
        Concentration values
    network : dict
        Serialized EMU network structure

    Returns
    -------
    jnp.ndarray
        Concentration array in order matching network["metab_ids_all"]
    """
    return jnp.array([concs.get(mid, 1.0) for mid in network["metab_ids_all"]])


def compute_mids_pure_jax_emu(concs_dict, params, network, timepoints, precomputed=None):
    """Compute MIDs using pure JAX EMU simulation (instationary).

    This is fully JAX-traceable and compatible with gradient-based MCMC.

    Parameters
    ----------
    concs_dict : dict
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, E
    params : dict
        Rate constants including k2f, k_pq_r
    network : dict
        Serialized EMU network structure
    timepoints : jnp.ndarray
        Simulation timepoints (including t=0)
    precomputed : dict, optional
        Precomputed matrix structure arrays for JAX tracing

    Returns
    -------
    dict
        {emu_id: mdv_array} for all product EMUs at final timepoint
    """
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)
    concs_array = compute_concs_array_pure_jax(concs_dict, network)

    return simulate_inst_mdvs_pure_jax(fluxes_array, concs_array, timepoints, network, precomputed)


def compute_mids_pure_jax_emu_ss(concs_dict, params, network, precomputed=None):
    """Compute MIDs using pure JAX steady-state EMU simulation.

    This is fully JAX-traceable and compatible with gradient-based MCMC.
    Simpler and faster than instationary simulation since it directly solves
    the steady-state equation X = pinv(A) @ B @ Y without time integration.

    Parameters
    ----------
    concs_dict : dict
        Steady-state concentrations with keys A, Q, P, EA, EPQ, EP, E
    params : dict
        Rate constants including k2f, k_pq_r
    network : dict
        Serialized EMU network structure
    precomputed : dict, optional
        Precomputed matrix structure arrays for JAX tracing

    Returns
    -------
    dict
        {emu_id: mdv_array} for all product EMUs at isotopic steady state
    """
    fluxes_array = compute_fluxes_array_pure_jax(concs_dict, params, network)

    return simulate_ss_mdvs_pure_jax(fluxes_array, network, precomputed)


def compute_condition_likelihood_emu(
    A_e,
    E_T,
    ref_flux,
    ref_concs,
    ref_mids,
    param_array,
    Q_e,
    P_e,
    initial_array,
    emu_network,
    precomputed,
):
    """Compute log-likelihood using pure JAX steady-state EMU simulation.

    This version uses the pure JAX steady-state EMU simulation instead of
    instationary/isotopomer ODE, enabling gradient-based MCMC with simpler
    and faster computation.

    Parameters
    ----------
    A_e : float
        External A concentration for this condition
    E_T : float
        Total enzyme concentration for this condition
    ref_flux : float
        Reference net flux value (v_net at steady state)
    ref_concs : jnp.array
        Reference concentration values (3,): [A, Q, P]
    ref_mids : jnp.array
        Reference MID values (7,): measurable metabolites [A_12(3), Q_1(2), P_1(2)]
    param_array : jnp.array
        Rate constants (11,): [k2r, k3f, k3r, k4f, k4r, k5f, k5r, k_A_ex, k_Q_ex, k_P_ex, k_pq_f]
    Q_e : float
        External Q concentration (fixed)
    P_e : float
        External P concentration (fixed)
    initial_array : jnp.array
        Initial conditions (6,)
    emu_network : dict
        Serialized EMU network structure
    precomputed : dict
        Precomputed matrix structure arrays for JAX tracing

    Returns
    -------
    float
        Log-likelihood for this condition
    """
    # Build parameter dict for ODE (matching MCMC_PARAM_NAMES order)
    params = {
        "k2r": param_array[0],
        "k3f": param_array[1],
        "k3r": param_array[2],
        "k4f": param_array[3],
        "k4r": param_array[4],
        "k5f": param_array[5],
        "k5r": param_array[6],
        "k_A_ex": param_array[7],
        "k_Q_ex": param_array[8],
        "k_P_ex": param_array[9],
        "k_pq_f": param_array[10],
        "A_e": A_e,
        "Q_e": Q_e,
        "P_e": P_e,
        "E_T": E_T,
    }

    # Compute dependent rate constants from Wegscheider constraints
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

    # Check if enzyme solver failed
    enzyme_failed = jnp.any(jnp.isnan(enzyme_sol.ys)) | (enzyme_sol.result != diffrax.RESULTS.discrete_terminating_event_occurred)

    # Extract concentrations from enzyme_ode steady state
    A = enzyme_y_ss[0]
    Q = enzyme_y_ss[1]
    P = enzyme_y_ss[2]
    EA = enzyme_y_ss[3]
    EPQ = enzyme_y_ss[4]
    EP = enzyme_y_ss[5]
    E = params["E_T"] - EA - EPQ - EP

    # Measurable concentrations only (A, Q, P)
    sim_concs = jnp.array([A, Q, P])

    # Compute net flux from enzyme_ode steady state
    sim_flux = params["k_A_ex"] * (A_e - A)

    # Log-likelihood for flux (normal distribution, std=0.1)
    flux_std = 0.1
    flux_residual = (ref_flux - sim_flux) / flux_std
    ll_flux = -0.5 * flux_residual**2 - jnp.log(flux_std * jnp.sqrt(2 * jnp.pi))

    # Log-likelihood for concentrations (lognormal distribution, std=0.1)
    conc_std = 0.1
    log_ref_concs = jnp.log(jnp.maximum(ref_concs, 1e-10))
    log_sim_concs = jnp.log(jnp.maximum(sim_concs, 1e-10))
    conc_residuals = (log_ref_concs - log_sim_concs) / conc_std
    ll_concs = jnp.sum(
        -0.5 * conc_residuals**2
        - jnp.log(conc_std * jnp.sqrt(2 * jnp.pi))
    )

    # Step 2: Run pure JAX steady-state EMU simulation from enzyme_ode steady state
    concs_dict = {"A": A, "Q": Q, "P": P, "EA": EA, "EPQ": EPQ, "EP": EP, "E": E}
    sim_mids_dict = compute_mids_pure_jax_emu_ss(concs_dict, params, emu_network, precomputed)

    # Extract MIDs for measurable metabolites only (A, Q, P)
    sim_mids_array = jnp.concatenate([
        sim_mids_dict["A_12"],
        sim_mids_dict["Q_1"],
        sim_mids_dict["P_1"],
    ])

    # Log-likelihood for MIDs (lognormal distribution, std=0.01)
    mid_std = 0.01
    log_ref_mids = jnp.log(jnp.maximum(ref_mids, 1e-10))
    log_sim_mids = jnp.log(jnp.maximum(sim_mids_array, 1e-10))
    mid_residuals = (log_ref_mids - log_sim_mids) / mid_std
    ll_mids = jnp.sum(
        -0.5 * mid_residuals**2
        - jnp.log(mid_std * jnp.sqrt(2 * jnp.pi))
    )

    ll_condition = ll_flux + ll_concs + ll_mids

    # Return -inf if enzyme solver failed
    return jnp.where(enzyme_failed, -jnp.inf, ll_condition)


def compute_log_posterior_jax(
    log_param_array,
    fixed_params_constant,
    fixed_params_varying,
    initial_array,
    ref_flux_array,
    ref_concs_arrays,
    ref_mids_arrays,
    prior_mean,
    prior_std,
):
    """JAX-compatible log-posterior for multi-condition MCMC sampling.

    Parameters
    ----------
    log_param_array : jnp.array
        Array of log-transformed MCMC parameters (log rate constants)
    fixed_params_constant : jnp.array
        Constant fixed parameters [Q_e, P_e]
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (num_conditions, 2) [[A_e, E_T], ...]
    initial_array : jnp.array
        Initial conditions [A, Q, P, EA, EPQ, EP]
    ref_flux_array : jnp.array
        Reference net flux values (num_conditions,)
    ref_concs_arrays : jnp.array
        Reference concentration values (num_conditions, 3): [A, Q, P]
    ref_mids_arrays : jnp.array
        Reference MID values (num_conditions, 7): measurable metabolites only
    prior_mean : jnp.array
        Prior mean for each parameter (log-scale)
    prior_std : float
        Prior standard deviation (log-scale)

    Returns
    -------
    float
        Log-posterior value
    """
    # Transform from log-space to natural scale
    param_array = jnp.exp(log_param_array)

    # Normal prior on log-parameters (no Jacobian needed since we sample in log-space)
    log_prior = jnp.sum(
        -0.5 * ((log_param_array - prior_mean) / prior_std) ** 2
        - jnp.log(prior_std * jnp.sqrt(2 * jnp.pi))
    )

    # Extract constant parameters
    Q_e = fixed_params_constant[0]
    P_e = fixed_params_constant[1]

    # Use vmap to compute likelihood for all conditions in parallel
    condition_lls = jax.vmap(
        lambda A_e, E_T, rf, rc, rm: compute_condition_likelihood(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e, initial_array
        )
    )(
        fixed_params_varying[:, 0],  # A_e values
        fixed_params_varying[:, 1],  # E_T values
        ref_flux_array,
        ref_concs_arrays,
        ref_mids_arrays,
    )

    # Sum likelihoods across conditions
    total_log_likelihood = jnp.sum(condition_lls)

    log_posterior = log_prior + total_log_likelihood
    return log_posterior


def compute_log_posterior_jax_emu(
    log_param_array,
    fixed_params_constant,
    fixed_params_varying,
    initial_array,
    ref_flux_array,
    ref_concs_arrays,
    ref_mids_arrays,
    prior_mean,
    prior_std,
    emu_network,
    precomputed,
):
    """JAX-compatible log-posterior using pure JAX steady-state EMU simulation.

    This version uses the pure JAX steady-state EMU simulation instead of
    instationary/isotopomer ODE, enabling gradient-based MCMC with full JAX
    tracing and simpler/faster computation.

    Parameters
    ----------
    log_param_array : jnp.array
        Array of log-transformed MCMC parameters (log rate constants)
    fixed_params_constant : jnp.array
        Constant fixed parameters [Q_e, P_e]
    fixed_params_varying : jnp.array
        Varying fixed parameters per condition (num_conditions, 2) [[A_e, E_T], ...]
    initial_array : jnp.array
        Initial conditions [A, Q, P, EA, EPQ, EP]
    ref_flux_array : jnp.array
        Reference net flux values (num_conditions,)
    ref_concs_arrays : jnp.array
        Reference concentration values (num_conditions, 3): [A, Q, P]
    ref_mids_arrays : jnp.array
        Reference MID values (num_conditions, 7) - measurable metabolites only [A_12(3), Q_1(2), P_1(2)]
    prior_mean : jnp.array
        Prior mean for each parameter (log-scale)
    prior_std : float
        Prior standard deviation (log-scale)
    emu_network : dict
        Serialized EMU network structure
    precomputed : dict
        Precomputed matrix structure arrays for JAX tracing

    Returns
    -------
    float
        Log-posterior value
    """
    # Transform from log-space to natural scale
    param_array = jnp.exp(log_param_array)

    # Normal prior on log-parameters (no Jacobian needed since we sample in log-space)
    log_prior = jnp.sum(
        -0.5 * ((log_param_array - prior_mean) / prior_std) ** 2
        - jnp.log(prior_std * jnp.sqrt(2 * jnp.pi))
    )

    # Extract constant parameters
    Q_e = fixed_params_constant[0]
    P_e = fixed_params_constant[1]

    # Use vmap to compute likelihood for all conditions in parallel
    condition_lls = jax.vmap(
        lambda A_e, E_T, rf, rc, rm: compute_condition_likelihood_emu(
            A_e, E_T, rf, rc, rm, param_array, Q_e, P_e, initial_array,
            emu_network, precomputed
        )
    )(
        fixed_params_varying[:, 0],  # A_e values
        fixed_params_varying[:, 1],  # E_T values
        ref_flux_array,
        ref_concs_arrays,
        ref_mids_arrays,
    )

    # Sum likelihoods across conditions
    total_log_likelihood = jnp.sum(condition_lls)

    log_posterior = log_prior + total_log_likelihood
    return log_posterior


def run_mcmc(
    ref_params,
    initial,
    conditions=None,
    n_warmup=200,
    n_samples=200,
    n_chains=1,
    seed=42,
    use_pure_jax_emu=True,
):
    """Run MCMC sampling using blackjax NUTS with multiple experimental conditions.

    Parameters
    ----------
    ref_params : dict
        Reference parameter set (used to generate synthetic data)
    initial : dict
        Initial conditions for ODE
    conditions : list of dict, optional
        Experimental conditions with keys 'A_e', 'E_T'. Defaults to EXPERIMENTAL_CONDITIONS.
    n_warmup : int
        Number of warmup iterations
    n_samples : int
        Number of samples to collect
    n_chains : int
        Number of independent chains to run in parallel
    seed : int
        Random seed
    use_pure_jax_emu : bool
        If True (default), use pure JAX EMU simulation for isotope labeling.
        This enables gradient-based MCMC with full JAX tracing.
        If False, use isotopomer ODE (legacy, slower).

    Returns
    -------
    dict
        MCMC results containing samples and diagnostics
    """
    if conditions is None:
        conditions = EXPERIMENTAL_CONDITIONS

    num_conditions = len(conditions)

    method = "pure JAX steady-state EMU" if use_pure_jax_emu else "isotopomer ODE"
    print("=" * 80)
    print(f"Running MCMC with blackjax NUTS ({num_conditions} conditions, {n_chains} chains, {method})")
    print("=" * 80)
    print(f"\nExperimental conditions:")
    for i, cond in enumerate(conditions):
        print(f"  Condition {i+1}: E_T={cond['E_T']:.4f}, A_e={cond['A_e']:.4f}")
    sys.stdout.flush()

    # Split fixed parameters into constant and varying
    # Constant: Q_e, P_e (same across all conditions)
    # Varying: A_e, E_T (different per condition)
    fixed_params_constant = jnp.array([
        ref_params["Q_e"],
        ref_params["P_e"],
    ])

    fixed_params_varying = jnp.array([
        [cond["A_e"], cond["E_T"]] for cond in conditions
    ])  # Shape: (num_conditions, 2)

    initial_array = jnp.array([
        initial["A"],
        initial["Q"],
        initial["P"],
        initial["EA"],
        initial["EPQ"],
        initial["EP"],
    ])

    # Generate reference data for all conditions
    print("\nGenerating reference data for all conditions...")
    sys.stdout.flush()
    ref_data_list = generate_reference_data_multi_condition(
        ref_params, initial, conditions
    )

    # Stack reference data into arrays (measurable quantities only)
    # Net flux (scalar per condition)
    ref_flux_array = jnp.array([
        rd["fluxes"]["v_net"]
        for rd in ref_data_list
    ])  # Shape: (num_conditions,)

    # Measurable concentrations: A, Q, P only
    ref_concs_arrays = jnp.array([
        [
            rd["concentrations"]["A"],
            rd["concentrations"]["Q"],
            rd["concentrations"]["P"],
        ]
        for rd in ref_data_list
    ])  # Shape: (num_conditions, 3)

    # MIDs for measurable metabolites only (A, Q, P - not enzyme intermediates)
    ref_mids_arrays = jnp.array([
        np.concatenate([
            np.array(rd["ss_mids"]["A_12"]),
            np.array(rd["ss_mids"]["Q_1"]),
            np.array(rd["ss_mids"]["P_1"]),
        ])
        for rd in ref_data_list
    ])  # Shape: (num_conditions, 7) - measurable metabolites only

    # Prior: log-normal centered on reference values
    param_array_ref = params_dict_to_array(ref_params)
    prior_mean = jnp.log(param_array_ref)
    prior_std = 0.5  # Allow reasonable variation

    # Log-posterior function (JAX-compatible, takes log-transformed params)
    if use_pure_jax_emu:
        # Load serialized EMU network and precomputed structure
        emu_network, precomputed = get_pure_jax_emu_network()

        @jax.jit
        def logdensity_fn(log_param_array):
            return compute_log_posterior_jax_emu(
                log_param_array,
                fixed_params_constant,
                fixed_params_varying,
                initial_array,
                ref_flux_array,
                ref_concs_arrays,
                ref_mids_arrays,
                prior_mean,
                prior_std,
                emu_network,
                precomputed,
            )
    else:
        @jax.jit
        def logdensity_fn(log_param_array):
            return compute_log_posterior_jax(
                log_param_array,
                fixed_params_constant,
                fixed_params_varying,
                initial_array,
                ref_flux_array,
                ref_concs_arrays,
                ref_mids_arrays,
                prior_mean,
                prior_std,
            )

    # Initialize at reference values (in log-space)
    rng_key = jax.random.PRNGKey(seed)
    initial_position = jnp.log(param_array_ref)

    print(f"Initial position (log-space): {initial_position}")
    print(f"Initial position (natural): {param_array_ref}")
    print(f"Initial log-posterior: {logdensity_fn(initial_position)}")
    sys.stdout.flush()

    # Setup NUTS sampler with window adaptation
    print(f"\nRunning {n_warmup} warmup iterations...")
    sys.stdout.flush()

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        num_steps=n_warmup,
        progress_bar=True,
    )

    rng_key, warmup_key = jax.random.split(rng_key)
    state, kernel, *_ = warmup.run(warmup_key, initial_position)

    print(f"Warmup complete.")
    sys.stdout.flush()

    # Sampling with multiple chains
    print(f"\nCollecting {n_samples} samples from {n_chains} chains...")
    sys.stdout.flush()

    # JIT-compile the kernel step for efficiency
    @jax.jit
    def one_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, info = kernel(subkey, state)
        return (state, key), (state.position, -state.potential_energy)

    # Run each chain
    all_chain_samples = []
    all_chain_log_probs = []

    for chain_idx in range(n_chains):
        rng_key, chain_key = jax.random.split(rng_key)

        # Initialize chain state (use warmup state as starting point)
        chain_state = state

        # Run sampling using scan for efficiency
        (final_state, _), (chain_positions, chain_log_probs) = jax.lax.scan(
            one_step, (chain_state, chain_key), None, length=n_samples
        )

        all_chain_samples.append(chain_positions)
        all_chain_log_probs.append(chain_log_probs)

        print(f"  Chain {chain_idx + 1}/{n_chains} complete")
        sys.stdout.flush()

    # Stack samples: shape (n_chains, n_samples, n_params)
    log_samples = jnp.stack(all_chain_samples)
    log_probs = jnp.stack(all_chain_log_probs)

    # Transform samples back to natural scale
    samples = jnp.exp(log_samples)

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("MCMC Results Summary")
    print("=" * 80)

    # Flatten chains for summary: shape (n_chains * n_samples, n_params)
    samples_flat = samples.reshape(-1, samples.shape[-1])
    log_samples_flat = log_samples.reshape(-1, log_samples.shape[-1])
    log_probs_flat = log_probs.reshape(-1)

    results = {
        "samples": samples,  # Natural scale, shape (n_chains, n_samples, n_params)
        "samples_flat": samples_flat,  # Flattened, shape (n_chains * n_samples, n_params)
        "log_samples": log_samples,  # Log scale
        "log_probs": log_probs,
        "param_names": MCMC_PARAM_NAMES,
        "reference": param_array_ref,
        "n_chains": n_chains,
        "n_samples": n_samples,
    }

    print(f"\nParameter estimates (mean ± std, natural scale, {n_chains} chains × {n_samples} samples):")
    for i, name in enumerate(MCMC_PARAM_NAMES):
        mean = float(jnp.mean(samples_flat[:, i]))
        std = float(jnp.std(samples_flat[:, i]))
        true_val = float(param_array_ref[i])
        print(f"  {name:10s}: {mean:8.4f} ± {std:6.4f}  (true: {true_val:.4f})")

    sys.stdout.flush()

    return results


def print_results(concs, net_rates, fluxes, ss_mdvs):
    """Print simulation results."""
    print("\n" + "=" * 80)
    print("Steady-State Concentrations")
    print("=" * 80)
    for met_id, conc in concs.items():
        print(f"  {met_id}: {conc:.6f}")

    enzyme_total = concs["E"] + concs["EA"] + concs["EPQ"] + concs["EP"]
    print(f"\n  Enzyme conservation check: E + EA + EPQ + EP = {enzyme_total:.6f}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Steady-State Net Fluxes")
    print("=" * 80)
    for rate_id, rate_val in net_rates.items():
        print(f"  {rate_id}: {rate_val:.6f}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Forward/Backward Fluxes")
    print("=" * 80)
    for flux_id, flux_val in fluxes.items():
        print(f"  {flux_id}: {flux_val:.6f}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Steady-State Labeling (MID)")
    print("=" * 80)
    for emu_id, mdv in ss_mdvs.items():
        print(f"  {emu_id}: {mdv}")
    sys.stdout.flush()


def main():
    print("=" * 80)
    print("Enzyme Kinetics ODE Simulation with Isotope Labeling Analysis")
    print("=" * 80)
    sys.stdout.flush()

    output_dir = create_output_directory()
    params = DEFAULT_PARAMS.copy()
    initial = DEFAULT_INITIAL.copy()

    # Step 1: Apply thermodynamic constraints
    print("\n" + "=" * 80)
    print("Step 1: Applying Wegscheider Thermodynamic Constraints")
    print("=" * 80)
    print(f"  Formation energies: {FORMATION_ENERGIES}")
    print(f"  Temperature: {TEMPERATURE} K")

    # Compute Gibbs energies for display
    RT = R_GAS * TEMPERATURE
    delta_G_enzyme = FORMATION_ENERGIES["P"] + FORMATION_ENERGIES["Q"] - FORMATION_ENERGIES["A"]
    delta_G_pq = FORMATION_ENERGIES["Q"] - FORMATION_ENERGIES["P"]
    K_enzyme = np.exp(-delta_G_enzyme / RT)
    K_pq = np.exp(-delta_G_pq / RT)

    print(f"  ΔG°(A → P + Q) = {delta_G_enzyme:.2f} kJ/mol  →  K_eq = {K_enzyme:.4f}")
    print(f"  ΔG°(P → Q) = {delta_G_pq:.2f} kJ/mol  →  K_pq = {K_pq:.4f}")

    # Apply constraints to compute dependent rate constants
    params = compute_wegscheider_constraints(params)
    print(f"\n  Computed dependent rate constants:")
    print(f"    k2f = {params['k2f']:.6f}")
    print(f"    k_pq_r = {params['k_pq_r']:.6f}")
    sys.stdout.flush()

    # Step 2: Simulate ODE to steady state
    print("\n" + "=" * 80)
    print("Step 2: Simulating ODE System to Steady State")
    print("=" * 80)
    print(f"  Parameters: {params}")
    print(f"  Initial conditions: {initial}")
    sys.stdout.flush()

    sol = simulate_ode_to_steady_state(params, initial)

    if sol.result not in (diffrax.RESULTS.successful, diffrax.RESULTS.discrete_terminating_event_occurred):
        print(f"  ERROR: ODE integration failed: {sol.result}")
        sys.exit(1)

    print(f"  ODE integration successful (diffrax, steady-state event)")
    sys.stdout.flush()

    # Step 3: Extract steady state
    print("\n" + "=" * 80)
    print("Step 3: Extracting Steady-State Values")
    print("=" * 80)
    sys.stdout.flush()

    concs, net_rates = get_steady_state(sol, params)
    fluxes = compute_forward_backward_fluxes(concs, params)

    # Step 4: Run EMU simulation to steady state
    print("\n" + "=" * 80)
    print("Step 4: Running EMU Simulation to Steady State")
    print("=" * 80)
    sys.stdout.flush()

    ref_data = generate_reference_data_single_condition(params, initial)
    ss_mdvs = ref_data["ss_mids"]

    print(f"  EMU simulation complete")
    print(f"  Simulated EMUs: {list(ss_mdvs.keys())}")
    sys.stdout.flush()

    # Print results
    print_results(concs, net_rates, fluxes, ss_mdvs)

    # Step 5: Run MCMC parameter estimation
    print("\n" + "=" * 80)
    print("Step 5: Running MCMC Parameter Estimation")
    print("=" * 80)
    sys.stdout.flush()

    mcmc_results = run_mcmc(
        ref_params=DEFAULT_PARAMS,
        initial=DEFAULT_INITIAL,
    )

    # Save MCMC results
    np.savez(
        output_dir / "mcmc_results.npz",
        samples=mcmc_results["samples"],
        log_samples=mcmc_results["log_samples"],
        log_probs=mcmc_results["log_probs"],
        param_names=mcmc_results["param_names"],
        reference=mcmc_results["reference"],
    )
    print(f"\nResults saved to {output_dir / 'mcmc_results.npz'}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Simulation Complete!")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

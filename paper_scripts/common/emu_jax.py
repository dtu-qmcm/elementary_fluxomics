#!/usr/bin/env python3
"""JAX-compatible EMU MDV simulation (rewrite of freeflux _calculate_inst_MDVs).

This module provides JAX-compatible implementations of:
1. MDV convolution (gen_conv equivalent)
2. Matrix exponential integration for INST simulation
3. Full INST MDV simulation

The goal is to enable autodiff through isotope labeling simulations while
leveraging freeflux for EMU decomposition and matrix construction.

Mathematical formulation (from freeflux _calculate_inst_MDVs):
    dX/dt = M⁻¹·A·X + M⁻¹·B·Y

    Where:
    - X = metabolite MDVs (product EMUs)
    - Y = source MDVs (substrate EMUs)
    - A = flux matrix (from EMU adjacency)
    - B = input matrix
    - M = concentration diagonal matrix

    Solution using matrix exponential:
    F = M⁻¹ @ A
    Phi = expm(F * dt)                    # State transition matrix
    Gamma = (Phi - I) @ F^-1              # First integration matrix
    Omega = (Gamma/dt - I) @ F^-1         # Second integration matrix

    X_t1 = Phi @ X_t0 - Gamma @ G_t0 - Omega @ (G_t1 - G_t0)
"""

from collections.abc import Iterable
from functools import reduce
from pathlib import Path
from typing import Any

import jax
# Enable float64 precision for JAX
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import toml
from freeflux import Model


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

# Gas constant in kJ/(mol*K)
R_GAS = 8.314e-3


def load_fluxes_from_toml(flux_filepath: Path) -> dict[str, float]:
    """Load net fluxes from TOML file and convert units.

    Parameters
    ----------
    flux_filepath : Path
        Path to TOML file with flux values

    Returns
    -------
    dict
        {flux_id: value} in per-second units
    """
    data = toml.load(str(flux_filepath))
    fluxes = {}
    conversion_factor = 1000 / 3600  # mmol/gDW/h to per second

    for section in ["external_fluxes", "internal_fluxes"]:
        if section in data:
            for rxn_id, value in data[section].items():
                if rxn_id == "glc_uptake":
                    rxn_id = "glc_input"
                elif rxn_id == "pyr_secretion":
                    continue
                fluxes[rxn_id] = value * conversion_factor

    return fluxes


def load_thermo_data(filepath: Path, overrides: dict = None) -> tuple[dict, float]:
    """Load thermodynamic data from TOML file with optional overrides.

    Parameters
    ----------
    filepath : Path
        Path to thermodynamics TOML file
    overrides : dict, optional
        Override values for specific reactions

    Returns
    -------
    tuple
        (thermo_dict, temperature)
    """
    data = toml.load(str(filepath))
    temperature = data.get("conditions", {}).get("temperature", 298.15)

    thermo = {}
    for rxn_id, rxn_data in data.get("reactions", {}).items():
        thermo[rxn_id] = rxn_data.get("deltaG_prime", -5.0)

    if overrides:
        thermo.update(overrides)

    return thermo, temperature


def net_to_forward_backward(v_net: float, deltaG_prime: float, temperature: float) -> tuple[float, float]:
    """Convert net flux to forward and backward fluxes using Gibbs energy.

    Parameters
    ----------
    v_net : float
        Net flux value
    deltaG_prime : float
        Standard Gibbs energy of reaction (kJ/mol)
    temperature : float
        Temperature (K)

    Returns
    -------
    tuple
        (v_forward, v_backward)
    """
    exp_term = np.exp(-deltaG_prime / (R_GAS * temperature))

    if exp_term > 1e6:
        return abs(v_net), 0.0

    if abs(exp_term - 1) < 1e-6:
        v_f = abs(v_net) * 1000
        v_b = v_f - v_net
        return v_f, v_b

    v_f = v_net * exp_term / (exp_term - 1)
    v_b = v_net / (exp_term - 1)

    return v_f, v_b


def prepare_fluxes_for_model(
    model,
    net_fluxes: dict[str, float],
    thermo: dict[str, float],
    temperature: float,
) -> dict[str, float]:
    """Prepare flux dict with forward/backward values for reversible reactions.

    Parameters
    ----------
    model
        A freeflux Model object
    net_fluxes : dict
        Net flux values {reaction_id: value}
    thermo : dict
        Gibbs energy values {reaction_id: deltaG}
    temperature : float
        Temperature (K)

    Returns
    -------
    dict
        Flux values with forward/backward decomposition for reversible reactions
    """
    # Find reversible reactions
    reversible_rxns = {
        fid.rsplit("_", 1)[0] for fid in model.totalfluxids if fid.endswith("_f")
    }

    fluxes = {}
    for rxn_id, v_net in net_fluxes.items():
        deltaG = thermo.get(rxn_id, -5.0)

        if rxn_id in reversible_rxns:
            v_f, v_b = net_to_forward_backward(v_net, deltaG, temperature)
            fluxes[f"{rxn_id}_f"] = v_f
            fluxes[f"{rxn_id}_b"] = v_b
        else:
            fluxes[rxn_id] = v_net

    return fluxes


def mdv_convolve(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """Convolve two MDV arrays (JAX-compatible equivalent of freeflux gen_conv).

    This performs the discrete convolution of two MDV arrays, which represents
    the combined labeling pattern when two EMU fragments are joined.

    For MDVs of lengths n+1 and m+1 (representing n and m atoms respectively),
    the result has length n+m+1 (representing n+m atoms).

    Parameters
    ----------
    arr1 : jnp.ndarray
        First MDV array (length n+1 for n atoms)
    arr2 : jnp.ndarray
        Second MDV array (length m+1 for m atoms)

    Returns
    -------
    jnp.ndarray
        Convolved MDV array (length n+m+1)
    """
    # jnp.convolve computes the full discrete convolution
    # For MDVs with n+1 and m+1 elements, result has n+m+1 elements
    return jnp.convolve(arr1, arr2)


@jax.jit
def inst_mdv_step(
    X_t0: jnp.ndarray,
    Y_t0: jnp.ndarray,
    Y_t1: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    M: jnp.ndarray,
    delta_t: float,
) -> jnp.ndarray:
    """Single time step of INST MDV simulation (JAX-compatible).

    Implements the matrix exponential integration scheme from freeflux.

    Parameters
    ----------
    X_t0 : jnp.ndarray
        Previous metabolite MDVs (product EMUs), shape (n_products, n_mass_isotopomers)
    Y_t0 : jnp.ndarray
        Previous source MDVs (substrate EMUs), shape (n_sources, n_mass_isotopomers)
    Y_t1 : jnp.ndarray
        Current source MDVs (substrate EMUs), shape (n_sources, n_mass_isotopomers)
    A : jnp.ndarray
        Flux matrix from EMU adjacency, shape (n_products, n_products)
    B : jnp.ndarray
        Input matrix, shape (n_products, n_sources)
    M : jnp.ndarray
        Concentration diagonal matrix, shape (n_products, n_products)
    delta_t : float
        Time step

    Returns
    -------
    jnp.ndarray
        Updated metabolite MDVs at t1, shape (n_products, n_mass_isotopomers)
    """
    # Compute inverse of concentration matrix
    Minv = jnp.linalg.pinv(M)

    # Compute F = M⁻¹ @ A (effective rate matrix)
    F = Minv @ A

    # Compute pseudo-inverse of F
    Finv = jnp.linalg.pinv(F)

    # Identity matrix
    I = jnp.eye(F.shape[0])

    # State transition matrix: Φ = expm(F * Δt)
    Phi = jax.scipy.linalg.expm(F * delta_t)

    # First integration matrix: Γ = (Φ - I) @ F⁻¹
    Gamma = (Phi - I) @ Finv

    # Second integration matrix: Ω = (Γ/Δt - I) @ F⁻¹
    Omega = (Gamma / delta_t - I) @ Finv

    # Compute G vectors (input contributions)
    G_t0 = Minv @ B @ Y_t0
    G_t1 = Minv @ B @ Y_t1

    # Update formula: X_t1 = Φ @ X_t0 - Γ @ G_t0 - Ω @ (G_t1 - G_t0)
    X_t1 = Phi @ X_t0 - Gamma @ G_t0 - Omega @ (G_t1 - G_t0)

    return X_t1


def extract_emu_matrices(
    simulator,
    fluxes: dict[str, float],
    concentrations: dict[str, float],
) -> dict[int, dict[str, Any]]:
    """Extract pre-computed EMU matrices from a prepared freeflux simulator.

    The simulator must be prepared (isim.prepare() called) before calling this.

    Parameters
    ----------
    simulator
        A prepared freeflux INST simulator (from model.simulator("inst"))
    fluxes : dict
        Flux values {flux_id: value}
    concentrations : dict
        Concentration values {metabolite_id: value}

    Returns
    -------
    dict
        EMU size -> {
            "A": flux matrix (numpy),
            "B": input matrix (numpy),
            "M": concentration matrix (numpy),
            "product_emus": list of product EMU objects,
            "source_emus": list of source EMU objects/tuples,
            "flux_ids_A": flux IDs for A matrix,
            "flux_ids_B": flux IDs for B matrix,
            "metab_ids": metabolite IDs for M matrix,
        }
    """
    model = simulator.model
    matrices = {}

    for size in sorted(model.matrix_As):
        lambA, flux_ids_A, product_emus = model.matrix_As[size]
        lambB, flux_ids_B, source_emus = model.matrix_Bs[size]
        lambM, metab_ids = model.matrix_Ms[size]

        # Evaluate matrices with current flux/concentration values
        flux_values_A = [fluxes.get(fid, model.total_fluxes.get(fid, 0.0)) for fid in flux_ids_A]
        flux_values_B = [fluxes.get(fid, model.total_fluxes.get(fid, 0.0)) for fid in flux_ids_B]
        conc_values = [concentrations.get(mid, model.concentrations.get(mid, 1.0)) for mid in metab_ids]

        A = lambA(*flux_values_A)
        B = lambB(*flux_values_B)
        M = lambM(*conc_values)

        matrices[size] = {
            "A": np.array(A),
            "B": np.array(B),
            "M": np.array(M),
            "product_emus": product_emus,
            "source_emus": source_emus,
            "flux_ids_A": flux_ids_A,
            "flux_ids_B": flux_ids_B,
            "metab_ids": metab_ids,
        }

    return matrices


def get_initial_states(model) -> tuple[dict, dict]:
    """Get initial X and Y states from freeflux model.

    Parameters
    ----------
    model
        A prepared freeflux model (from simulator.model)

    Returns
    -------
    tuple
        (initial_X, initial_Y) where each is {size: array}
    """
    initial_X = {}
    initial_Y = {}

    for size in sorted(model.matrix_As):
        initial_X[size] = np.array(model.initial_matrix_Xs[size])
        initial_Y[size] = np.array(model.initial_matrix_Ys[size])

    return initial_X, initial_Y


def get_substrate_mdvs(model) -> dict:
    """Get substrate MDVs from freeflux model.

    Parameters
    ----------
    model
        A freeflux model

    Returns
    -------
    dict
        EMU -> MDV array
    """
    return {emu: np.array(mdv) for emu, mdv in model.substrate_MDVs.items()}


def simulate_inst_mdvs_jax(
    matrices: dict[int, dict],
    initial_X: dict[int, np.ndarray],
    initial_Y: dict[int, np.ndarray],
    substrate_mdvs: dict,
    timepoints: list[float],
    source_emu_lookup: dict,
) -> dict[str, dict[float, jnp.ndarray]]:
    """Full INST MDV simulation (JAX-compatible).

    This function iterates through timepoints and EMU sizes, computing
    the MDV evolution using the matrix exponential integration scheme.

    Parameters
    ----------
    matrices : dict
        EMU size -> {"A", "B", "M", "product_emus", "source_emus"} from extract_emu_matrices
    initial_X : dict
        EMU size -> initial metabolite MDVs
    initial_Y : dict
        EMU size -> initial source MDVs
    substrate_mdvs : dict
        EMU -> substrate MDV array (fixed labeling patterns)
    timepoints : list
        Simulation timepoints (starting from 0 or first positive time)
    source_emu_lookup : dict
        Mapping from source EMU objects to their IDs (for convolution lookups)

    Returns
    -------
    dict
        {emu_id: {timepoint: mdv_array}} for all product EMUs at all timepoints
    """
    sim_mdvs = {}  # emu_id -> {timepoint -> mdv}
    Ys = {}  # timepoint -> {size -> Y array}
    Xs = {}  # timepoint -> {size -> X array}

    # Initialize at t=0
    t1 = 0.0
    for size in sorted(matrices.keys()):
        Ys.setdefault(t1, {})[size] = jnp.array(initial_Y[size])
        Xs.setdefault(t1, {})[size] = jnp.array(initial_X[size])

    # Process each timepoint
    for t in timepoints:
        if t != 0.0:
            t0 = t1
            t1 = t
            delta_t = t1 - t0

            # Process EMU sizes in ascending order (smaller EMUs first, for convolutions)
            for size in sorted(matrices.keys()):
                mat = matrices[size]
                A = jnp.array(mat["A"])
                B = jnp.array(mat["B"])
                M = jnp.array(mat["M"])
                product_emus = mat["product_emus"]
                source_emus = mat["source_emus"]

                # Get previous state
                X_t0 = Xs[t0][size]
                Y_t0 = Ys[t0][size]

                # Build Y_t1 from source EMUs
                Y_t1_list = []
                for source_emu in source_emus:
                    if not isinstance(source_emu, Iterable) or isinstance(source_emu, str):
                        # Single source EMU - look up in substrate MDVs
                        source_mdv = _get_substrate_mdv(source_emu, substrate_mdvs, source_emu_lookup)
                    else:
                        # Tuple of EMUs - need to convolve
                        mdvs = []
                        for emu in source_emu:
                            # Check if this EMU was computed in this simulation
                            emu_id = _get_emu_id(emu, source_emu_lookup)
                            if emu_id in sim_mdvs and t1 in sim_mdvs[emu_id]:
                                mdv = sim_mdvs[emu_id][t1]
                            else:
                                # Must be a substrate MDV
                                mdv = _get_substrate_mdv(emu, substrate_mdvs, source_emu_lookup)
                            mdvs.append(jnp.array(mdv))
                        # Convolve all MDVs
                        source_mdv = reduce(mdv_convolve, mdvs)
                    Y_t1_list.append(source_mdv)

                Y_t1 = jnp.array(Y_t1_list)

                # Run INST step
                X_t1 = inst_mdv_step(X_t0, Y_t0, Y_t1, A, B, M, delta_t)

                # Store results
                Ys.setdefault(t1, {})[size] = Y_t1
                Xs.setdefault(t1, {})[size] = X_t1

                # Store individual EMU MDVs
                for i, product_emu in enumerate(product_emus):
                    emu_id = _get_emu_id(product_emu, source_emu_lookup)
                    if emu_id not in sim_mdvs:
                        sim_mdvs[emu_id] = {}
                    sim_mdvs[emu_id][t1] = X_t1[i]

    return sim_mdvs


def _get_emu_id(emu, lookup: dict) -> str:
    """Get EMU ID string from EMU object."""
    if hasattr(emu, "id"):
        return emu.id
    if emu in lookup:
        return lookup[emu]
    return str(emu)


def _get_substrate_mdv(emu, substrate_mdvs: dict, lookup: dict) -> jnp.ndarray:
    """Get substrate MDV for an EMU."""
    # Try direct lookup
    if emu in substrate_mdvs:
        return jnp.array(substrate_mdvs[emu])

    # Try by ID
    emu_id = _get_emu_id(emu, lookup)
    for key, mdv in substrate_mdvs.items():
        key_id = _get_emu_id(key, lookup)
        if key_id == emu_id:
            return jnp.array(mdv)

    raise KeyError(f"Substrate MDV not found for EMU: {emu}")


def build_source_emu_lookup(model) -> dict:
    """Build lookup dict from EMU objects to their string IDs.

    Parameters
    ----------
    model
        A freeflux model

    Returns
    -------
    dict
        EMU object -> EMU ID string
    """
    lookup = {}

    # From substrate MDVs
    for emu in model.substrate_MDVs:
        if hasattr(emu, "id"):
            lookup[emu] = emu.id

    # From all matrices
    for size in model.matrix_As:
        _, _, product_emus = model.matrix_As[size]
        _, _, source_emus = model.matrix_Bs[size]

        for emu in product_emus:
            if hasattr(emu, "id"):
                lookup[emu] = emu.id

        for source_emu in source_emus:
            if isinstance(source_emu, Iterable) and not isinstance(source_emu, str):
                for emu in source_emu:
                    if hasattr(emu, "id"):
                        lookup[emu] = emu.id
            elif hasattr(source_emu, "id"):
                lookup[source_emu] = source_emu.id

    return lookup


def run_freeflux_inst_simulation(
    reactions_file: Path,
    target_emus: dict[str, list[str]],
    timepoints: list[float],
    labeling_substrate: str,
    labeling_pattern: list[str],
    labeling_percentage: list[float],
    labeling_purity: list[float],
    fluxes: dict[str, float],
    concentrations: dict[str, float],
    use_jax: bool = True,
    pure_unlabeled: bool = False,
) -> dict[str, dict[float, np.ndarray]]:
    """Run INST simulation using freeflux for setup and optionally JAX for computation.

    This is the main entry point for running INST simulations with the JAX backend.

    Parameters
    ----------
    reactions_file : Path
        Path to reactions.tsv file
    target_emus : dict
        Target EMUs {metabolite: [atom_patterns]}
    timepoints : list
        Simulation timepoints
    labeling_substrate : str
        Substrate for labeling (e.g., "Glc_ext")
    labeling_pattern : list
        Labeling patterns (e.g., ["110000"])
    labeling_percentage : list
        Percentage of each labeling pattern
    labeling_purity : list
        Purity of each labeling pattern
    fluxes : dict
        Flux values {flux_id: value}
    concentrations : dict
        Concentration values {metabolite_id: value}
    use_jax : bool
        If True, use JAX implementation; if False, use freeflux native
    pure_unlabeled : bool
        If True, override initial states and substrate MDVs to use pure unlabeled
        [1, 0, ...] instead of natural abundance, matching build_emu_network().
        Only applies when use_jax=True.

    Returns
    -------
    dict
        {emu_id: {timepoint: mdv_array}}
    """
    # Build model and prepare simulator
    model = Model("emu_jax_model")
    model.read_from_file(str(reactions_file))

    with model.simulator("inst") as isim:
        isim.set_target_EMUs(target_emus)
        isim.set_timepoints(timepoints)
        isim.set_labeling_strategy(
            labeling_substrate,
            labeling_pattern=labeling_pattern,
            percentage=labeling_percentage,
            purity=labeling_purity,
        )

        # Set fluxes
        for flux_id, value in fluxes.items():
            isim.set_flux(flux_id, value)

        # Set concentrations
        for metab_id, value in concentrations.items():
            isim.set_concentration(metab_id, value)

        # Prepare the model (builds EMU network and matrices)
        isim.prepare(n_jobs=1)

        if not use_jax:
            # Use freeflux native simulation
            res = isim.simulate()
            mdvs = {}
            for emu_id in res.simulated_EMUs:
                mdv_result = res.simulated_MDV(emu_id)
                mdvs[emu_id] = {t: np.array(mdv_result[t]) for t in res.timepoints if t > 0}
            return mdvs

        # Extract matrices and run JAX simulation
        matrices = extract_emu_matrices(isim, fluxes, concentrations)
        initial_X, initial_Y = get_initial_states(isim.model)
        substrate_mdvs = get_substrate_mdvs(isim.model)
        source_emu_lookup = build_source_emu_lookup(isim.model)

        if pure_unlabeled:
            # Override initial states with pure unlabeled (matching build_emu_network)
            # Identify labeled substrate EMU IDs
            labeled_emu_ids = set()
            for emu in isim.model.substrate_MDVs:
                emu_id = _get_emu_id_str(emu)
                parts = emu_id.rsplit("_", 1)
                metab_name = parts[0] if len(parts) == 2 else emu_id
                if metab_name == labeling_substrate:
                    labeled_emu_ids.add(emu_id)

            # Override initial_X: pure unlabeled [1, 0, ...]
            for size in initial_X:
                arr = initial_X[size]
                pure = np.zeros_like(arr)
                pure[:, 0] = 1.0
                initial_X[size] = pure

            # Override initial_Y: pure unlabeled for all source EMUs at t=0
            # (labeled substrate MDVs are applied at simulation time, not in initial_Y)
            for size in initial_Y:
                arr = initial_Y[size]
                pure = np.zeros_like(arr)
                pure[:, 0] = 1.0
                initial_Y[size] = pure

            # Override substrate_mdvs: recompute without natural abundance
            pat = labeling_pattern[0] if isinstance(labeling_pattern, list) else labeling_pattern
            pur = labeling_purity[0] if isinstance(labeling_purity, list) else labeling_purity
            pct = labeling_percentage[0] if isinstance(labeling_percentage, list) else labeling_percentage
            for emu, mdv in list(substrate_mdvs.items()):
                emu_id = _get_emu_id_str(emu) if hasattr(emu, 'id') else str(emu)
                if emu_id in labeled_emu_ids:
                    pure_mdv = _compute_substrate_mdv_pure(emu, pat, pur, pct)
                    substrate_mdvs[emu] = pure_mdv
                else:
                    pure_mdv = np.zeros_like(mdv)
                    pure_mdv[0] = 1.0
                    substrate_mdvs[emu] = pure_mdv

        mdvs = simulate_inst_mdvs_jax(
            matrices=matrices,
            initial_X=initial_X,
            initial_Y=initial_Y,
            substrate_mdvs=substrate_mdvs,
            timepoints=timepoints,
            source_emu_lookup=source_emu_lookup,
        )

        # Convert to numpy arrays for output
        return {
            emu_id: {t: np.array(mdv) for t, mdv in timepoints_mdvs.items()}
            for emu_id, timepoints_mdvs in mdvs.items()
        }


def run_glycolysis_simulation(
    timepoints: list[float] = None,
    target_emus: dict[str, list[str]] = None,
    gibbs_overrides: dict[str, float] = None,
    use_jax: bool = True,
) -> dict[str, dict[float, np.ndarray]]:
    """Run INST simulation on glycolysis_mech_gly3p model.

    This is a convenience function that loads the glycolysis model with
    default fluxes from TOML files.

    Parameters
    ----------
    timepoints : list, optional
        Simulation timepoints. Defaults to logarithmically spaced from 1e-3 to 1e3.
    target_emus : dict, optional
        Target EMUs. Defaults to g6p, f6p, dhap, g3p.
    gibbs_overrides : dict, optional
        Override Gibbs energy values for specific reactions.
    use_jax : bool
        If True, use JAX implementation.

    Returns
    -------
    dict
        {emu_id: {timepoint: mdv_array}}
    """
    model_dir = PROJECT_DIR / "models" / "glycolysis_mech_gly3p"
    reactions_file = model_dir / "reactions.tsv"
    fluxes_file = model_dir / "glycolysis_fluxes.toml"
    thermo_file = model_dir / "glycolysis_thermodynamics.toml"

    if timepoints is None:
        timepoints = [0.0] + list(np.logspace(-3, 3, 50))

    if target_emus is None:
        target_emus = {
            "g6p": ["123456"],
            "f6p": ["123456"],
            "dhap": ["123"],
            "g3p": ["123"],
        }

    # Load net fluxes and thermodynamics
    net_fluxes = load_fluxes_from_toml(fluxes_file)
    thermo, temperature = load_thermo_data(thermo_file, overrides=gibbs_overrides)

    # Build model to get flux IDs
    model = Model("glycolysis")
    model.read_from_file(str(reactions_file))

    # Convert net fluxes to forward/backward
    fluxes = prepare_fluxes_for_model(model, net_fluxes, thermo, temperature)

    # Default concentrations
    concentrations = {}
    for metab in model.metabolites:
        if metab in ("FBA_dhap", "EC2", "EC3"):
            concentrations[metab] = 0.1
        else:
            concentrations[metab] = 1.0

    return run_freeflux_inst_simulation(
        reactions_file=reactions_file,
        target_emus=target_emus,
        timepoints=timepoints,
        labeling_substrate="Glc_ext",
        labeling_pattern=["110000"],
        labeling_percentage=[1.0],
        labeling_purity=[1.0],
        fluxes=fluxes,
        concentrations=concentrations,
        use_jax=use_jax,
    )


def main():
    """Demo: Run INST simulation on glycolysis_mech_gly3p model."""
    import sys

    print("=" * 80)
    print("EMU JAX Demo: INST Simulation")
    print("=" * 80)
    sys.stdout.flush()

    timepoints = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    target_emus = {
        "g6p": ["123456"],
        "f6p": ["123456"],
        "dhap": ["123"],
        "g3p": ["123"],
    }

    print("\nRunning freeflux native simulation...")
    sys.stdout.flush()
    mdvs_freeflux = run_glycolysis_simulation(
        timepoints=timepoints,
        target_emus=target_emus,
        use_jax=False,
    )

    print("\nRunning JAX simulation...")
    sys.stdout.flush()
    mdvs_jax = run_glycolysis_simulation(
        timepoints=timepoints,
        target_emus=target_emus,
        use_jax=True,
    )

    print("\n" + "=" * 80)
    print("Comparison: Freeflux vs JAX")
    print("=" * 80)

    max_diffs = []
    for emu_id in sorted(mdvs_freeflux.keys()):
        if emu_id in mdvs_jax:
            print(f"\nEMU: {emu_id}")
            for t in sorted(mdvs_freeflux[emu_id].keys()):
                if t in mdvs_jax[emu_id]:
                    ff_mdv = mdvs_freeflux[emu_id][t]
                    jax_mdv = mdvs_jax[emu_id][t]
                    max_diff = np.max(np.abs(ff_mdv - jax_mdv))
                    max_diffs.append(max_diff)
                    print(f"  t={t:.4f}: max diff = {max_diff:.2e}")
        else:
            print(f"\nEMU: {emu_id} - not in JAX output")

    if max_diffs:
        print(f"\nOverall maximum difference: {max(max_diffs):.2e}")

    sys.stdout.flush()


# =============================================================================
# Pure JAX EMU Simulation (JAX-Traceable for MCMC)
# =============================================================================


def extract_matrix_structure_A(lambA, flux_ids, base_flux_values):
    """Extract sparse structure of A matrix via numerical differentiation.

    Parameters
    ----------
    lambA : callable
        Lambda function that takes flux values and returns A matrix
    flux_ids : list
        List of flux IDs in the order expected by lambA
    base_flux_values : list
        Base flux values for numerical differentiation

    Returns
    -------
    list
        List of (row, col, flux_idx, coefficient) tuples
    """
    eps = 1e-8
    base_flux_array = np.array(base_flux_values, dtype=float)
    base_A = np.array(lambA(*base_flux_array))

    structure = []

    for row in range(base_A.shape[0]):
        for col in range(base_A.shape[1]):
            if abs(base_A[row, col]) > 1e-10:
                is_constant = True
                for flux_idx in range(len(flux_ids)):
                    perturbed = base_flux_array.copy()
                    perturbed[flux_idx] += eps
                    perturbed_A = np.array(lambA(*perturbed))
                    if abs(perturbed_A[row, col] - base_A[row, col]) > 1e-10:
                        is_constant = False
                        break
                if is_constant and abs(base_A[row, col]) > 1e-10:
                    structure.append((int(row), int(col), -1, float(base_A[row, col])))

    for flux_idx in range(len(flux_ids)):
        perturbed = base_flux_array.copy()
        perturbed[flux_idx] += eps
        perturbed_A = np.array(lambA(*perturbed))
        dA = (perturbed_A - base_A) / eps

        for row in range(dA.shape[0]):
            for col in range(dA.shape[1]):
                if abs(dA[row, col]) > 1e-10:
                    structure.append((int(row), int(col), int(flux_idx), float(dA[row, col])))

    return structure


def extract_matrix_structure_B(lambB, flux_ids, base_flux_values):
    """Extract sparse structure of B matrix via numerical differentiation.

    Parameters
    ----------
    lambB : callable
        Lambda function that takes flux values and returns B matrix
    flux_ids : list
        List of flux IDs in the order expected by lambB
    base_flux_values : list
        Base flux values for numerical differentiation

    Returns
    -------
    list
        List of (row, col, flux_idx, coefficient) tuples
    """
    eps = 1e-8
    base_flux_array = np.array(base_flux_values, dtype=float)
    base_B = np.array(lambB(*base_flux_array))

    structure = []

    for row in range(base_B.shape[0]):
        for col in range(base_B.shape[1]):
            if abs(base_B[row, col]) > 1e-10:
                is_constant = True
                for flux_idx in range(len(flux_ids)):
                    perturbed = base_flux_array.copy()
                    perturbed[flux_idx] += eps
                    perturbed_B = np.array(lambB(*perturbed))
                    if abs(perturbed_B[row, col] - base_B[row, col]) > 1e-10:
                        is_constant = False
                        break
                if is_constant:
                    structure.append((int(row), int(col), -1, float(base_B[row, col])))

    for flux_idx in range(len(flux_ids)):
        perturbed = base_flux_array.copy()
        perturbed[flux_idx] += eps
        perturbed_B = np.array(lambB(*perturbed))
        dB = (perturbed_B - base_B) / eps

        for row in range(dB.shape[0]):
            for col in range(dB.shape[1]):
                if abs(dB[row, col]) > 1e-10:
                    structure.append((int(row), int(col), int(flux_idx), float(dB[row, col])))

    return structure


def extract_matrix_structure_M(lambM, metab_ids, base_conc_values):
    """Extract sparse structure of M matrix via numerical differentiation.

    Parameters
    ----------
    lambM : callable
        Lambda function that takes concentration values and returns M matrix
    metab_ids : list
        List of metabolite IDs in the order expected by lambM
    base_conc_values : list
        Base concentration values for numerical differentiation

    Returns
    -------
    list
        List of (row, conc_idx, coefficient) tuples (diagonal only)
    """
    eps = 1e-8
    base_conc_array = np.array(base_conc_values, dtype=float)
    base_M = np.array(lambM(*base_conc_array))

    structure = []
    n_products = base_M.shape[0]

    for row in range(n_products):
        if abs(base_M[row, row]) > 1e-10:
            is_constant = True
            for conc_idx in range(len(metab_ids)):
                perturbed = base_conc_array.copy()
                perturbed[conc_idx] += eps
                perturbed_M = np.array(lambM(*perturbed))
                if abs(perturbed_M[row, row] - base_M[row, row]) > 1e-10:
                    is_constant = False
                    break
            if is_constant:
                structure.append((int(row), -1, float(base_M[row, row])))

    for conc_idx in range(len(metab_ids)):
        perturbed = base_conc_array.copy()
        perturbed[conc_idx] += eps
        perturbed_M = np.array(lambM(*perturbed))
        dM = (perturbed_M - base_M) / eps

        for row in range(n_products):
            if abs(dM[row, row]) > 1e-10:
                structure.append((int(row), int(conc_idx), float(dM[row, row])))

    return structure


def _get_emu_id_str(emu) -> str:
    """Extract EMU ID string from EMU object."""
    if hasattr(emu, "id"):
        return emu.id
    return str(emu)


def _compute_substrate_mdv_pure(emu, labeling_pattern, purity, percentage):
    """Compute substrate MDV without natural abundance.

    Same algorithm as freeflux's get_substrate_MDV but uses [1, 0] for
    unlabeled positions instead of natural abundance [0.9893, 0.0107].

    Parameters
    ----------
    emu : EMU object
        Freeflux EMU with atom_nos attribute (1-indexed)
    labeling_pattern : str
        Labeling pattern string (e.g., "10")
    purity : float
        Isotopic purity for labeled positions
    percentage : float
        Molar fraction of tracer
    """
    atom_nos = emu.atom_nos  # 1-indexed
    n_atoms = len(atom_nos)

    # Build per-atom MDVs and convolve
    result = np.array([1.0])  # identity for convolution
    for atom_no in atom_nos:
        if labeling_pattern[atom_no - 1] == '1':
            atom_mdv = np.array([1 - purity, purity])
        else:
            atom_mdv = np.array([1.0, 0.0])  # pure unlabeled, no natural abundance
        result = np.convolve(result, atom_mdv)

    # Weight by percentage, add pure unlabeled remainder
    mdv = percentage * result
    if percentage < 1.0:
        unlabeled = np.zeros(n_atoms + 1)
        unlabeled[0] = 1.0
        mdv += (1 - percentage) * unlabeled

    return mdv


def build_emu_network(
    reactions_file: Path,
    target_emus: dict[str, list[str]],
    labeling_substrate: str,
    labeling_pattern: list[str],
    labeling_percentage: list[float],
    labeling_purity: list[float],
) -> dict:
    """Build EMU network structure at runtime for JAX-traceable simulation.

    This replaces the previous serialize-to-JSON approach. The network is built
    from freeflux's EMU decomposition using numerical differentiation to extract
    matrix structure, then returned as a dict ready for precompute_matrix_structure().

    Initial states use pure unlabeled MDVs ([1, 0, ...]) instead of natural
    abundance. Substrate MDVs for non-labeled substrates also use pure unlabeled.

    Parameters
    ----------
    reactions_file : Path
        Path to reactions.tsv file
    target_emus : dict
        Target EMUs {metabolite: [atom_patterns]}
    labeling_substrate : str
        Substrate for labeling (e.g., "A_ext")
    labeling_pattern : list
        Labeling patterns (e.g., ["10"])
    labeling_percentage : list
        Percentage of each labeling pattern
    labeling_purity : list
        Purity of each labeling pattern

    Returns
    -------
    dict
        EMU network structure for pure JAX simulation
    """
    from collections.abc import Iterable

    model = Model("emu_builder")
    model.read_from_file(str(reactions_file))

    with model.simulator("inst") as isim:
        isim.set_target_EMUs(target_emus)
        isim.set_timepoints([0.0, 1.0])  # Dummy timepoints for setup
        isim.set_labeling_strategy(
            labeling_substrate,
            labeling_pattern=labeling_pattern,
            percentage=labeling_percentage,
            purity=labeling_purity,
        )

        # Set dummy non-zero fluxes to avoid division errors
        for flux_id in model.totalfluxids:
            isim.set_flux(flux_id, 1.0)

        # Set dummy concentrations
        for metab_id in model.metabolites:
            isim.set_concentration(metab_id, 1.0)

        isim.prepare(n_jobs=1)

        # Build network structure
        network = {
            "sizes": sorted(list(model.matrix_As.keys())),
            "flux_ids_all": list(model.totalfluxids),
            "metab_ids_all": list(model.metabolites),
            "matrix_A_structure": {},
            "matrix_B_structure": {},
            "matrix_M_structure": {},
            "flux_ids_A": {},
            "flux_ids_B": {},
            "metab_ids_M": {},
            "product_emu_ids": {},
            "source_emu_info": {},
            "initial_X": {},
            "initial_Y": {},
            "substrate_mdvs": {},
            "matrix_shapes": {},
        }

        # Collect labeled substrate EMU IDs for later
        # EMU IDs are like "A_ext_1", "A_ext_12" - strip the atom suffix to get metabolite name
        labeled_substrate_emu_ids = set()
        for emu in model.substrate_MDVs:
            emu_id = _get_emu_id_str(emu)
            # The metabolite name is everything except the last underscore-separated atom pattern
            # e.g., "A_ext_1" -> "A_ext", "A_ext_12" -> "A_ext", "P_ext_in_1" -> "P_ext_in"
            parts = emu_id.rsplit("_", 1)
            metab_name = parts[0] if len(parts) == 2 else emu_id
            if metab_name == labeling_substrate:
                labeled_substrate_emu_ids.add(emu_id)

        # Process each EMU size
        for size in network["sizes"]:
            lambA, flux_ids_A, product_emus = model.matrix_As[size]
            lambB, flux_ids_B, source_emus = model.matrix_Bs[size]
            lambM, metab_ids = model.matrix_Ms[size]

            network["flux_ids_A"][str(size)] = list(flux_ids_A)
            network["flux_ids_B"][str(size)] = list(flux_ids_B)
            network["metab_ids_M"][str(size)] = list(metab_ids)

            base_flux_values_A = [1.0] * len(flux_ids_A)
            base_flux_values_B = [1.0] * len(flux_ids_B)
            base_conc_values = [1.0] * len(metab_ids)

            A_structure = extract_matrix_structure_A(lambA, flux_ids_A, base_flux_values_A)
            B_structure = extract_matrix_structure_B(lambB, flux_ids_B, base_flux_values_B)
            M_structure = extract_matrix_structure_M(lambM, metab_ids, base_conc_values)

            network["matrix_A_structure"][str(size)] = A_structure
            network["matrix_B_structure"][str(size)] = B_structure
            network["matrix_M_structure"][str(size)] = M_structure

            network["product_emu_ids"][str(size)] = [_get_emu_id_str(emu) for emu in product_emus]

            source_info = []
            for source_emu in source_emus:
                if isinstance(source_emu, Iterable) and not isinstance(source_emu, str):
                    source_info.append({
                        "type": "composite",
                        "components": [_get_emu_id_str(emu) for emu in source_emu],
                    })
                else:
                    source_info.append({
                        "type": "single",
                        "id": _get_emu_id_str(source_emu),
                    })
            network["source_emu_info"][str(size)] = source_info

            # Override initial_X: pure unlabeled MDVs [1, 0, ...]
            freeflux_X = np.array(model.initial_matrix_Xs[size])
            n_products_in_size, n_mass = freeflux_X.shape
            pure_X = np.zeros_like(freeflux_X)
            pure_X[:, 0] = 1.0  # M+0 = 1.0, rest = 0.0
            network["initial_X"][str(size)] = pure_X.tolist()

            # Override initial_Y: pure unlabeled for all source EMUs at t=0
            # (labeled substrate MDVs are stored separately in substrate_mdvs
            # and applied at simulation time, not in initial_Y)
            freeflux_Y = np.array(model.initial_matrix_Ys[size])
            pure_Y = np.zeros_like(freeflux_Y)
            pure_Y[:, 0] = 1.0
            network["initial_Y"][str(size)] = pure_Y.tolist()

            base_A = np.array(lambA(*base_flux_values_A))
            base_B = np.array(lambB(*base_flux_values_B))
            base_M = np.array(lambM(*base_conc_values))
            network["matrix_shapes"][str(size)] = {
                "A": list(base_A.shape),
                "B": list(base_B.shape),
                "M": list(base_M.shape),
            }

        # Store substrate MDVs: recompute without natural abundance
        pat = labeling_pattern[0] if isinstance(labeling_pattern, list) else labeling_pattern
        pur = labeling_purity[0] if isinstance(labeling_purity, list) else labeling_purity
        pct = labeling_percentage[0] if isinstance(labeling_percentage, list) else labeling_percentage
        for emu, mdv in model.substrate_MDVs.items():
            emu_id = _get_emu_id_str(emu)
            if emu_id in labeled_substrate_emu_ids:
                # Recompute MDV without natural abundance
                pure_mdv = _compute_substrate_mdv_pure(emu, pat, pur, pct)
                network["substrate_mdvs"][emu_id] = pure_mdv.tolist()
            else:
                # Pure unlabeled: [1, 0, ...]
                mdv_arr = np.array(mdv)
                pure_mdv = np.zeros_like(mdv_arr)
                pure_mdv[0] = 1.0
                network["substrate_mdvs"][emu_id] = pure_mdv.tolist()

    return network


def precompute_matrix_structure(network: dict) -> dict:
    """Precompute JAX-compatible structure arrays for matrix building.

    This converts the serialized structure into fixed JAX arrays that can
    be used for fully JAX-traceable matrix construction.

    Parameters
    ----------
    network : dict
        Serialized EMU network structure

    Returns
    -------
    dict
        Precomputed structure arrays for each matrix type and size
    """
    flux_index_map = {fid: i for i, fid in enumerate(network["flux_ids_all"])}
    metab_index_map = {mid: i for i, mid in enumerate(network["metab_ids_all"])}

    precomputed = {
        "flux_index_map": flux_index_map,
        "metab_index_map": metab_index_map,
    }

    for size in network["sizes"]:
        size_str = str(size)

        # A matrix structure
        flux_ids_A = network["flux_ids_A"][size_str]
        flux_lookup_A = [flux_index_map[fid] for fid in flux_ids_A]
        structure_A = network["matrix_A_structure"][size_str]

        if structure_A:
            rows_A = jnp.array([s[0] for s in structure_A], dtype=jnp.int32)
            cols_A = jnp.array([s[1] for s in structure_A], dtype=jnp.int32)
            flux_idxs_A = jnp.array([s[2] for s in structure_A], dtype=jnp.int32)
            coeffs_A = jnp.array([s[3] for s in structure_A])
            global_flux_idxs_A = jnp.array([
                flux_lookup_A[s[2]] if s[2] >= 0 else 0
                for s in structure_A
            ], dtype=jnp.int32)
            precomputed[f"A_{size}"] = (rows_A, cols_A, flux_idxs_A, coeffs_A, global_flux_idxs_A)
        else:
            precomputed[f"A_{size}"] = None

        # B matrix structure
        flux_ids_B = network["flux_ids_B"][size_str]
        flux_lookup_B = [flux_index_map[fid] for fid in flux_ids_B]
        structure_B = network["matrix_B_structure"][size_str]

        if structure_B:
            rows_B = jnp.array([s[0] for s in structure_B], dtype=jnp.int32)
            cols_B = jnp.array([s[1] for s in structure_B], dtype=jnp.int32)
            flux_idxs_B = jnp.array([s[2] for s in structure_B], dtype=jnp.int32)
            coeffs_B = jnp.array([s[3] for s in structure_B])
            global_flux_idxs_B = jnp.array([
                flux_lookup_B[s[2]] if s[2] >= 0 else 0
                for s in structure_B
            ], dtype=jnp.int32)
            precomputed[f"B_{size}"] = (rows_B, cols_B, flux_idxs_B, coeffs_B, global_flux_idxs_B)
        else:
            precomputed[f"B_{size}"] = None

        # M matrix structure
        metab_ids_M = network["metab_ids_M"][size_str]
        metab_lookup_M = [metab_index_map[mid] for mid in metab_ids_M]
        structure_M = network["matrix_M_structure"][size_str]

        if structure_M:
            rows_M = jnp.array([s[0] for s in structure_M], dtype=jnp.int32)
            conc_idxs_M = jnp.array([s[1] for s in structure_M], dtype=jnp.int32)
            coeffs_M = jnp.array([s[2] for s in structure_M])
            global_conc_idxs_M = jnp.array([
                metab_lookup_M[s[1]] if s[1] >= 0 else 0
                for s in structure_M
            ], dtype=jnp.int32)
            precomputed[f"M_{size}"] = (rows_M, conc_idxs_M, coeffs_M, global_conc_idxs_M)
        else:
            precomputed[f"M_{size}"] = None

    return precomputed


def build_matrix_A_pure_jax(
    fluxes_array: jnp.ndarray,
    network: dict,
    size: int,
    precomputed: dict,
) -> jnp.ndarray:
    """Build A matrix using pure JAX operations (fully traceable).

    Parameters
    ----------
    fluxes_array : jnp.ndarray
        Flux values in order matching network["flux_ids_all"]
    network : dict
        Serialized EMU network structure
    size : int
        EMU size to build matrix for
    precomputed : dict
        Precomputed structure arrays from precompute_matrix_structure()

    Returns
    -------
    jnp.ndarray
        A matrix for the specified EMU size
    """
    size_str = str(size)
    shape = network["matrix_shapes"][size_str]["A"]
    A = jnp.zeros((shape[0], shape[1]))

    struct = precomputed.get(f"A_{size}")
    if struct is None:
        return A

    rows, cols, flux_idxs, coeffs, global_flux_idxs = struct

    # Compute values: for flux-dependent terms, multiply coeff by flux
    # For constant terms (flux_idx < 0), just use coeff
    is_flux_dependent = flux_idxs >= 0
    flux_values = jnp.where(
        is_flux_dependent,
        coeffs * fluxes_array[global_flux_idxs],
        coeffs
    )

    # Build matrix using scatter
    A = A.at[rows, cols].add(flux_values)
    return A


def build_matrix_B_pure_jax(
    fluxes_array: jnp.ndarray,
    network: dict,
    size: int,
    precomputed: dict,
) -> jnp.ndarray:
    """Build B matrix using pure JAX operations (fully traceable).

    Parameters
    ----------
    fluxes_array : jnp.ndarray
        Flux values in order matching network["flux_ids_all"]
    network : dict
        Serialized EMU network structure
    size : int
        EMU size to build matrix for
    precomputed : dict
        Precomputed structure arrays from precompute_matrix_structure()

    Returns
    -------
    jnp.ndarray
        B matrix for the specified EMU size
    """
    size_str = str(size)
    shape = network["matrix_shapes"][size_str]["B"]
    B = jnp.zeros((shape[0], shape[1]))

    struct = precomputed.get(f"B_{size}")
    if struct is None:
        return B

    rows, cols, flux_idxs, coeffs, global_flux_idxs = struct

    is_flux_dependent = flux_idxs >= 0
    flux_values = jnp.where(
        is_flux_dependent,
        coeffs * fluxes_array[global_flux_idxs],
        coeffs
    )

    B = B.at[rows, cols].add(flux_values)
    return B


def build_matrix_M_pure_jax(
    concs_array: jnp.ndarray,
    network: dict,
    size: int,
    precomputed: dict,
) -> jnp.ndarray:
    """Build M (concentration diagonal) matrix using pure JAX operations (fully traceable).

    Parameters
    ----------
    concs_array : jnp.ndarray
        Concentration values in order matching network["metab_ids_all"]
    network : dict
        Serialized EMU network structure
    size : int
        EMU size to build matrix for
    precomputed : dict
        Precomputed structure arrays from precompute_matrix_structure()

    Returns
    -------
    jnp.ndarray
        M matrix for the specified EMU size
    """
    size_str = str(size)
    shape = network["matrix_shapes"][size_str]["M"]
    n = shape[0]
    M = jnp.zeros((n, n))

    struct = precomputed.get(f"M_{size}")
    if struct is None:
        return M

    rows, conc_idxs, coeffs, global_conc_idxs = struct

    is_conc_dependent = conc_idxs >= 0
    conc_values = jnp.where(
        is_conc_dependent,
        coeffs * concs_array[global_conc_idxs],
        coeffs
    )

    M = M.at[rows, rows].add(conc_values)
    return M


def build_all_matrices_pure_jax(
    fluxes_array: jnp.ndarray,
    concs_array: jnp.ndarray,
    network: dict,
    precomputed: dict,
) -> dict[int, dict[str, jnp.ndarray]]:
    """Build all EMU matrices for all sizes using pure JAX operations (fully traceable).

    Parameters
    ----------
    fluxes_array : jnp.ndarray
        Flux values in order matching network["flux_ids_all"]
    concs_array : jnp.ndarray
        Concentration values in order matching network["metab_ids_all"]
    network : dict
        Serialized EMU network structure
    precomputed : dict
        Precomputed structure arrays from precompute_matrix_structure()

    Returns
    -------
    dict
        {size: {"A": array, "B": array, "M": array}}
    """
    matrices = {}
    for size in network["sizes"]:
        A = build_matrix_A_pure_jax(fluxes_array, network, size, precomputed)
        B = build_matrix_B_pure_jax(fluxes_array, network, size, precomputed)
        M = build_matrix_M_pure_jax(concs_array, network, size, precomputed)
        matrices[size] = {"A": A, "B": B, "M": M}

    return matrices


def get_source_mdv_pure_jax(
    source_info: dict,
    substrate_mdvs: dict[str, jnp.ndarray],
    computed_mdvs: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Get source MDV, handling convolutions for composite sources.

    Parameters
    ----------
    source_info : dict
        Source EMU info from network["source_emu_info"]
    substrate_mdvs : dict
        Substrate MDVs {emu_id: array}
    computed_mdvs : dict
        Already computed MDVs from smaller sizes {emu_id: array}

    Returns
    -------
    jnp.ndarray
        Source MDV (convolved if composite)
    """
    if source_info["type"] == "single":
        emu_id = source_info["id"]
        if emu_id in substrate_mdvs:
            return substrate_mdvs[emu_id]
        elif emu_id in computed_mdvs:
            return computed_mdvs[emu_id]
        else:
            raise KeyError(f"Source MDV not found: {emu_id}")
    else:
        # Composite source - need to convolve
        components = source_info["components"]
        mdvs = []
        for comp_id in components:
            if comp_id in substrate_mdvs:
                mdvs.append(substrate_mdvs[comp_id])
            elif comp_id in computed_mdvs:
                mdvs.append(computed_mdvs[comp_id])
            else:
                raise KeyError(f"Component MDV not found: {comp_id}")

        # Convolve all MDVs
        result = mdvs[0]
        for mdv in mdvs[1:]:
            result = mdv_convolve(result, mdv)
        return result


def simulate_inst_mdvs_pure_jax(
    fluxes_array: jnp.ndarray,
    concs_array: jnp.ndarray,
    timepoints: jnp.ndarray,
    network: dict,
    precomputed: dict = None,
) -> dict[str, jnp.ndarray]:
    """Fully JAX-traceable INST MDV simulation.

    This function performs EMU INST simulation using only JAX operations,
    making it compatible with gradient-based MCMC (blackjax NUTS).

    Parameters
    ----------
    fluxes_array : jnp.ndarray
        Flux values in order matching network["flux_ids_all"]
    concs_array : jnp.ndarray
        Concentration values in order matching network["metab_ids_all"]
    timepoints : jnp.ndarray
        Simulation timepoints (first must be 0.0)
    network : dict
        EMU network structure from build_emu_network()
    precomputed : dict, optional
        Precomputed structure arrays from precompute_matrix_structure().
        If None, will be computed (slower, use for one-off calls).

    Returns
    -------
    dict
        {emu_id: mdv_array} for all product EMUs at final timepoint
    """
    # Precompute structure if not provided
    if precomputed is None:
        precomputed = precompute_matrix_structure(network)

    # Build matrices for all sizes using precomputed structure
    matrices = build_all_matrices_pure_jax(fluxes_array, concs_array, network, precomputed)

    # Load substrate MDVs
    substrate_mdvs = {k: jnp.array(v) for k, v in network["substrate_mdvs"].items()}

    # Initialize states
    Xs = {}
    Ys = {}
    for size in network["sizes"]:
        size_str = str(size)
        Xs[size] = jnp.array(network["initial_X"][size_str])
        Ys[size] = jnp.array(network["initial_Y"][size_str])

    # Track computed MDVs for convolutions
    computed_mdvs = {}

    # Store product EMU info for extraction
    product_emu_ids = {
        size: network["product_emu_ids"][str(size)]
        for size in network["sizes"]
    }
    source_emu_info = {
        size: network["source_emu_info"][str(size)]
        for size in network["sizes"]
    }

    # Process timepoints starting from index 1 (skip t=0)
    # Note: We use Python loop here since len(timepoints) is known at trace time
    # and we use jnp.where to handle the dt computation in a trace-friendly way
    n_timepoints = len(timepoints)
    t_prev = timepoints[0]  # Should be 0.0

    for t_idx in range(1, n_timepoints):
        t = timepoints[t_idx]
        dt = t - t_prev

        # Process EMU sizes in ascending order (for convolutions)
        # This loop is over a static list, so it's OK for tracing
        for size in sorted(network["sizes"]):
            mat = matrices[size]
            A = mat["A"]
            B = mat["B"]
            M = mat["M"]

            X_prev = Xs[size]
            Y_prev = Ys[size]

            # Build Y_current from source EMUs
            source_info_list = source_emu_info[size]
            Y_curr_list = []
            for src_info in source_info_list:
                src_mdv = get_source_mdv_pure_jax(src_info, substrate_mdvs, computed_mdvs)
                Y_curr_list.append(src_mdv)
            Y_curr = jnp.array(Y_curr_list)

            # Run INST step
            X_curr = inst_mdv_step(X_prev, Y_prev, Y_curr, A, B, M, dt)

            # Update states
            Xs[size] = X_curr
            Ys[size] = Y_curr

            # Store computed MDVs for this size
            for i, emu_id in enumerate(product_emu_ids[size]):
                computed_mdvs[emu_id] = X_curr[i]

        t_prev = t

    # Return final MDVs
    return computed_mdvs


def simulate_ss_mdvs_pure_jax(
    fluxes_array: jnp.ndarray,
    network: dict,
    precomputed: dict = None,
) -> dict[str, jnp.ndarray]:
    """Fully JAX-traceable steady-state EMU MDV simulation.

    Solves the isotopic steady-state equation: X = pinv(A) @ B @ Y

    This is simpler than instationary simulation - no time integration,
    no concentration matrix M, just linear algebra.

    Parameters
    ----------
    fluxes_array : jnp.ndarray
        Flux values in order matching network["flux_ids_all"]
    network : dict
        EMU network structure from build_emu_network()
    precomputed : dict, optional
        Precomputed structure arrays from precompute_matrix_structure().
        If None, will be computed.

    Returns
    -------
    dict
        {emu_id: mdv_array} for all product EMUs at steady state
    """
    if precomputed is None:
        precomputed = precompute_matrix_structure(network)

    # Load substrate MDVs
    substrate_mdvs = {k: jnp.array(v) for k, v in network["substrate_mdvs"].items()}

    # Track computed MDVs for cascading dependencies
    computed_mdvs = {}

    # Process EMU sizes in ascending order (smaller first for dependencies)
    for size in sorted(network["sizes"]):
        # Build A matrix (flux-weighted adjacency)
        A = build_matrix_A_pure_jax(fluxes_array, network, size, precomputed)

        # Build B matrix (input coefficients)
        B = build_matrix_B_pure_jax(fluxes_array, network, size, precomputed)

        # Build Y vector from source EMUs
        source_info_list = network["source_emu_info"][str(size)]
        Y_list = []
        for src_info in source_info_list:
            src_mdv = get_source_mdv_pure_jax(src_info, substrate_mdvs, computed_mdvs)
            Y_list.append(src_mdv)
        Y = jnp.array(Y_list)

        # Solve: A @ X = B @ Y  →  X = pinv(A) @ B @ Y
        X = jnp.linalg.pinv(A) @ B @ Y

        # Store computed MDVs for this size
        product_emu_ids = network["product_emu_ids"][str(size)]
        for i, emu_id in enumerate(product_emu_ids):
            computed_mdvs[emu_id] = X[i]

    return computed_mdvs


def run_freeflux_ss_simulation(
    reactions_file: Path,
    target_emus: dict[str, list[str]],
    labeling_substrate: str,
    labeling_pattern: list[str],
    labeling_percentage: list[float],
    labeling_purity: list[float],
    fluxes: dict[str, float],
    use_jax: bool = True,
) -> dict[str, np.ndarray]:
    """Run steady-state EMU simulation using freeflux backend.

    Parameters
    ----------
    reactions_file : Path
        Path to reactions.tsv file
    target_emus : dict
        Target EMUs {metabolite: [atom_patterns]}
    labeling_substrate : str
        Substrate for labeling
    labeling_pattern : list
        Labeling patterns
    labeling_percentage : list
        Percentage of each pattern
    labeling_purity : list
        Purity of each pattern
    fluxes : dict
        Flux values {flux_id: value}
    use_jax : bool
        If True, use pure JAX implementation; if False, use freeflux native

    Returns
    -------
    dict
        {emu_id: mdv_array} for steady-state MDVs
    """
    model = Model("emu_ss_model")
    model.read_from_file(str(reactions_file))

    with model.simulator("ss") as sim:  # Note: "ss" for steady-state
        sim.set_target_EMUs(target_emus)
        sim.set_labeling_strategy(
            labeling_substrate,
            labeling_pattern=labeling_pattern,
            percentage=labeling_percentage,
            purity=labeling_purity,
        )

        for flux_id, value in fluxes.items():
            sim.set_flux(flux_id, value)

        sim.prepare(n_jobs=1)

        if not use_jax:
            # Use freeflux native
            res = sim.simulate()
            return {emu_id: np.array(res.simulated_MDV(emu_id))
                    for emu_id in res.simulated_EMUs}

        # Use pure JAX implementation
        network = build_emu_network(
            reactions_file=reactions_file,
            target_emus=target_emus,
            labeling_substrate=labeling_substrate,
            labeling_pattern=labeling_pattern,
            labeling_percentage=labeling_percentage,
            labeling_purity=labeling_purity,
        )
        fluxes_array = create_flux_array_from_dict(fluxes, network)

        mdvs = simulate_ss_mdvs_pure_jax(fluxes_array, network)
        return {k: np.array(v) for k, v in mdvs.items()}


def create_flux_array_from_dict(
    fluxes: dict[str, float],
    network: dict,
) -> jnp.ndarray:
    """Create flux array from dict in order expected by network.

    Parameters
    ----------
    fluxes : dict
        Flux values {flux_id: value}
    network : dict
        Serialized EMU network structure

    Returns
    -------
    jnp.ndarray
        Flux array in order matching network["flux_ids_all"]
    """
    return jnp.array([fluxes.get(fid, 0.0) for fid in network["flux_ids_all"]])


def create_conc_array_from_dict(
    concs: dict[str, float],
    network: dict,
) -> jnp.ndarray:
    """Create concentration array from dict in order expected by network.

    Parameters
    ----------
    concs : dict
        Concentration values {metab_id: value}
    network : dict
        Serialized EMU network structure

    Returns
    -------
    jnp.ndarray
        Concentration array in order matching network["metab_ids_all"]
    """
    return jnp.array([concs.get(mid, 1.0) for mid in network["metab_ids_all"]])


if __name__ == "__main__":
    main()

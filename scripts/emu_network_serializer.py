#!/usr/bin/env python3
"""Serialize EMU network structure for JAX-traceable simulation.

This module extracts the sparse structure of EMU matrices from freeflux's lambda
functions using numerical differentiation. The serialized structure can then be
used to build matrices using pure JAX operations, enabling gradient-based MCMC.

The key insight is that freeflux represents matrices as symbolic lambda functions:
    A = lambA(*flux_values)
    B = lambB(*flux_values)
    M = lambM(*conc_values)

We extract the structure by:
1. Evaluating at a base point
2. Perturbing each input and measuring the derivative
3. Recording (row, col, input_idx, coefficient) for each non-zero derivative
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from freeflux import Model


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent


def extract_matrix_structure_A(lambA, flux_ids, base_flux_values):
    """Extract sparse structure of A matrix via numerical differentiation.

    The A matrix maps product EMUs to product EMUs (flux-dependent diagonal
    and off-diagonal terms).

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

    # Check for constant terms (non-zero at base)
    # These are typically zero in EMU A matrices but we check anyway
    for row in range(base_A.shape[0]):
        for col in range(base_A.shape[1]):
            if abs(base_A[row, col]) > 1e-10:
                # Check if this is a constant or flux-dependent
                is_constant = True
                for flux_idx in range(len(flux_ids)):
                    perturbed = base_flux_array.copy()
                    perturbed[flux_idx] += eps
                    perturbed_A = np.array(lambA(*perturbed))
                    if abs(perturbed_A[row, col] - base_A[row, col]) > 1e-10:
                        is_constant = False
                        break
                if is_constant and abs(base_A[row, col]) > 1e-10:
                    # This is a constant term - record with flux_idx=-1
                    structure.append((int(row), int(col), -1, float(base_A[row, col])))

    # Extract flux-dependent terms via numerical differentiation
    for flux_idx in range(len(flux_ids)):
        perturbed = base_flux_array.copy()
        perturbed[flux_idx] += eps
        perturbed_A = np.array(lambA(*perturbed))
        dA = (perturbed_A - base_A) / eps

        # Record non-zero derivatives
        for row in range(dA.shape[0]):
            for col in range(dA.shape[1]):
                if abs(dA[row, col]) > 1e-10:
                    structure.append((int(row), int(col), int(flux_idx), float(dA[row, col])))

    return structure


def extract_matrix_structure_B(lambB, flux_ids, base_flux_values):
    """Extract sparse structure of B matrix via numerical differentiation.

    The B matrix maps source EMUs to product EMUs (input contribution).

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

    # Check for constant terms
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

    # Extract flux-dependent terms
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

    The M matrix is diagonal with concentration values. It has dimensions
    (n_products, n_products) where n_products is the number of product EMUs.
    Each diagonal entry depends on the concentration of the metabolite that
    the corresponding EMU belongs to.

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

    # Check for constant terms on diagonal
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

    # Extract concentration-dependent terms
    for conc_idx in range(len(metab_ids)):
        perturbed = base_conc_array.copy()
        perturbed[conc_idx] += eps
        perturbed_M = np.array(lambM(*perturbed))
        dM = (perturbed_M - base_M) / eps

        # M is diagonal - check all diagonal entries
        for row in range(n_products):
            if abs(dM[row, row]) > 1e-10:
                structure.append((int(row), int(conc_idx), float(dM[row, row])))

    return structure


def get_emu_id(emu) -> str:
    """Extract EMU ID string from EMU object."""
    if hasattr(emu, "id"):
        return emu.id
    return str(emu)


def serialize_emu_network(
    reactions_file: Path,
    target_emus: dict[str, list[str]],
    labeling_substrate: str,
    labeling_pattern: list[str],
    labeling_percentage: list[float],
    labeling_purity: list[float],
) -> dict[str, Any]:
    """Serialize EMU network structure for JAX-traceable simulation.

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
        Percentage of each labeling pattern
    labeling_purity : list
        Purity of each labeling pattern

    Returns
    -------
    dict
        Serialized EMU network structure
    """
    model = Model("emu_serializer")
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
            "source_emu_info": {},  # info about source EMUs and convolutions
            "initial_X": {},
            "initial_Y": {},
            "substrate_mdvs": {},
            "matrix_shapes": {},
        }

        # Process each EMU size
        for size in network["sizes"]:
            lambA, flux_ids_A, product_emus = model.matrix_As[size]
            lambB, flux_ids_B, source_emus = model.matrix_Bs[size]
            lambM, metab_ids = model.matrix_Ms[size]

            # Store flux/metab IDs for this size
            network["flux_ids_A"][str(size)] = list(flux_ids_A)
            network["flux_ids_B"][str(size)] = list(flux_ids_B)
            network["metab_ids_M"][str(size)] = list(metab_ids)

            # Base values for numerical differentiation
            base_flux_values_A = [1.0] * len(flux_ids_A)
            base_flux_values_B = [1.0] * len(flux_ids_B)
            base_conc_values = [1.0] * len(metab_ids)

            # Extract matrix structures
            A_structure = extract_matrix_structure_A(lambA, flux_ids_A, base_flux_values_A)
            B_structure = extract_matrix_structure_B(lambB, flux_ids_B, base_flux_values_B)
            M_structure = extract_matrix_structure_M(lambM, metab_ids, base_conc_values)

            network["matrix_A_structure"][str(size)] = A_structure
            network["matrix_B_structure"][str(size)] = B_structure
            network["matrix_M_structure"][str(size)] = M_structure

            # Store product EMU IDs
            network["product_emu_ids"][str(size)] = [get_emu_id(emu) for emu in product_emus]

            # Store source EMU info (including convolution requirements)
            source_info = []
            for source_emu in source_emus:
                if isinstance(source_emu, (list, tuple)):
                    # Composite source EMU (requires convolution)
                    source_info.append({
                        "type": "composite",
                        "components": [get_emu_id(emu) for emu in source_emu],
                    })
                else:
                    # Single source EMU
                    source_info.append({
                        "type": "single",
                        "id": get_emu_id(source_emu),
                    })
            network["source_emu_info"][str(size)] = source_info

            # Store initial states
            network["initial_X"][str(size)] = np.array(model.initial_matrix_Xs[size]).tolist()
            network["initial_Y"][str(size)] = np.array(model.initial_matrix_Ys[size]).tolist()

            # Store matrix shapes
            base_A = np.array(lambA(*base_flux_values_A))
            base_B = np.array(lambB(*base_flux_values_B))
            base_M = np.array(lambM(*base_conc_values))
            network["matrix_shapes"][str(size)] = {
                "A": list(base_A.shape),
                "B": list(base_B.shape),
                "M": list(base_M.shape),
            }

        # Store substrate MDVs
        for emu, mdv in model.substrate_MDVs.items():
            emu_id = get_emu_id(emu)
            network["substrate_mdvs"][emu_id] = np.array(mdv).tolist()

    return network


def save_emu_network(network: dict, output_path: Path):
    """Save serialized EMU network to JSON file.

    Parameters
    ----------
    network : dict
        Serialized EMU network structure
    output_path : Path
        Output file path
    """
    with open(output_path, "w") as f:
        json.dump(network, f, indent=2)
    print(f"Saved EMU network to {output_path}")


def load_emu_network(input_path: Path) -> dict:
    """Load serialized EMU network from JSON file.

    Parameters
    ----------
    input_path : Path
        Input file path

    Returns
    -------
    dict
        Serialized EMU network structure
    """
    with open(input_path) as f:
        return json.load(f)


def serialize_simple_network():
    """Serialize the simple_network model used in enzyme_simulation.py."""
    reactions_file = PROJECT_DIR / "models" / "simple_network" / "reactions.tsv"
    output_path = PROJECT_DIR / "models" / "simple_network" / "emu_network.json"

    target_emus = {
        "A": ["12"],
        "EA": ["12"],
        "EPQ": ["12"],
        "EP": ["1"],
        "Q": ["1"],
        "P": ["1"],
    }

    network = serialize_emu_network(
        reactions_file=reactions_file,
        target_emus=target_emus,
        labeling_substrate="A_ext",
        labeling_pattern=["11"],
        labeling_percentage=[1.0],
        labeling_purity=[1.0],
    )

    save_emu_network(network, output_path)

    # Print summary
    print("\nEMU Network Summary:")
    print(f"  Sizes: {network['sizes']}")
    print(f"  Total flux IDs: {len(network['flux_ids_all'])}")
    for size in network["sizes"]:
        print(f"\n  Size {size}:")
        print(f"    Product EMUs: {network['product_emu_ids'][str(size)]}")
        print(f"    A matrix structure: {len(network['matrix_A_structure'][str(size)])} non-zero entries")
        print(f"    B matrix structure: {len(network['matrix_B_structure'][str(size)])} non-zero entries")
        print(f"    M matrix structure: {len(network['matrix_M_structure'][str(size)])} non-zero entries")

    return network


if __name__ == "__main__":
    serialize_simple_network()

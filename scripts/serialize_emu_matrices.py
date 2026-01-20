#!/usr/bin/env python3
"""Serialize EMU matrices from freeflux for use in enzyme_simulation.

This script uses freeflux to build the EMU network and extract matrices,
then serializes them to .npz files that can be loaded by pure JAX code
without requiring freeflux.

The serialized matrices include:
- Lambda functions for A, B, M matrices (as coefficient arrays)
- EMU network structure (product EMUs, source EMUs)
- Initial states and substrate MDVs
"""

from pathlib import Path
import sys

import numpy as np
from freeflux import Model

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ENZYME_SIM_DIR = PROJECT_DIR / "enzyme_simulation"
REACTIONS_FILE = ENZYME_SIM_DIR / "models" / "simple_network" / "reactions.tsv"
OUTPUT_FILE = ENZYME_SIM_DIR / "models" / "simple_network" / "emu_matrices.npz"


# Target EMUs for the enzyme simulation
# These correspond to the measurable metabolites in the system
TARGET_EMUS = {
    "A": ["12"],       # 2-atom substrate
    "EA": ["12"],      # 2-atom enzyme-substrate complex
    "EPQ": ["12"],     # 2-atom enzyme-product complex
    "EP": ["1"],       # 1-atom enzyme-product (single carbon)
    "Q": ["1"],        # 1-atom product
    "P": ["1"],        # 1-atom product
}


def extract_matrix_coefficients(lamb_func, n_vars):
    """Extract matrix coefficients from a freeflux lambda function.

    The lambda function takes flux/concentration values and returns a matrix.
    We evaluate it with unit values to extract the coefficient structure.

    Parameters
    ----------
    lamb_func : callable
        Lambda function that constructs the matrix
    n_vars : int
        Number of variables (flux IDs or metabolite IDs)

    Returns
    -------
    np.ndarray
        Coefficient tensor of shape (n_vars, *matrix_shape) where each
        coefficient[i] gives the contribution of variable i to the matrix.
    """
    # Evaluate with zeros to get the constant term (should be zero for A, B)
    zeros = [0.0] * n_vars
    constant = np.array(lamb_func(*zeros))

    # Evaluate with unit value for each variable to get coefficients
    coefficients = []
    for i in range(n_vars):
        unit = [0.0] * n_vars
        unit[i] = 1.0
        mat_i = np.array(lamb_func(*unit)) - constant
        coefficients.append(mat_i)

    return np.array(coefficients), constant


def serialize_emu_network(
    reactions_file: Path,
    output_file: Path,
    target_emus: dict,
    labeling_substrate: str = "A_ext",
    labeling_pattern: list = None,
):
    """Build EMU network with freeflux and serialize to .npz file.

    Parameters
    ----------
    reactions_file : Path
        Path to reactions.tsv file
    output_file : Path
        Path to output .npz file
    target_emus : dict
        Target EMUs {metabolite: [atom_patterns]}
    labeling_substrate : str
        Substrate with labeling (default "A_ext")
    labeling_pattern : list
        Labeling patterns (default ["11"] for fully labeled 2-carbon)
    """
    if labeling_pattern is None:
        labeling_pattern = ["11"]  # Fully labeled on both atoms

    print("=" * 80)
    print("EMU Matrix Serialization")
    print("=" * 80)
    print(f"\nReactions file: {reactions_file}")
    print(f"Output file: {output_file}")
    print(f"Target EMUs: {target_emus}")
    print(f"Labeling substrate: {labeling_substrate}")
    print(f"Labeling pattern: {labeling_pattern}")
    sys.stdout.flush()

    # Build model
    print("\nBuilding freeflux model...")
    sys.stdout.flush()

    model = Model("enzyme_emu")
    model.read_from_file(str(reactions_file))

    print(f"  Reactions: {model.reactions}")
    print(f"  Metabolites: {model.metabolites}")
    sys.stdout.flush()

    # Use INST simulator to build EMU network
    with model.simulator("inst") as isim:
        isim.set_target_EMUs(target_emus)
        isim.set_timepoints([0.0, 1.0])  # Dummy timepoints
        isim.set_labeling_strategy(
            labeling_substrate,
            labeling_pattern=labeling_pattern,
            percentage=[1.0],
            purity=[1.0],
        )

        # Set dummy flux values (we only need the matrix structure)
        for fid in model.totalfluxids:
            isim.set_flux(fid, 1.0)

        # Set dummy concentrations
        for mid in model.metabolites:
            isim.set_concentration(mid, 1.0)

        # Prepare the model (builds EMU network and matrices)
        print("\nPreparing EMU network...")
        sys.stdout.flush()
        isim.prepare(n_jobs=1)

        # Extract data from the model
        emu_model = isim.model

        # Get all EMU sizes present
        emu_sizes = sorted(emu_model.matrix_As.keys())
        print(f"  EMU sizes: {emu_sizes}")

        # Serialize data for each EMU size
        data = {}

        for size in emu_sizes:
            print(f"\nProcessing EMU size {size}...")
            sys.stdout.flush()

            lambA, flux_ids_A, product_emus = emu_model.matrix_As[size]
            lambB, flux_ids_B, source_emus = emu_model.matrix_Bs[size]
            lambM, metab_ids = emu_model.matrix_Ms[size]

            # Extract product EMU IDs
            product_emu_ids = [emu.id for emu in product_emus]
            print(f"  Product EMUs: {product_emu_ids}")

            # Extract source EMU info (may be tuples for convolutions)
            source_emu_info = []
            for semu in source_emus:
                if hasattr(semu, "id"):
                    source_emu_info.append(semu.id)
                elif hasattr(semu, "__iter__") and not isinstance(semu, str):
                    # Tuple of EMUs for convolution
                    source_emu_info.append([e.id for e in semu])
                else:
                    source_emu_info.append(str(semu))
            print(f"  Source EMUs: {source_emu_info}")

            # Extract matrix coefficients
            n_flux_A = len(flux_ids_A)
            n_flux_B = len(flux_ids_B)
            n_conc = len(metab_ids)

            print(f"  Flux IDs (A): {flux_ids_A}")
            print(f"  Flux IDs (B): {flux_ids_B}")
            print(f"  Metabolite IDs (M): {metab_ids}")

            # A matrix coefficients
            if n_flux_A > 0:
                A_coeffs, A_const = extract_matrix_coefficients(lambA, n_flux_A)
            else:
                # Evaluate with no arguments to get the constant matrix
                A_const = np.array(lambA())
                A_coeffs = np.zeros((0,) + A_const.shape)

            # B matrix coefficients
            if n_flux_B > 0:
                B_coeffs, B_const = extract_matrix_coefficients(lambB, n_flux_B)
            else:
                B_const = np.array(lambB())
                B_coeffs = np.zeros((0,) + B_const.shape)

            # M matrix coefficients (diagonal)
            if n_conc > 0:
                M_coeffs, M_const = extract_matrix_coefficients(lambM, n_conc)
            else:
                M_const = np.array(lambM())
                M_coeffs = np.zeros((0,) + M_const.shape)

            print(f"  A matrix shape: {A_const.shape}, coeffs shape: {A_coeffs.shape}")
            print(f"  B matrix shape: {B_const.shape}, coeffs shape: {B_coeffs.shape}")
            print(f"  M matrix shape: {M_const.shape}, coeffs shape: {M_coeffs.shape}")

            # Store in data dict
            data[f"size_{size}_A_coeffs"] = A_coeffs
            data[f"size_{size}_A_const"] = A_const
            data[f"size_{size}_B_coeffs"] = B_coeffs
            data[f"size_{size}_B_const"] = B_const
            data[f"size_{size}_M_coeffs"] = M_coeffs
            data[f"size_{size}_M_const"] = M_const
            data[f"size_{size}_flux_ids_A"] = np.array(flux_ids_A, dtype=object)
            data[f"size_{size}_flux_ids_B"] = np.array(flux_ids_B, dtype=object)
            data[f"size_{size}_metab_ids"] = np.array(metab_ids, dtype=object)
            data[f"size_{size}_product_emu_ids"] = np.array(product_emu_ids, dtype=object)
            data[f"size_{size}_source_emu_info"] = np.array(source_emu_info, dtype=object)

            # Initial states
            data[f"size_{size}_initial_X"] = np.array(emu_model.initial_matrix_Xs[size])
            data[f"size_{size}_initial_Y"] = np.array(emu_model.initial_matrix_Ys[size])

        # Store EMU sizes
        data["emu_sizes"] = np.array(emu_sizes)

        # Store substrate MDVs
        substrate_mdvs = {}
        for emu, mdv in emu_model.substrate_MDVs.items():
            emu_id = emu.id if hasattr(emu, "id") else str(emu)
            substrate_mdvs[emu_id] = np.array(mdv)

        # Override substrate MDVs with correct labeling patterns
        # A_ext is fully labeled (pattern "11")
        # Q_ext_in, P_ext_in are fully unlabeled
        substrate_mdvs_corrected = {}
        for emu_id, mdv in substrate_mdvs.items():
            if emu_id.startswith("A_ext"):
                # A_ext is fully labeled
                if len(mdv) == 2:  # 1-atom EMU
                    substrate_mdvs_corrected[emu_id] = np.array([0.0, 1.0])
                elif len(mdv) == 3:  # 2-atom EMU
                    substrate_mdvs_corrected[emu_id] = np.array([0.0, 0.0, 1.0])
                else:
                    substrate_mdvs_corrected[emu_id] = mdv
            elif emu_id.startswith("Q_ext") or emu_id.startswith("P_ext"):
                # Q_ext_in, P_ext_in are fully unlabeled
                if len(mdv) == 2:  # 1-atom EMU
                    substrate_mdvs_corrected[emu_id] = np.array([1.0, 0.0])
                elif len(mdv) == 3:  # 2-atom EMU
                    substrate_mdvs_corrected[emu_id] = np.array([1.0, 0.0, 0.0])
                else:
                    substrate_mdvs_corrected[emu_id] = mdv
            else:
                substrate_mdvs_corrected[emu_id] = mdv

        # Store substrate MDVs as separate arrays with keys
        substrate_emu_ids = list(substrate_mdvs_corrected.keys())
        data["substrate_emu_ids"] = np.array(substrate_emu_ids, dtype=object)
        for emu_id, mdv in substrate_mdvs_corrected.items():
            data[f"substrate_mdv_{emu_id}"] = mdv

        print(f"\nSubstrate EMUs (corrected labeling):")
        for emu_id, mdv in substrate_mdvs_corrected.items():
            print(f"  {emu_id}: {mdv}")

    # Save to npz file
    print(f"\nSaving to {output_file}...")
    sys.stdout.flush()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_file), **data)

    print(f"Done! File size: {output_file.stat().st_size / 1024:.1f} KB")
    sys.stdout.flush()

    return data


def main():
    """Main entry point."""
    serialize_emu_network(
        reactions_file=REACTIONS_FILE,
        output_file=OUTPUT_FILE,
        target_emus=TARGET_EMUS,
        labeling_substrate="A_ext",
        labeling_pattern=["11"],  # Fully labeled on both atoms
    )


if __name__ == "__main__":
    main()

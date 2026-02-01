#!/usr/bin/env python3
"""Serialize EMU network structure for the random uni-bi enzyme mechanism.

This script creates the emu_network.json file for the random mechanism model,
which is used by the pure JAX EMU simulation in enzyme_simulation_random_instationary.py.

The random mechanism has additional reactions compared to the ordered mechanism:
- v4a_f/v4a_b: EPQ <-> EP + Q (path 1: Q releases first)
- v4b_f/v4b_b: EPQ <-> EQ + P (path 2: P releases first)
- v5a_f/v5a_b: EP <-> E + P
- v5b_f/v5b_b: EQ <-> E + Q
"""

from pathlib import Path

from emu_network_serializer import serialize_emu_network, save_emu_network


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent


def serialize_random_network():
    """Serialize the simple_network_random model for enzyme_simulation_random_instationary.py."""
    reactions_file = PROJECT_DIR / "models" / "simple_network_random" / "reactions.tsv"
    output_path = PROJECT_DIR / "models" / "simple_network_random" / "emu_network.json"

    # Target EMUs for the random mechanism
    # Same metabolites as ordered, plus EQ enzyme intermediate
    target_emus = {
        "A": ["12"],       # 2-atom substrate
        "EA": ["12"],      # 2-atom enzyme-substrate complex
        "EPQ": ["12"],     # 2-atom enzyme-product complex
        "EP": ["1"],       # 1-atom enzyme-product (atom a)
        "EQ": ["1"],       # 1-atom enzyme-product (atom b) - NEW for random
        "Q": ["1"],        # 1-atom product
        "P": ["1"],        # 1-atom product
    }

    print("=" * 80)
    print("Serializing EMU Network for Random Uni-Bi Mechanism")
    print("=" * 80)
    print(f"\nReactions file: {reactions_file}")
    print(f"Output file: {output_path}")
    print(f"Target EMUs: {target_emus}")

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
    print(f"  Flux IDs: {network['flux_ids_all']}")
    print(f"  Metabolite IDs: {network['metab_ids_all']}")

    for size in network["sizes"]:
        print(f"\n  Size {size}:")
        print(f"    Product EMUs: {network['product_emu_ids'][str(size)]}")
        print(f"    Source EMUs: {network['source_emu_info'][str(size)]}")
        print(f"    A matrix structure: {len(network['matrix_A_structure'][str(size)])} non-zero entries")
        print(f"    B matrix structure: {len(network['matrix_B_structure'][str(size)])} non-zero entries")
        print(f"    M matrix structure: {len(network['matrix_M_structure'][str(size)])} non-zero entries")
        print(f"    Matrix shapes: A={network['matrix_shapes'][str(size)]['A']}, "
              f"B={network['matrix_shapes'][str(size)]['B']}, "
              f"M={network['matrix_shapes'][str(size)]['M']}")

    print(f"\nSubstrate MDVs:")
    for emu_id, mdv in network["substrate_mdvs"].items():
        print(f"  {emu_id}: {mdv}")

    return network


if __name__ == "__main__":
    serialize_random_network()

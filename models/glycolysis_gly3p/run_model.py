"""
Glycolysis + Pentose Phosphate Pathway Model - INST Forward Simulation

This script runs forward simulation of isotopically nonstationary (INST) 13C labeling
using freeflux with 1,2-13C glucose labeling. It predicts MDV timecourses given
known fluxes and concentrations.
"""

import numpy as np
import matplotlib.pyplot as plt
import toml
from freeflux import Model

MODEL_FILE = 'reactions.tsv'
FLUXES_FILE = 'glycolysis_fluxes.toml'
THERMO_FILE = 'glycolysis_thermodynamics.toml'

# Gas constant in kJ/(mol·K)
R = 8.314e-3

model = Model('glycolysis_gly3p')
model.read_from_file(MODEL_FILE)

# Define timepoints for simulation (logspaced from 1e-3 to 1e2 seconds)
TIMEPOINTS = list(np.logspace(-3, 2, 20))

# Store simulated MDV data for plotting
SIMULATED_MDVS = {}


def load_thermo_data(filepath):
    """Load thermodynamic data from TOML file.

    Returns dict of {rxn_id: deltaG_prime} in kJ/mol and temperature in K.
    """
    data = toml.load(filepath)

    temperature = data.get('conditions', {}).get('temperature', 298.15)

    thermo = {}
    for rxn_id, rxn_data in data.get('reactions', {}).items():
        thermo[rxn_id] = rxn_data.get('deltaG_prime', 0.0)

    return thermo, temperature


def net_to_forward_backward(v_net, deltaG_prime, temperature):
    """Convert net flux to forward and backward fluxes using Gibbs energy.

    For a reversible reaction at steady state:
        v_f / v_b = exp(-ΔG°'/(R*T))
        v_net = v_f - v_b

    Solving:
        v_b = v_net / (exp(-ΔG°'/(RT)) - 1)
        v_f = v_net * exp(-ΔG°'/(RT)) / (exp(-ΔG°'/(RT)) - 1)

    Args:
        v_net: Net flux (forward - backward)
        deltaG_prime: Standard Gibbs energy change in kJ/mol
        temperature: Temperature in K

    Returns:
        (v_forward, v_backward) tuple
    """
    exp_term = np.exp(-deltaG_prime / (R * temperature))

    # For very favorable reactions, backward flux is negligible
    if exp_term > 1e6:
        return abs(v_net), 0.0

    # For reactions near equilibrium (ΔG ≈ 0), handle numerical issues
    if abs(exp_term - 1) < 1e-6:
        v_f = abs(v_net) * 1000  # Large exchange flux
        v_b = v_f - v_net
        return v_f, v_b

    v_f = v_net * exp_term / (exp_term - 1)
    v_b = v_net / (exp_term - 1)

    return v_f, v_b


def load_fluxes_from_toml(flux_filepath, thermo_filepath):
    """Load fluxes from TOML file and convert units.

    TOML file has fluxes in mmol/gDCW/h.
    Freeflux needs fluxes in umol/gCDW/s when concentrations are in umol/gCDW.

    Conversion: 1 mmol/gDCW/h = 1000/3600 umol/gCDW/s = 0.2778 umol/gCDW/s

    For reversible reactions, uses Gibbs energy to calculate forward/backward fluxes.
    """
    data = toml.load(flux_filepath)
    thermo, temperature = load_thermo_data(thermo_filepath)

    fluxes = {}
    conversion_factor = 1000 / 3600  # mmol/h to umol/s

    # Combine external and internal fluxes
    for section in ['external_fluxes', 'internal_fluxes']:
        if section in data:
            for rxn_id, value in data[section].items():
                # Map TOML flux names to reaction IDs in the model
                # Some names differ (e.g., glc_uptake vs glc_input)
                if rxn_id == 'glc_uptake':
                    rxn_id = 'glc_input'
                elif rxn_id == 'pyr_secretion':
                    continue  # This is an output, handled by pyr_drain

                v_net = value * conversion_factor
                fluxes[rxn_id] = {
                    'net': v_net,
                    'deltaG': thermo.get(rxn_id, -5.0),  # Default -5 kJ/mol if not found
                    'temperature': temperature
                }

    return fluxes


def main():
    # Load fluxes from TOML file
    fluxes = load_fluxes_from_toml(FLUXES_FILE, THERMO_FILE)
    print(f"Loaded {len(fluxes)} fluxes from {FLUXES_FILE}")

    with model.simulator('inst') as isim:
        # Set target EMUs (metabolites and their carbon positions to track)
        isim.set_target_EMUs({
            'g6p': ['123456'],      # Full 6-carbon
            'f6p': ['123456'],      # Full 6-carbon
            'pgc6': ['123456'],     # Full 6-carbon (6-phosphogluconate)
            'r5p': ['12345'],       # Full 5-carbon
            'dhap': ['123'],        # Full 3-carbon
            'g3p': ['123'],         # Full 3-carbon
        })

        # Set timepoints for simulation
        isim.set_timepoints(TIMEPOINTS)

        # 1,2-13C glucose labeling (C1 and C2 labeled at 100%)
        isim.set_labeling_strategy(
            'Glc_ext',
            labeling_pattern=['110000'],  # C1,C2 labeled
            percentage=[1.0],
            purity=[1.0],
        )

        # Set fluxes from TOML file
        # Determine reversible reactions from model's expected flux IDs
        reversible_rxns = {fid.rsplit('_', 1)[0] for fid in model.totalfluxids if fid.endswith('_f')}
        print(f"Reversible reactions: {sorted(reversible_rxns)}")

        for rxn_id, flux_data in fluxes.items():
            v_net = flux_data['net']
            deltaG = flux_data['deltaG']
            temp = flux_data['temperature']

            if rxn_id in reversible_rxns:
                # Calculate forward and backward fluxes from thermodynamics
                v_f, v_b = net_to_forward_backward(v_net, deltaG, temp)
                isim.set_flux(f'{rxn_id}_f', v_f)
                isim.set_flux(f'{rxn_id}_b', v_b)
            else:
                # For irreversible reactions, set directly
                isim.set_flux(rxn_id, v_net)

        # Set metabolite concentrations (umol/gCDW)
        # Using typical cellular concentrations
        metabolite_concs = {
            'glc__D_e': 1.0,    # External glucose
            'glc__D': 1.0,      # Cytoplasmic glucose
            'g6p': 1.0,         # Glucose-6-phosphate
            'f6p': 1.0,         # Fructose-6-phosphate
            'fdp': 1.0,         # Fructose-1,6-bisphosphate
            'dhap': 1.0,        # Dihydroxyacetone phosphate
            'g3p': 1.0,         # Glyceraldehyde-3-phosphate
            'gly3p': 1.0,       # Glycerol-3-phosphate
            'pg3': 1.0,         # 3-Phosphoglycerate
            'pep': 1.0,         # Phosphoenolpyruvate
            'pyr': 1.0,         # Pyruvate
            'lac': 1.0,         # Lactate
            'pgc6': 1.0,        # 6-Phosphogluconate
            'ru5p__D': 1.0,     # Ribulose-5-phosphate
            'r5p': 1.0,         # Ribose-5-phosphate
            'xu5p__D': 1.0,     # Xylulose-5-phosphate
            's7p': 1.0,         # Sedoheptulose-7-phosphate
            'e4p': 1.0,         # Erythrose-4-phosphate
            'EC2': 0.01,        # Transketolase-bound 2-carbon carrier
            'EC3': 0.01,        # Transaldolase-bound 3-carbon carrier
        }

        for metab_id, conc in metabolite_concs.items():
            isim.set_concentration(metab_id, conc)

        print("Preparing INST simulator...")
        isim.prepare(n_jobs=1)

        print("Running forward simulation...")
        res = isim.simulate()

        # Extract simulated MDVs for plotting
        # res.simulated_MDV(emu_id) returns MDVs at all timepoints
        # res.timepoints includes 0 at the start
        sim_timepoints = res.timepoints
        for emu_id in res.simulated_EMUs:
            mdv_result = res.simulated_MDV(emu_id)
            mdv_list = []
            for t in sim_timepoints:
                mdv_list.append(np.array(mdv_result[t]))
            SIMULATED_MDVS[emu_id] = (sim_timepoints, mdv_list)

    # Plot the simulated MDV timecourses
    plot_mdv_timecourse()

    return res


def plot_mdv_timecourse():
    """Plot the simulated MDV timecourse for key metabolites."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Simulated MDV Timecourse - 1,2-¹³C Glucose Labeling\n(Glycolysis Model - Forward Simulation)',
                 fontsize=14, fontweight='bold')

    # Define which metabolites to plot and their properties
    # EMU IDs use underscore format: metabolite_carbons
    metabolites = [
        ('g6p_123456', 'G6P (full)', ['M+0', 'M+1', 'M+2', 'M+3', 'M+4', 'M+5', 'M+6']),
        ('f6p_123456', 'F6P (full)', ['M+0', 'M+1', 'M+2', 'M+3', 'M+4', 'M+5', 'M+6']),
        ('pgc6_123456', '6PGC (full)', ['M+0', 'M+1', 'M+2', 'M+3', 'M+4', 'M+5', 'M+6']),
        ('r5p_12345', 'R5P (full)', ['M+0', 'M+1', 'M+2', 'M+3', 'M+4', 'M+5']),
        ('dhap_123', 'DHAP (full)', ['M+0', 'M+1', 'M+2', 'M+3']),
        ('g3p_123', 'G3P (full)', ['M+0', 'M+1', 'M+2', 'M+3']),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (metab_id, title, labels) in enumerate(metabolites):
        ax = axes[idx // 3, idx % 3]

        if metab_id not in SIMULATED_MDVS:
            ax.text(0.5, 0.5, f'No data for {metab_id}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        timepoints, mdv_list = SIMULATED_MDVS[metab_id]
        mdv_data = np.array(mdv_list)

        # Filter out t=0 for log scale
        plot_mask = [t > 0 for t in timepoints]
        plot_times = [t for t in timepoints if t > 0]
        plot_mdv = mdv_data[plot_mask]

        # Plot each mass isotopomer
        for i, label in enumerate(labels):
            if i < plot_mdv.shape[1]:
                ax.plot(plot_times, plot_mdv[:, i], 'o-', color=colors[i],
                       label=label, linewidth=2, markersize=4)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fractional abundance')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlim(min(t for t in timepoints if t > 0) * 0.5, max(timepoints) * 1.5)

    plt.tight_layout()
    plt.savefig('mdv_timecourse.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdv_timecourse.pdf', bbox_inches='tight')
    print("\nPlots saved to mdv_timecourse.png and mdv_timecourse.pdf")
    plt.show()


if __name__ == "__main__":
    main()

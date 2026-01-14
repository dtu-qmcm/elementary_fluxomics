#!/usr/bin/env python3
"""Compare parameter estimates across simulation types.

Creates a plot showing 95% credible intervals for each parameter on log scale,
with true values as horizontal gray lines and dashed separators between parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_results():
    """Load MCMC results from all three simulations."""
    simulations = {
        "No Labeling": RESULTS_DIR / "enzyme_simulation_no_labeling" / "mcmc_results.npz",
        "Steady-State": RESULTS_DIR / "enzyme_simulation" / "mcmc_results.npz",
        "Instationary": RESULTS_DIR / "enzyme_simulation_instationary" / "mcmc_results.npz",
    }

    results = {}
    for name, path in simulations.items():
        if path.exists():
            data = np.load(path, allow_pickle=True)
            results[name] = {
                "samples": data["samples"],
                "log_samples": data["log_samples"],
                "param_names": data["param_names"],
                "reference": data["reference"],
            }
            print(f"Loaded {name}: {data['samples'].shape[0]} samples")
        else:
            print(f"Warning: {path} not found, skipping {name}")

    return results


def compute_credible_intervals(samples, percentiles=(2.5, 97.5)):
    """Compute credible intervals for each parameter."""
    lower = np.percentile(samples, percentiles[0], axis=0)
    upper = np.percentile(samples, percentiles[1], axis=0)
    median = np.percentile(samples, 50, axis=0)
    return lower, median, upper


def main():
    print("=" * 70)
    print("Parameter Comparison Plot")
    print("=" * 70)

    # Create figures directory if it doesn't exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results()

    if not results:
        print("Error: No results found. Run the simulations first.")
        return

    # Get parameter names from any result
    first_result = next(iter(results.values()))
    param_names = list(first_result["param_names"])
    reference = first_result["reference"]
    n_params = len(param_names)

    # Batlow colormap colors (perceptually uniform, colorblind-friendly)
    colors = {
        "No Labeling": "#011959",      # Dark blue (batlow 0.15)
        "Steady-State": "#a5a847",     # Yellow-green (batlow 0.5)
        "Instationary": "#fca50a",     # Orange (batlow 0.85)
    }

    # Set up figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # X positions for parameters
    x_positions = np.arange(n_params)

    # Offset for each simulation type to avoid overlap
    n_sims = len(results)
    offsets = np.linspace(-0.25, 0.25, n_sims)

    # Plot 95% CI for each simulation
    for idx, (sim_name, sim_data) in enumerate(results.items()):
        # Use log samples for log-scale plotting
        log_samples = sim_data["log_samples"]

        # Compute 95% CI on log scale
        lower, median, upper = compute_credible_intervals(log_samples)

        # Plot error bars (CI)
        offset = offsets[idx]
        color = colors.get(sim_name, "gray")

        ax.errorbar(
            x_positions + offset,
            median,
            yerr=[median - lower, upper - median],
            fmt='o',
            color=color,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            markersize=6,
            label=sim_name,
        )

    # Add horizontal gray lines for true values (log scale)
    log_reference = np.log(reference)
    for i, (x, log_true) in enumerate(zip(x_positions, log_reference)):
        ax.hlines(
            log_true,
            x - 0.4,
            x + 0.4,
            colors='gray',
            linestyles='-',
            linewidth=1.5,
            alpha=0.5,
        )

    # Add dashed vertical separators between parameters
    for x in x_positions[:-1] + 0.5:
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('log(parameter value)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Parameter', fontsize=12, fontweight='bold')

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "parameter_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")

    # Also save as PDF for publication quality
    pdf_path = FIGURES_DIR / "parameter_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()

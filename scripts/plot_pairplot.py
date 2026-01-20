#!/usr/bin/env python3
"""Create pairplot of posterior distributions from all three models.

Overlays the posterior samples from no-labeling, steady-state, and instationary
models with different colors to visualize parameter correlations and uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
            samples = data["log_samples"]

            # Handle different array shapes:
            # (n_samples, n_params) or (n_chains, n_samples, n_params)
            if samples.ndim == 3:
                n_chains, n_samples, n_params = samples.shape
                samples = samples.reshape(-1, n_params)
                total_samples = n_chains * n_samples
            else:
                total_samples = samples.shape[0]

            results[name] = {
                "log_samples": samples,
                "param_names": list(data["param_names"]),
                "reference": np.log(data["reference"]),
            }
            print(f"Loaded {name}: {total_samples} samples")
        else:
            print(f"Warning: {path} not found, skipping {name}")

    return results


def create_pairplot(results, max_samples=200):
    """Create a pairplot with all posteriors overlaid."""
    # Batlow colormap colors (perceptually uniform, colorblind-friendly)
    colors = {
        "No Labeling": "#011959",      # Dark blue
        "Steady-State": "#a5a847",     # Yellow-green
        "Instationary": "#fca50a",     # Orange
    }

    # Get parameter names from first result
    first_result = next(iter(results.values()))
    param_names = first_result["param_names"]
    reference = first_result["reference"]
    n_params = len(param_names)

    # Create figure
    fig, axes = plt.subplots(n_params, n_params, figsize=(16, 16))

    # Plot order (reverse so instationary is on top, most visible)
    plot_order = ["No Labeling", "Steady-State", "Instationary"]

    for sim_name in plot_order:
        if sim_name not in results:
            continue

        samples = results[sim_name]["log_samples"]
        color = colors[sim_name]

        # Subsample if needed
        if len(samples) > max_samples:
            idx = np.random.choice(len(samples), max_samples, replace=False)
            samples = samples[idx]

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram/KDE
                    ax.hist(samples[:, i], bins=30, alpha=0.5, color=color,
                            density=True, label=sim_name if i == 0 else None)
                    # Add true value line
                    ax.axvline(reference[i], color='gray', linestyle='-',
                              linewidth=1.5, alpha=0.7)
                elif i > j:
                    # Lower triangle: scatter plot
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.3,
                              s=5, color=color, rasterized=True)
                    # Add true value marker
                    ax.axhline(reference[i], color='gray', linestyle='-',
                              linewidth=0.5, alpha=0.5)
                    ax.axvline(reference[j], color='gray', linestyle='-',
                              linewidth=0.5, alpha=0.5)
                else:
                    # Upper triangle: leave empty or hide
                    ax.set_visible(False)

    # Set labels
    for i in range(n_params):
        # Y-axis labels on left column
        axes[i, 0].set_ylabel(param_names[i], fontsize=9)
        # X-axis labels on bottom row
        axes[n_params-1, i].set_xlabel(param_names[i], fontsize=9)

        # Remove tick labels from interior plots
        for j in range(n_params):
            if i < n_params - 1:
                axes[i, j].set_xticklabels([])
            if j > 0 and i != j:
                axes[i, j].set_yticklabels([])

    # Add legend to top-left
    handles = [plt.Line2D([0], [0], color=colors[name], linewidth=3, label=name)
               for name in plot_order if name in results]
    fig.legend(handles=handles, loc='upper right', fontsize=12,
               bbox_to_anchor=(0.98, 0.98))

    # Add title
    fig.suptitle('Posterior Distributions (log scale)', fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig


def main():
    print("=" * 70)
    print("Posterior Pairplot")
    print("=" * 70)

    # Create figures directory if it doesn't exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results()

    if not results:
        print("Error: No results found. Run the simulations first.")
        return

    # Create pairplot
    print("\nGenerating pairplot...")
    fig = create_pairplot(results)

    # Save figure
    output_path = FIGURES_DIR / "posterior_pairplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")

    # Also save as PDF for publication quality
    pdf_path = FIGURES_DIR / "posterior_pairplot.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()

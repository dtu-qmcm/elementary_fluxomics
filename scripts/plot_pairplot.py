#!/usr/bin/env python3
"""Create pairplot of posterior distributions from all three models.

Overlays the posterior samples from no-labeling, steady-state, and instationary
models with different colors to visualize parameter correlations and uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from style import apply_style, COLORS, COLOR_ORDER, DOUBLE_COL

apply_style()

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
    # Get parameter names from first result
    first_result = next(iter(results.values()))
    all_param_names = first_result["param_names"]

    # Filter to elementary reaction rate constants only
    elementary = ["k2r", "k3f", "k3r", "k4f", "k4r", "k5f", "k5r"]
    param_idx = [i for i, n in enumerate(all_param_names) if n in elementary]
    param_names = [all_param_names[i] for i in param_idx]
    reference = first_result["reference"][param_idx]
    n_params = len(param_names)

    # Subset samples for each result
    for name in results:
        results[name]["log_samples"] = results[name]["log_samples"][:, param_idx]

    # Create figure — full page width, square
    fig, axes = plt.subplots(n_params, n_params, figsize=(DOUBLE_COL, DOUBLE_COL))

    for sim_name in COLOR_ORDER:
        if sim_name not in results:
            continue

        samples = results[sim_name]["log_samples"]
        color = COLORS[sim_name]

        # Subsample if needed
        if len(samples) > max_samples:
            idx = np.random.choice(len(samples), max_samples, replace=False)
            samples = samples[idx]

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(samples[:, i], bins=30, alpha=0.6, color=color,
                            density=True, label=sim_name if i == 0 else None)
                    ax.axvline(reference[i], color='gray', linestyle='-',
                              linewidth=1.0, alpha=0.7)
                elif i > j:
                    # Lower triangle: scatter plot
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.7,
                              s=3, color=color, edgecolors='none', rasterized=True)
                    ax.axhline(reference[i], color='gray', linestyle='-',
                              linewidth=0.4, alpha=0.5)
                    ax.axvline(reference[j], color='gray', linestyle='-',
                              linewidth=0.4, alpha=0.5)
                else:
                    ax.set_visible(False)

    # Set labels
    for i in range(n_params):
        axes[i, 0].set_ylabel(param_names[i])
        axes[n_params-1, i].set_xlabel(param_names[i])

        for j in range(n_params):
            if i < n_params - 1:
                axes[i, j].set_xticklabels([])
            if j > 0 and i != j:
                axes[i, j].set_yticklabels([])

    fig.suptitle('Posterior Distributions (log scale)', fontweight='bold', y=0.995)

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
    plt.savefig(output_path)
    print(f"\nFigure saved to {output_path}")

    pdf_path = FIGURES_DIR / "posterior_pairplot.pdf"
    plt.savefig(pdf_path)
    print(f"PDF saved to {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()

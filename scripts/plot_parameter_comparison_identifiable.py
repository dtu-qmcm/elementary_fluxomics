#!/usr/bin/env python3
"""Compare parameter estimates across simulation types (identifiable-only MCMC).

Creates a plot with 95% credible intervals for the 7 identifiable parameters
on log scale, comparing no-labeling, steady-state, and instationary results.

The 4 unidentifiable parameters (k3f, k3r, k4f, k4r) are fixed to their
literature values and not shown.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from style import apply_style, COLORS, COLOR_ORDER, DOUBLE_COL

apply_style()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_results():
    """Load MCMC results from all three identifiable-only simulations."""
    simulations = {
        "No Labeling": RESULTS_DIR / "mcmc_identifiable_no_labeling" / "mcmc_results.npz",
        "Steady-State": RESULTS_DIR / "mcmc_identifiable_steady_state" / "mcmc_results.npz",
        "Instationary": RESULTS_DIR / "mcmc_identifiable_instationary" / "mcmc_results.npz",
    }

    results = {}
    for name, path in simulations.items():
        if path.exists():
            data = np.load(path, allow_pickle=True)
            samples = data["samples"]
            log_samples = data["log_samples"]

            if samples.ndim == 3:
                n_chains, n_samples, n_params = samples.shape
                samples = samples.reshape(-1, n_params)
                log_samples = log_samples.reshape(-1, n_params)
                total_samples = n_chains * n_samples
            else:
                total_samples = samples.shape[0]

            rhats = data.get("rhats", None)
            if rhats is not None:
                rhats = np.array(rhats)

            results[name] = {
                "samples": samples,
                "log_samples": log_samples,
                "param_names": list(data["param_names"]),
                "reference": data["reference"],
                "rhats": rhats,
            }
            rhat_info = f", R-hat: {rhats.min():.3f}-{rhats.max():.3f}" if rhats is not None else ""
            print(f"Loaded {name}: {total_samples} samples{rhat_info}")
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
    print("Parameter Comparison Plot (Identifiable-Only MCMC)")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()

    if not results:
        print("Error: No results found. Run the simulations first.")
        return

    first_result = next(iter(results.values()))
    param_names = first_result["param_names"]
    reference = first_result["reference"]
    n_params = len(param_names)

    # Single panel figure
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.65, DOUBLE_COL * 0.38))

    x_positions = np.arange(n_params)
    n_sims = len(results)
    offsets = np.linspace(-0.25, 0.25, n_sims)

    for idx, (sim_name, sim_data) in enumerate(results.items()):
        log_samples = sim_data["log_samples"]
        lower, median, upper = compute_credible_intervals(log_samples)
        offset = offsets[idx]
        color = COLORS.get(sim_name, "gray")

        ax.errorbar(
            x_positions + offset,
            median,
            yerr=[median - lower, upper - median],
            fmt='o',
            color=color,
            capsize=3,
            capthick=1.0,
            linewidth=1.0,
            markersize=4,
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

    ax.set_xticks(x_positions)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_ylabel('log(parameter value)')
    ax.set_xlabel('Parameter')

    ax.legend(loc='upper right')

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.95)

    output_path = FIGURES_DIR / "parameter_comparison_identifiable.png"
    plt.savefig(output_path)
    print(f"\nFigure saved to {output_path}")

    pdf_path = FIGURES_DIR / "parameter_comparison_identifiable.pdf"
    plt.savefig(pdf_path)
    print(f"PDF saved to {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()

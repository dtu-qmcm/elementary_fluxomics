#!/usr/bin/env python3
"""Compare parameter estimates across simulation types.

Creates a combined plot with:
- Left (2:1 ratio): 95% credible intervals for each parameter on log scale
- Right: Histograms of k_f/k_r ratios for intermediate steps (k3, k4, k5)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from style import apply_style, COLORS, COLOR_ORDER, DOUBLE_COL

apply_style()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_scripts" / "mcmc_11_params"
FIGURES_DIR = RESULTS_DIR / "figures"


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
            samples = data["samples"]
            log_samples = data["log_samples"]

            # Handle different array shapes:
            # (n_samples, n_params) or (n_chains, n_samples, n_params)
            if samples.ndim == 3:
                # Flatten chains: (n_chains, n_samples, n_params) -> (n_chains*n_samples, n_params)
                n_chains, n_samples, n_params = samples.shape
                samples = samples.reshape(-1, n_params)
                log_samples = log_samples.reshape(-1, n_params)
                total_samples = n_chains * n_samples
            else:
                total_samples = samples.shape[0]

            # Load R-hat values if available
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


def compute_ratios(samples, param_names, reference):
    """Compute k_f/k_r ratios for intermediate steps."""
    # Find indices for k3f, k3r, k4f, k4r, k5f, k5r
    ratio_pairs = [("k3f", "k3r"), ("k4f", "k4r"), ("k5f", "k5r")]

    ratios = {}
    ref_ratios = {}
    for kf_name, kr_name in ratio_pairs:
        if kf_name in param_names and kr_name in param_names:
            kf_idx = param_names.index(kf_name)
            kr_idx = param_names.index(kr_name)
            ratio_name = f"{kf_name}/{kr_name}"
            ratios[ratio_name] = samples[:, kf_idx] / samples[:, kr_idx]
            ref_ratios[ratio_name] = reference[kf_idx] / reference[kr_idx]

    return ratios, ref_ratios


def main():
    print("=" * 70)
    print("Parameter Comparison Plot with Ratios")
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
    param_names = first_result["param_names"]
    reference = first_result["reference"]
    n_params = len(param_names)

    # Set up figure with GridSpec for 2:1 ratio
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.2)

    # Left panel: parameter comparison
    ax_params = fig.add_subplot(gs[0, 0])

    # Right panel: ratio comparison (same style as left)
    ax_ratios = fig.add_subplot(gs[0, 1])

    # ========== Left Panel: Parameter Comparison ==========
    x_positions = np.arange(n_params)
    n_sims = len(results)
    offsets = np.linspace(-0.25, 0.25, n_sims)

    for idx, (sim_name, sim_data) in enumerate(results.items()):
        log_samples = sim_data["log_samples"]
        lower, median, upper = compute_credible_intervals(log_samples)
        offset = offsets[idx]
        color = COLORS.get(sim_name, "gray")

        ax_params.errorbar(
            x_positions + offset,
            median,
            yerr=[median - lower, upper - median],
            fmt='o',
            color=color,
            capsize=3,
            capthick=1.0,
            linewidth=1.0,
            markersize=4,
        )

    # Add horizontal gray lines for true values (log scale)
    log_reference = np.log(reference)
    for i, (x, log_true) in enumerate(zip(x_positions, log_reference)):
        ax_params.hlines(
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
        ax_params.axvline(x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax_params.set_xticks(x_positions)
    ax_params.set_xticklabels(param_names, rotation=45, ha='right')
    ax_params.set_ylabel('log(parameter value)')
    ax_params.set_xlabel('Parameter')

    # ========== Right Panel: Ratio Comparison (same style as left) ==========
    ratio_names = ["k3f/k3r", "k4f/k4r", "k5f/k5r"]
    ratio_labels = [r"$k_{3f}/k_{3r}$", r"$k_{4f}/k_{4r}$", r"$k_{5f}/k_{5r}$"]
    n_ratios = len(ratio_names)
    x_ratio_positions = np.arange(n_ratios)

    # Get reference ratios (same for all simulations)
    first_sim = next(iter(results.values()))
    _, ref_ratios = compute_ratios(first_sim["samples"], first_sim["param_names"], first_sim["reference"])

    for idx, (sim_name, sim_data) in enumerate(results.items()):
        samples = sim_data["samples"]
        param_names_sim = sim_data["param_names"]
        ref = sim_data["reference"]

        ratios, _ = compute_ratios(samples, param_names_sim, ref)
        offset = offsets[idx]
        color = COLORS.get(sim_name, "gray")

        # Compute log ratios and their CIs
        medians = []
        lowers = []
        uppers = []
        for ratio_name in ratio_names:
            if ratio_name in ratios:
                log_ratio = np.log(ratios[ratio_name])
                lower, median, upper = compute_credible_intervals(log_ratio.reshape(-1, 1))
                medians.append(median[0])
                lowers.append(lower[0])
                uppers.append(upper[0])

        medians = np.array(medians)
        lowers = np.array(lowers)
        uppers = np.array(uppers)

        ax_ratios.errorbar(
            x_ratio_positions + offset,
            medians,
            yerr=[medians - lowers, uppers - medians],
            fmt='o',
            color=color,
            capsize=3,
            capthick=1.0,
            linewidth=1.0,
            markersize=4,
        )

    # Add horizontal gray lines for true ratio values (log scale)
    for i, ratio_name in enumerate(ratio_names):
        if ratio_name in ref_ratios:
            log_ref = np.log(ref_ratios[ratio_name])
            ax_ratios.hlines(
                log_ref,
                i - 0.4,
                i + 0.4,
                colors='gray',
                linestyles='-',
                linewidth=1.5,
                alpha=0.5,
            )

    # Add dashed vertical separators between ratios
    for x in x_ratio_positions[:-1] + 0.5:
        ax_ratios.axvline(x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax_ratios.set_xticks(x_ratio_positions)
    ax_ratios.set_xticklabels(ratio_labels)
    ax_ratios.set_ylabel('log(ratio)')
    ax_ratios.set_xlabel('Ratio')

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.95)

    # Save figure
    output_path = FIGURES_DIR / "parameter_comparison.png"
    plt.savefig(output_path)
    print(f"\nFigure saved to {output_path}")

    pdf_path = FIGURES_DIR / "parameter_comparison.pdf"
    plt.savefig(pdf_path)
    print(f"PDF saved to {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()

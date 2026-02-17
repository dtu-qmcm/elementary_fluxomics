#!/usr/bin/env python3
"""Plot free energy profiles (transition state diagrams) for the ordered uni-bi mechanism.

Shows Gibbs energy of each enzyme state and transition state barriers
computed from rate constants using microscopic reversibility and the Eyring equation.
Overlays 95% credible intervals from MCMC parameter estimates.

Produces two figures:
  - 7 free parameters (k3f, k3r, k4f, k4r fixed to true values)
  - 11 free parameters (all rate constants sampled)

Catalytic cycle:
    E + A ⇌ EA ⇌ EPQ ⇌ EP + Q ⇌ E + P
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from style import apply_style, COLORS, DOUBLE_COL
from enzyme_utils import (
    DEFAULT_PARAMS,
    compute_wegscheider_constraints,
    R_GAS,
    TEMPERATURE,
)

apply_style()

SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)

STATE_LABELS = ["E + A", "EA", "EPQ", "EP + Q", "E + P"]
TS_LABELS = ["TS1", "TS2", "TS3", "TS4"]

# Two sets of MCMC experiments: (display_name, results_dir)
MCMC_7_PARAMS = [
    ("No Labeling", "mcmc_identifiable_no_labeling"),
    ("Steady-State", "mcmc_identifiable_steady_state"),
    ("Instationary", "mcmc_identifiable_instationary"),
]

MCMC_11_PARAMS = [
    ("No Labeling", "enzyme_simulation_no_labeling"),
    ("Steady-State", "enzyme_simulation"),
    ("Instationary", "enzyme_simulation_instationary"),
]


def compute_state_energies(params):
    """Compute cumulative Gibbs energies for each enzyme state.

    Uses ΔG_step = -RT * ln(kf / kr) for each elementary step.
    Reference: G(E + A) = 0.
    """
    RT = R_GAS * TEMPERATURE
    steps = [
        ("k2f", "k2r"),  # E + A ⇌ EA
        ("k3f", "k3r"),  # EA ⇌ EPQ
        ("k4f", "k4r"),  # EPQ ⇌ EP + Q
        ("k5f", "k5r"),  # EP ⇌ E + P
    ]
    energies = [0.0]  # G(E + A) = 0
    for kf_name, kr_name in steps:
        dG = -RT * np.log(params[kf_name] / params[kr_name])
        energies.append(energies[-1] + dG)
    return np.array(energies)


def compute_transition_state_energies(params, state_energies):
    """Compute transition state energies using the Eyring equation.

    ΔG‡ = -RT * ln(k / (kB*T/h))
    G(TS) = G(reactant) + ΔG‡_forward
    """
    RT = R_GAS * TEMPERATURE
    kBT_h = K_B * TEMPERATURE / H_PLANCK  # ~6.212e12 s^-1
    forward_ks = [params["k2f"], params["k3f"], params["k4f"], params["k5f"]]
    ts_energies = []
    for i, k in enumerate(forward_ks):
        dG_barrier = -RT * np.log(k / kBT_h)
        ts_energies.append(state_energies[i] + dG_barrier)
    return np.array(ts_energies)


def samples_to_energy_arrays(samples, param_names):
    """Convert MCMC samples to arrays of state and TS energies.

    Returns:
        state_energies: (n_samples, 5) array
        ts_energies: (n_samples, 4) array
    """
    n_samples = samples.shape[0]
    state_energies = np.zeros((n_samples, 5))
    ts_energies = np.zeros((n_samples, 4))

    name_to_idx = {name: i for i, name in enumerate(param_names)}

    for s in range(n_samples):
        params = DEFAULT_PARAMS.copy()
        for name in param_names:
            params[name] = float(samples[s, name_to_idx[name]])
        params = compute_wegscheider_constraints(params)

        state_energies[s] = compute_state_energies(params)
        ts_energies[s] = compute_transition_state_energies(params, state_energies[s])

    return state_energies, ts_energies


def load_mcmc_results(experiment_name, dir_name):
    """Load MCMC results and compute energy distributions."""
    path = RESULTS_DIR / dir_name / "mcmc_results.npz"
    data = np.load(path, allow_pickle=True)
    samples = data["samples"]
    param_names = list(data["param_names"])

    if samples.ndim == 3:
        samples = samples.reshape(-1, samples.shape[-1])

    print(f"  {experiment_name}: {samples.shape[0]} samples, "
          f"{len(param_names)} params")

    return samples_to_energy_arrays(samples, param_names)


def plot_diagram(state_energies, ts_energies, mcmc_results, title, filename):
    """Plot a transition state diagram with MCMC credible intervals."""

    x_states = np.array([0, 2.5, 5, 7.5, 10], dtype=float)
    x_ts = np.array([1.25, 3.75, 6.25, 8.75], dtype=float)

    n_exp = len(mcmc_results)
    bar_hw = 0.25
    gap = 0.08
    group_width = n_exp * 2 * bar_hw + (n_exp - 1) * gap
    mcmc_offsets = np.linspace(-group_width / 2 + bar_hw,
                               group_width / 2 - bar_hw, n_exp)
    true_hw = group_width / 2 + 0.05

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))

    # --- True bars (underlaid) ---
    for x, G in zip(x_states, state_energies):
        ax.plot(
            [x - true_hw, x + true_hw], [G, G],
            color="k", linewidth=2.0, solid_capstyle="round", zorder=1,
        )
    for x, G in zip(x_ts, ts_energies):
        ax.plot(
            [x - true_hw * 0.7, x + true_hw * 0.7], [G, G],
            color="k", linewidth=1.4, linestyle="--", zorder=1,
        )

    # --- MCMC bars ---
    exp_names = list(mcmc_results.keys())
    for exp_idx, exp_name in enumerate(exp_names):
        mcmc_state, mcmc_ts = mcmc_results[exp_name]
        color = COLORS[exp_name]
        offset = mcmc_offsets[exp_idx]

        state_lo = np.percentile(mcmc_state, 2.5, axis=0)
        state_med = np.percentile(mcmc_state, 50, axis=0)
        state_hi = np.percentile(mcmc_state, 97.5, axis=0)

        for i, x in enumerate(x_states):
            xc = x + offset
            ax.plot([xc, xc], [state_lo[i], state_hi[i]],
                    color=color, linewidth=1.4, zorder=3)
            ax.plot([xc - bar_hw, xc + bar_hw], [state_med[i], state_med[i]],
                    color=color, linewidth=2.0, solid_capstyle="round", zorder=4)

        ts_lo = np.percentile(mcmc_ts, 2.5, axis=0)
        ts_med = np.percentile(mcmc_ts, 50, axis=0)
        ts_hi = np.percentile(mcmc_ts, 97.5, axis=0)

        for i, x in enumerate(x_ts):
            xc = x + offset
            ax.plot([xc, xc], [ts_lo[i], ts_hi[i]],
                    color=color, linewidth=1.4, zorder=3)
            ax.plot([xc - bar_hw * 0.7, xc + bar_hw * 0.7],
                    [ts_med[i], ts_med[i]],
                    color=color, linewidth=1.4, linestyle="--", zorder=4)

    # --- State labels ---
    label_layout = {
        "E + A":  -9,
        "EA":      3,
        "EPQ":     3,
        "EP + Q": -9,
        "E + P":  -9,
    }
    for x, G, label in zip(x_states, state_energies, STATE_LABELS):
        dy = label_layout[label]
        ax.annotate(
            label, xy=(x, G), xytext=(0, dy),
            textcoords="offset points",
            ha="center", va="top" if dy < 0 else "bottom",
            fontsize=7,
        )

    # --- Barrier height annotations ---
    for i, (x, ts_G) in enumerate(zip(x_ts, ts_energies)):
        dG_barrier = ts_G - state_energies[i]
        ax.annotate(
            r"$\Delta G^{\ddag}$" + f" = {dG_barrier:.1f}",
            xy=(x, ts_G), xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=5.5, color="0.35",
        )

    # --- Legend ---
    legend_handles = [
        Line2D([0], [0], color="k", linewidth=2.0, label="True"),
    ]
    for exp_name in exp_names:
        legend_handles.append(
            Line2D([0], [0], color=COLORS[exp_name], linewidth=2.0,
                   label=f"{exp_name} (95% CI)")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6)

    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Gibbs free energy (kJ/mol)")
    ax.set_xlabel("Reaction coordinate")
    ax.set_xticks([])

    y_min = min(state_energies) - 8
    y_max = max(ts_energies) + 8
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-1.2, 11.7)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.93)

    for fmt in ("pdf", "png"):
        path = FIGURES_DIR / f"{filename}.{fmt}"
        plt.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # True parameter energies
    params = compute_wegscheider_constraints(DEFAULT_PARAMS)
    state_energies = compute_state_energies(params)
    ts_energies = compute_transition_state_energies(params, state_energies)

    print("True state energies (kJ/mol):")
    for label, G in zip(STATE_LABELS, state_energies):
        print(f"  {label:>8s}: {G:+.2f}")
    print(f"Overall ΔG = {state_energies[-1] - state_energies[0]:.2f} kJ/mol")

    print(f"\nTrue transition state energies (kJ/mol):")
    for label, G in zip(TS_LABELS, ts_energies):
        print(f"  {label}: {G:.1f}")

    # --- 7 free parameters ---
    print("\n" + "=" * 60)
    print("7 free parameters (k3f, k3r, k4f, k4r fixed)")
    print("=" * 60)
    mcmc_7 = {}
    for exp_name, dir_name in MCMC_7_PARAMS:
        mcmc_state, mcmc_ts = load_mcmc_results(exp_name, dir_name)
        mcmc_7[exp_name] = (mcmc_state, mcmc_ts)

    plot_diagram(
        state_energies, ts_energies, mcmc_7,
        title="7 free parameters",
        filename="transition_state_diagram_7params",
    )

    # --- 11 free parameters ---
    print("\n" + "=" * 60)
    print("11 free parameters (all rate constants sampled)")
    print("=" * 60)
    mcmc_11 = {}
    for exp_name, dir_name in MCMC_11_PARAMS:
        mcmc_state, mcmc_ts = load_mcmc_results(exp_name, dir_name)
        mcmc_11[exp_name] = (mcmc_state, mcmc_ts)

    plot_diagram(
        state_energies, ts_energies, mcmc_11,
        title="11 free parameters",
        filename="transition_state_diagram_11params",
    )


if __name__ == "__main__":
    main()

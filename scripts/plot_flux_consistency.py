#!/usr/bin/env python3
"""Plot comparing enzyme_ode and isotopomer_ode steady-state consistency."""

import sys
from pathlib import Path

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import diffrax
from diffrax import SteadyStateEvent
import matplotlib.pyplot as plt
import numpy as np
from isotopomer_simulation import (
    isotopomer_ode,
    enzyme_ode,
    DEFAULT_PARAMS,
    DEFAULT_INITIAL_TOTAL,
    build_initial_state,
)


def simulate_enzyme_to_steady_state(params, initial_array):
    """Simulate enzyme_ode to steady state."""
    term = diffrax.ODETerm(enzyme_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-12, atol=1e-14)
    discrete_terminating_event = SteadyStateEvent(rtol=1e-10, atol=1e-12)
    saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=jnp.inf, dt0=1e-10, y0=initial_array,
        args=params, saveat=saveat, stepsize_controller=stepsize_controller,
        max_steps=int(1e7), discrete_terminating_event=discrete_terminating_event,
        throw=False,
    )
    return sol.ys[0] if sol.ys.ndim > 1 else sol.ys


def simulate_isotopomer_timecourse(params, initial_total, t_eval):
    """Simulate isotopomer_ode over a timecourse."""
    y0 = build_initial_state(initial_total)
    term = diffrax.ODETerm(isotopomer_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-12, atol=1e-14)
    saveat = diffrax.SaveAt(ts=t_eval)

    sol = diffrax.diffeqsolve(
        term, solver, t0=t_eval[0], t1=t_eval[-1], dt0=1e-6, y0=y0,
        args=params, saveat=saveat, stepsize_controller=stepsize_controller,
        max_steps=int(1e7), throw=False,
    )
    return sol


def sum_isotopomers(y):
    """Sum isotopomers to get total concentrations [A, Q, P, EA, EPQ, EP]."""
    A = y[0] + y[1] + y[2] + y[3]
    EA = y[4] + y[5] + y[6] + y[7]
    EPQ = y[8] + y[9] + y[10] + y[11]
    EP = y[12] + y[13]
    Q = y[14] + y[15]
    P = y[16] + y[17]
    return jnp.array([A, Q, P, EA, EPQ, EP])


def sum_isotopomer_derivatives(dydt):
    """Sum isotopomer derivatives to get total dX/dt for each metabolite.

    Returns [dA/dt, dQ/dt, dP/dt, dEA/dt, dEPQ/dt, dEP/dt].
    """
    dA = dydt[0] + dydt[1] + dydt[2] + dydt[3]
    dEA = dydt[4] + dydt[5] + dydt[6] + dydt[7]
    dEPQ = dydt[8] + dydt[9] + dydt[10] + dydt[11]
    dEP = dydt[12] + dydt[13]
    dQ = dydt[14] + dydt[15]
    dP = dydt[16] + dydt[17]
    return jnp.array([dA, dQ, dP, dEA, dEPQ, dEP])


def compute_net_fluxes(y, params):
    """Compute net fluxes from state vector."""
    A, Q, P, EA, EPQ, EP = y[0], y[1], y[2], y[3], y[4], y[5]
    E = params["E_T"] - EA - EPQ - EP

    return {
        "v_A_ex": float(params["k_A_ex"] * (params["A_e"] - A)),
        "v2": float(A * E * params["k2f"] - EA * params["k2r"]),
        "v3": float(EA * params["k3f"] - EPQ * params["k3r"]),
        "v4": float(EPQ * params["k4f"] - Q * EP * params["k4r"]),
        "v5": float(EP * params["k5f"] - E * P * params["k5r"]),
        "v_Q_ex": float(params["k_Q_ex"] * (params["Q_e"] - Q)),
        "v_P_ex": float(params["k_P_ex"] * (params["P_e"] - P)),
        "v_pq": float(P * params["k_pq_f"] - Q * params["k_pq_r"]),
    }


def main():
    params = DEFAULT_PARAMS.copy()
    initial_total = DEFAULT_INITIAL_TOTAL.copy()
    initial_array = jnp.array([
        initial_total["A"], initial_total["Q"], initial_total["P"],
        initial_total["EA"], initial_total["EPQ"], initial_total["EP"],
    ])

    # Run steady-state simulation for enzyme_ode
    enzyme_ss = simulate_enzyme_to_steady_state(params, initial_array)

    # Run timecourse simulation for isotopomer_ode to t=1e3
    t_eval = jnp.linspace(0, 1000, 200)
    iso_sol = simulate_isotopomer_timecourse(params, initial_total, t_eval)

    # Compute summed concentrations and derivatives at each timepoint
    n_times = len(iso_sol.ts)
    iso_totals_timecourse = np.zeros((n_times, 6))
    iso_dydt_summed = np.zeros((n_times, 6))
    enzyme_dydt = np.zeros((n_times, 6))

    for i in range(n_times):
        y_iso = iso_sol.ys[i]
        # Sum isotopomers to get total concentrations
        iso_totals_timecourse[i] = np.array(sum_isotopomers(y_iso))
        # Compute isotopomer derivatives and sum them
        dydt_iso = isotopomer_ode(iso_sol.ts[i], y_iso, params)
        iso_dydt_summed[i] = np.array(sum_isotopomer_derivatives(dydt_iso))
        # Compute enzyme_ode derivatives using summed concentrations
        y_enzyme = jnp.array(iso_totals_timecourse[i])
        enzyme_dydt[i] = np.array(enzyme_ode(iso_sol.ts[i], y_enzyme, params))

    # Get final state (t=1e3) from isotopomer simulation
    iso_totals_final = np.array(sum_isotopomers(iso_sol.ys[-1]))
    enzyme_ss_array = np.array(enzyme_ss)

    # Compute fluxes at t=1e3 for both systems
    iso_fluxes = compute_net_fluxes(iso_totals_final, params)
    enzyme_fluxes = compute_net_fluxes(enzyme_ss_array, params)

    # Create figure with 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    conc_names = ["A", "Q", "P", "EA", "EPQ", "EP"]

    # Panel A: Concentration absolute error at t=1e3
    ax = axes[0]
    conc_abs_error = np.abs(iso_totals_final - enzyme_ss_array)

    x = np.arange(len(conc_names))
    ax.bar(x, conc_abs_error, color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conc_names)
    ax.set_ylabel("Absolute Error")
    ax.set_title("(A) Concentration Error at t=1000\n(|Σ isotopomer − enzyme SS|)")
    ax.set_yscale("log")

    # Add max error annotation
    max_conc_err = np.max(conc_abs_error)
    ax.text(0.95, 0.95, f"Max: {max_conc_err:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel B: Flux absolute error at t=1e3
    ax = axes[1]
    flux_names = list(iso_fluxes.keys())
    iso_flux_vals = np.array([iso_fluxes[k] for k in flux_names])
    enzyme_flux_vals = np.array([enzyme_fluxes[k] for k in flux_names])
    flux_abs_error = np.abs(iso_flux_vals - enzyme_flux_vals)

    x = np.arange(len(flux_names))
    ax.bar(x, flux_abs_error, color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(flux_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Absolute Error")
    ax.set_title("(B) Net Flux Error at t=1000\n(|Σ isotopomer − enzyme SS|)")
    ax.set_yscale("log")

    # Add max error annotation
    max_flux_err = np.max(flux_abs_error)
    ax.text(0.95, 0.95, f"Max: {max_flux_err:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel C: Difference in dX/dt between isotopomer and enzyme ODEs over time
    ax = axes[2]
    times = np.array(iso_sol.ts)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    # Compute difference: isotopomer - enzyme
    diff_dydt = iso_dydt_summed - enzyme_dydt

    for i, name in enumerate(conc_names):
        ax.plot(times, diff_dydt[:, i], "-", color=colors[i], linewidth=1.5, label=name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Δ(dX/dt)")
    ax.set_title("(C) Derivative Difference\n(Σ isotopomer − enzyme)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)

    # Add max error annotation
    max_diff = np.max(np.abs(diff_dydt))
    ax.text(0.95, 0.95, f"Max |diff|: {max_diff:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = "results/flux_consistency.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # Print numerical comparison
    print("\n" + "="*60)
    print("Numerical Comparison at t=1000")
    print("="*60)
    print("\nConcentrations:")
    print(f"{'Species':<8} {'Isotopomer':<12} {'Enzyme SS':<12} {'|Error|':<12}")
    for i, name in enumerate(conc_names):
        print(f"{name:<8} {iso_totals_final[i]:<12.6f} {enzyme_ss_array[i]:<12.6f} {conc_abs_error[i]:<12.2e}")

    print("\nNet Fluxes:")
    print(f"{'Flux':<10} {'Isotopomer':<12} {'Enzyme SS':<12} {'|Error|':<12}")
    for i, name in enumerate(flux_names):
        print(f"{name:<10} {iso_flux_vals[i]:<12.6f} {enzyme_flux_vals[i]:<12.6f} {flux_abs_error[i]:<12.2e}")


if __name__ == "__main__":
    main()

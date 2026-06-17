#!/usr/bin/env python3
"""Likelihood-ratio confidence contours via rotating-ray + optimistix.

Instead of a brute-force 2D grid (see scripts/gibbs_landscape_2d.py, which this
does NOT replace), this traces the likelihood-ratio confidence boundary for the
two-parameter FBA1/FBA2 Gibbs-energy system directly.

We know the *true point* that generates the reference ("observed") MDVs, so the
LR statistic is ~0 there and grows away from it:

    LR(theta, d) = sum_over_data (log(mdv_test+EPS) - log(mdv_ref+EPS))^2 / scale^2

The INST statistic is additionally divided by the number of timepoints, so its
per-measurement magnitude is comparable to the single-snapshot SS statistic
(normalizes for the extra measurements INST takes over the timecourse).

From the true point we cast a ray at angle theta (a unit vector in raw kJ/mol):

    FBA1 = true1 + d*cos(theta),  FBA2 = true2 + d*sin(theta)

and for each significance level alpha we solve for the distance d where LR
crosses the chi-squared threshold chi2.ppf(1 - alpha, df=2). This is the
(1 - alpha) confidence region (df=2 -> joint region over 2 params; df=1 would
give per-parameter profile intervals). The crossing is found with optimistix
Bisection. Because the objective calls freeflux (numpy, not JAX-traceable) we
wrap it in jax.pure_callback; optimistix's root_find is internally jitted, so a
plain-Python objective cannot be used directly.

Cases: {instationary (INST), steady-state (SS)} x {far-from-equilibrium (FFE),
near-equilibrium (NE)} = 4. First attempt: 10 angles, alpha [0.9,0.75,0.5]
(i.e. the 10%, 25%, 50% confidence regions).

Usage:
    uv run python scripts/gibbs_confidence_contours.py          # full 4-case run
    uv run python scripts/gibbs_confidence_contours.py --smoke  # fast sanity run
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)  # float64 for robust root-finding

import cmcrameri.cm as cmc
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
import pandas as pd
from scipy.stats import chi2

# Reuse simulation machinery and constants from the sibling landscape script.
from gibbs_landscape_2d import (
    EPSILON,
    LOGNORMAL_SCALE,
    MODEL_DIR,
    PROJECT_DIR,
    TIMEPOINTS,
    run_simulation,
)
from style import DOUBLE_COL, apply_style

# ============================================================================
# Configuration
# ============================================================================

# Significance levels (alpha). The confidence region at confidence (1 - alpha)
# uses the chi-squared threshold chi2.ppf(1 - alpha, df).
ALPHA_LEVELS = [0.9, 0.75, 0.5]
DF = 2  # 2 free parameters (FBA1, FBA2) -> joint confidence region (Wilks)
THRESHOLDS = {a: float(chi2.ppf(1.0 - a, df=DF)) for a in ALPHA_LEVELS}
# -> {0.9: 0.2107, 0.75: 0.5754, 0.5: 1.3863}  (the 10%, 25%, 50% conf regions)

N_ANGLES = 10

# Ray cap window (kJ/mol): keep coordinates physically sane. Positive dG flips
# flux direction and destabilizes freeflux, so keep DG_HI small.
DG_LO, DG_HI = -15.0, 2.0

# Bisection tolerances on the distance d. rtol=1e-2 is loose enough that even
# very steep crossings (NE INST, where LR explodes over ~0.005 kJ/mol) satisfy
# optimistix's residual test and converge in ~17 steps; tighter rtol makes the
# steep cases run to max_steps without ever flagging convergence (the value is
# still correct to ~1%, which is plenty for a contour).
BISECTION_RTOL = 1e-2
BISECTION_ATOL = 1e-3

# Bracket search: initial probe distance and max doublings.
D_START = 0.5
MAX_BRACKET_DOUBLINGS = 40

EQUILIBRIUM_CASES = {
    "ffe": {
        "thermo_file": MODEL_DIR / "glycolysis_thermodynamics.toml",
        "true1": -2.545,  # total_fba_dg / 2 (= -5.09 / 2)
        "true2": -2.545,
        "label": "Far-from-equilibrium",
    },
    "ne": {
        "thermo_file": MODEL_DIR / "glycolysis_thermodynamics_ne.toml",
        "true1": -0.435,  # total_fba_dg / 2 (= -0.87 / 2)
        "true2": -0.435,
        "label": "Near-equilibrium",
    },
}
SIM_MODES = ["inst", "ss"]
RESULTS_DIR = PROJECT_DIR / "results" / "gibbs_confidence_contours"


# ============================================================================
# Chi-squared / likelihood-ratio statistic (pure numpy)
# ============================================================================

def _normalize(mdv):
    """Clip negatives and normalize an MDV to sum 1 (mirrors landscape script)."""
    mdv = np.maximum(np.asarray(mdv, dtype=float), 0.0)
    s = mdv.sum()
    return mdv / s if s > 0 else mdv


def _chisq_pair(mdv_ref, mdv_test, scale):
    """Log-normal chi-squared between two MDV vectors."""
    r = _normalize(mdv_ref)
    t = _normalize(mdv_test)
    return float(np.sum((np.log(t + EPSILON) - np.log(r + EPSILON)) ** 2 / scale**2))


def chisq_timecourse(ref_tc, test_tc, scale):
    """LR statistic for INST, normalized per timepoint.

    Sums (log residual)^2 / scale^2 over EMUs x timepoints, then divides by the
    number of timepoints. INST measures every EMU at N timepoints whereas SS
    measures each once, so dividing by N puts the INST statistic on a
    per-measurement footing comparable to SS (normalizes for the extra
    measurements taken over the timecourse). This is equivalent to scaling the
    chi-squared cutoff up by N for the raw summed statistic.
    """
    total = 0.0
    n_tp = 0
    for emu in ref_tc:
        if emu not in test_tc:
            continue
        tps = [tp for tp in ref_tc[emu] if tp in test_tc[emu]]
        n_tp = max(n_tp, len(tps))
        for tp in tps:
            total += _chisq_pair(ref_tc[emu][tp], test_tc[emu][tp], scale)
    return total / n_tp if n_tp > 0 else total


def chisq_ss(ref_ss, test_ss, scale):
    """LR statistic for SS: sum over EMUs."""
    total = 0.0
    for emu in ref_ss:
        if emu not in test_ss:
            continue
        total += _chisq_pair(ref_ss[emu], test_ss[emu], scale)
    return total


# ============================================================================
# Per-equilibrium state: reference MDVs + simulation cache
# ============================================================================

_STATE = {}  # {"thermo_file", "ref_tc", "ref_ss", "cache"}


def lr_statistic(fba1, fba2):
    """Return (chi2_timecourse, chi2_ss) at (FBA1, FBA2). Memoized per case.

    One run_simulation call yields BOTH INST and SS MDVs, so the cache is shared
    between the inst and ss cases of the same equilibrium mode.
    """
    key = (round(float(fba1), 6), round(float(fba2), 6))
    cache = _STATE["cache"]
    if key in cache:
        return cache[key]
    try:
        mdvs_tc, mdvs_ss = run_simulation(
            {"FBA1": float(fba1), "FBA2": float(fba2)}, TIMEPOINTS, _STATE["thermo_file"]
        )
        chi2_tc = chisq_timecourse(_STATE["ref_tc"], mdvs_tc, LOGNORMAL_SCALE)
        chi2_ss_val = chisq_ss(_STATE["ref_ss"], mdvs_ss, LOGNORMAL_SCALE)
    except Exception as e:  # freeflux blow-up at extreme dG -> looks "outside"
        print(
            f"      sim failed at FBA1={fba1:.3f} FBA2={fba2:.3f}: {type(e).__name__}: {e}",
            flush=True,
        )
        chi2_tc = chi2_ss_val = 1e6
    cache[key] = (chi2_tc, chi2_ss_val)
    return chi2_tc, chi2_ss_val


# ============================================================================
# Ray geometry
# ============================================================================

def ray_point(theta, d, true1, true2):
    return true1 + d * np.cos(theta), true2 + d * np.sin(theta)


def ray_d_cap(theta, true1, true2):
    """Largest d before either coordinate leaves [DG_LO, DG_HI]."""
    caps = []
    for base, comp in ((true1, np.cos(theta)), (true2, np.sin(theta))):
        if comp > 1e-9:
            caps.append((DG_HI - base) / comp)
        elif comp < -1e-9:
            caps.append((DG_LO - base) / comp)
    return min(caps) if caps else np.inf


def lr_along_ray(theta, d, true1, true2, mode):
    f1, f2 = ray_point(theta, d, true1, true2)
    tc, ss = lr_statistic(f1, f2)
    return ss if mode == "ss" else tc


def find_upper_bracket(theta, true1, true2, mode, target, d_cap):
    """Geometrically expand d until LR(d) > target or we hit d_cap.

    Returns (d_upper, bounded). bounded=False -> no crossing within the ray cap
    (flat / unidentifiable direction).
    """
    d = min(D_START, 0.9 * d_cap)
    for _ in range(MAX_BRACKET_DOUBLINGS):
        if d > d_cap:
            d = d_cap
        lr = lr_along_ray(theta, d, true1, true2, mode)
        if lr > target:
            return d, True
        if d >= d_cap:
            return d_cap, False
        d *= 2.0
    return d, False


# ============================================================================
# optimistix root-finding (pure_callback wraps the freeflux objective)
# ============================================================================

def make_objective(theta, true1, true2, mode):
    """Build optimistix objective f(d, threshold) = LR(d) - threshold.

    LR is a freeflux black box, so it runs on the host via jax.pure_callback.
    """

    def host_lr(d_np):
        return np.float64(lr_along_ray(theta, float(d_np), true1, true2, mode))

    def objective(d, threshold):
        lr = jax.pure_callback(host_lr, jax.ShapeDtypeStruct((), jnp.float64), d)
        return lr - threshold

    return objective


def solve_crossing(objective, threshold, d_upper):
    """Find d in [0, d_upper] where LR(d) == threshold.

    Returns (d, accepted, converged). Bisection always brackets the root, so a
    finite in-bracket value is a valid crossing to ~float precision even if
    optimistix flags max_steps; `accepted` reflects that, `converged` reports
    optimistix's own criterion for transparency.

    Requires a sign change on the bracket: LR(0)-thr = -thr < 0 and
    LR(d_upper)-thr > 0 (guaranteed by find_upper_bracket / descending reuse).
    """
    solver = optx.Bisection(rtol=BISECTION_RTOL, atol=BISECTION_ATOL)
    try:
        sol = optx.root_find(
            objective,
            solver,
            y0=jnp.asarray(0.5 * d_upper, dtype=jnp.float64),
            args=jnp.asarray(float(threshold), dtype=jnp.float64),
            options={"lower": 0.0, "upper": float(d_upper)},
            max_steps=60,
            throw=False,
        )
    except Exception as e:  # e.g. no sign change in bracket
        print(f"      root_find failed: {type(e).__name__}: {e}", flush=True)
        return np.nan, False, False
    val = float(sol.value)
    converged = bool(sol.result == optx.RESULTS.successful)
    accepted = converged or (np.isfinite(val) and 0.0 < val <= d_upper * (1 + 1e-9))
    return val, accepted, converged


# ============================================================================
# Per-case solve
# ============================================================================

def _row(eq, mode, ai, theta, alpha, true1, true2, d, ok, unbounded):
    if np.isfinite(d):
        f1, f2 = ray_point(theta, d, true1, true2)
    else:
        f1 = f2 = np.nan
    return {
        "equilibrium": eq,
        "sim_mode": mode,
        "case": f"{eq}_{mode}",
        "angle_idx": ai,
        "theta_rad": float(theta),
        "theta_deg": float(np.degrees(theta)),
        "alpha": alpha,
        "confidence": 1.0 - alpha,
        "threshold": THRESHOLDS[alpha],
        "true_fba1": true1,
        "true_fba2": true2,
        "d_crossing": float(d) if np.isfinite(d) else np.nan,
        "fba1_crossing": f1,
        "fba2_crossing": f2,
        "converged": bool(ok),
        "unbounded": bool(unbounded),
    }


def solve_case(eq, mode, true1, true2, angles, alphas):
    """Trace confidence crossings for one (equilibrium, sim_mode) case."""
    print(f"\n=== Solving {eq}_{mode} ===", flush=True)
    # Solve the largest threshold first, then reuse a shrinking bracket. The
    # largest threshold is the smallest alpha (= largest confidence region).
    alphas_by_threshold_desc = sorted(alphas, key=lambda a: THRESHOLDS[a], reverse=True)
    largest = alphas_by_threshold_desc[0]
    rows = []

    for ai, theta in enumerate(angles):
        deg = np.degrees(theta)
        d_cap = ray_d_cap(theta, true1, true2)
        d_up, bounded = find_upper_bracket(
            theta, true1, true2, mode, THRESHOLDS[largest], d_cap
        )

        if not bounded:
            print(
                f"  angle {ai} ({deg:5.1f} deg): UNBOUNDED "
                f"(max LR < {THRESHOLDS[largest]:.2f} within d_cap={d_cap:.2f})",
                flush=True,
            )
            for alpha in alphas_by_threshold_desc:
                rows.append(_row(eq, mode, ai, theta, alpha, true1, true2, np.nan, False, True))
            continue

        objective = make_objective(theta, true1, true2, mode)
        d_prev = d_up
        hits = {}
        for alpha in alphas_by_threshold_desc:
            d_hit, accepted, converged = solve_crossing(objective, THRESHOLDS[alpha], d_prev)
            d_rec = d_hit if (accepted and np.isfinite(d_hit)) else np.nan
            if np.isfinite(d_rec):
                d_prev = d_rec  # shrink bracket for the next (smaller) threshold
                hits[alpha] = d_rec
            rows.append(_row(eq, mode, ai, theta, alpha, true1, true2, d_rec, converged, False))
        summary = ", ".join(
            f"conf{round((1 - a) * 100)}%={hits[a]:.3f}" if a in hits
            else f"conf{round((1 - a) * 100)}%=FAIL"
            for a in alphas_by_threshold_desc
        )
        print(f"  angle {ai} ({deg:5.1f} deg): d_cap={d_cap:5.2f} | {summary}", flush=True)

    return rows


def run_equilibrium(eq, angles, modes, alphas):
    """Build the reference + cache for one equilibrium mode, solve all sim modes."""
    cfg = EQUILIBRIUM_CASES[eq]
    true1, true2 = cfg["true1"], cfg["true2"]
    _STATE["thermo_file"] = cfg["thermo_file"]
    _STATE["cache"] = {}

    print(f"\n########## {cfg['label']} ({eq}) ##########", flush=True)
    print(f"True point: FBA1={true1}, FBA2={true2}", flush=True)
    ref_tc, ref_ss = run_simulation({"FBA1": true1, "FBA2": true2}, TIMEPOINTS, cfg["thermo_file"])
    _STATE["ref_tc"], _STATE["ref_ss"] = ref_tc, ref_ss

    tc0, ss0 = lr_statistic(true1, true2)  # sanity: should be ~0
    print(f"True-point LR: timecourse={tc0:.3e}, ss={ss0:.3e}", flush=True)

    rows = []
    for mode in modes:
        rows += solve_case(eq, mode, true1, true2, angles, alphas)
    return rows


# ============================================================================
# Plotting
# ============================================================================

def plot_contours(df, eqs, modes, suffix=""):
    apply_style()
    # Colors keyed by alpha. Largest threshold (alpha 0.5 = 50% conf) is the
    # outermost contour; alpha 0.9 (10% conf) is innermost.
    alpha_colors = {0.9: cmc.batlow(0.20), 0.75: cmc.batlow(0.50), 0.5: cmc.batlow(0.80)}

    nrows, ncols = len(eqs), len(modes)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(DOUBLE_COL * ncols / 2, DOUBLE_COL * nrows / 2), squeeze=False
    )
    # Drop per-vertex markers when the contour is densely sampled.
    marker = "o" if df.angle_idx.nunique() <= 20 else ""

    for i, eq in enumerate(eqs):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            sub = df[(df.equilibrium == eq) & (df.sim_mode == mode)]
            if sub.empty:
                ax.set_visible(False)
                continue
            true1 = sub.true_fba1.iloc[0]
            true2 = sub.true_fba2.iloc[0]

            # Draw the largest threshold (outermost) first so it sits behind.
            for alpha in sorted(alpha_colors, key=lambda a: THRESHOLDS[a], reverse=True):
                cs = sub[sub.alpha == alpha].sort_values("theta_deg")
                xs = cs.fba2_crossing.to_numpy()
                ys = cs.fba1_crossing.to_numpy()
                finite = np.isfinite(xs) & np.isfinite(ys)
                if finite.sum() == 0:
                    continue
                xf, yf = xs[finite], ys[finite]
                if finite.all():  # close the polygon only if every ray is bounded
                    xf = np.append(xf, xf[0])
                    yf = np.append(yf, yf[0])
                ax.plot(
                    xf, yf, ls="-", marker=marker, color=alpha_colors[alpha], ms=3, lw=1.0,
                    label=f"{round((1 - alpha) * 100)}% (α={alpha})",
                )

            ax.plot(true2, true1, marker="*", color="red", ms=11, zorder=5,
                    linestyle="none", label="true point")

            if bool(sub.unbounded.any()):
                ax.text(0.03, 0.97, "open (unbounded) direction(s)", transform=ax.transAxes,
                        ha="left", va="top", fontsize=6, color="0.3")

            ax.set_xlabel("FBA2 ΔG (kJ/mol)")
            ax.set_ylabel("FBA1 ΔG (kJ/mol)")
            ax.invert_xaxis()  # match landscape orientation (more negative = upper-right)
            ax.invert_yaxis()
            ax.set_title(f"{EQUILIBRIUM_CASES[eq]['label']} — {mode.upper()}")
            if i == 0 and j == 0:
                ax.legend(loc="best", fontsize=6)

    fig.suptitle("Likelihood-ratio confidence contours (FBA1 vs FBA2 ΔG)", y=1.01)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = RESULTS_DIR / f"confidence_contours{suffix}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}", flush=True)
    plt.close(fig)


def plot_overlay(df, suffix=""):
    """Overlay INST and SS contours in one panel per equilibrium mode.

    Color encodes confidence (batlow); line style encodes the experiment
    (INST solid/o, SS dashed/s). INST is drawn last so it stays visible — its
    region is much tighter than SS, so the overlay shows the scale difference
    directly.
    """
    from matplotlib.lines import Line2D

    apply_style()
    alpha_colors = {0.9: cmc.batlow(0.20), 0.75: cmc.batlow(0.50), 0.5: cmc.batlow(0.80)}
    # Drop per-vertex markers when densely sampled (line alone traces the contour).
    dense = df.angle_idx.nunique() > 20
    mode_style = {
        "ss": dict(ls="--", marker=("" if dense else "s")),    # drawn first (larger region)
        "inst": dict(ls="-", marker=("" if dense else "o")),   # drawn last (on top)
    }

    eqs = [e for e in EQUILIBRIUM_CASES if (df.equilibrium == e).any()]
    fig, axes = plt.subplots(
        1, len(eqs), figsize=(DOUBLE_COL, DOUBLE_COL * 0.5), squeeze=False
    )

    for k, eq in enumerate(eqs):
        ax = axes[0][k]
        sub_eq = df[df.equilibrium == eq]
        true1 = sub_eq.true_fba1.iloc[0]
        true2 = sub_eq.true_fba2.iloc[0]

        for mode in ("ss", "inst"):
            sub = sub_eq[sub_eq.sim_mode == mode]
            if sub.empty:
                continue
            for alpha in sorted(alpha_colors, key=lambda a: THRESHOLDS[a], reverse=True):
                cs = sub[sub.alpha == alpha].sort_values("theta_deg")
                xs = cs.fba2_crossing.to_numpy()
                ys = cs.fba1_crossing.to_numpy()
                finite = np.isfinite(xs) & np.isfinite(ys)
                if finite.sum() == 0:
                    continue
                xf, yf = xs[finite], ys[finite]
                if finite.all():  # close polygon only if every ray is bounded
                    xf = np.append(xf, xf[0])
                    yf = np.append(yf, yf[0])
                ax.plot(
                    xf, yf, color=alpha_colors[alpha], ms=2.5, lw=1.0,
                    zorder=4 if mode == "inst" else 3, **mode_style[mode],
                )

        ax.plot(true2, true1, marker="*", color="red", ms=11, zorder=6,
                linestyle="none")
        if bool(sub_eq.unbounded.any()):
            ax.text(0.03, 0.03, "open (unbounded) direction(s)", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=6, color="0.3")
        ax.set_xlabel("FBA2 ΔG (kJ/mol)")
        ax.set_ylabel("FBA1 ΔG (kJ/mol)")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_title(EQUILIBRIUM_CASES[eq]["label"])

    # Manual legend: color = confidence, line style = experiment.
    conf_handles = [
        Line2D([], [], color=alpha_colors[a], lw=2, label=f"{round((1 - a) * 100)}% conf")
        for a in sorted(alpha_colors, key=lambda a: THRESHOLDS[a], reverse=True)
    ]
    mode_handles = [
        Line2D([], [], color="0.3", ls="-", marker="o", ms=3, label="INST"),
        Line2D([], [], color="0.3", ls="--", marker="s", ms=3, label="SS"),
    ]
    star = Line2D([], [], color="red", marker="*", ls="none", ms=8, label="true point")
    axes[0][0].legend(handles=conf_handles + mode_handles + [star], fontsize=6, loc="best")

    fig.suptitle("INST vs SS confidence regions overlaid (FBA1 vs FBA2 ΔG)", y=1.02)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = RESULTS_DIR / f"confidence_contours_overlay{suffix}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}", flush=True)
    plt.close(fig)


# ============================================================================
# Verification helpers
# ============================================================================

def check_nesting(df):
    """Per (case, angle): d should increase with the chi-squared threshold."""
    print("\n--- Nesting check (d should increase with threshold) ---", flush=True)
    violations = 0
    for (case, ai), g in df.groupby(["case", "angle_idx"]):
        ds = g.sort_values("threshold")["d_crossing"]
        finite = ds.dropna()
        if len(finite) >= 2 and not np.all(np.diff(finite.values) >= -1e-6):
            violations += 1
            print(f"  WARNING non-monotonic in {case} angle {ai}: {finite.tolist()}", flush=True)
    print(f"  nesting violations: {violations}", flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Fast sanity run: 4 angles, ffe only, ss only, alpha=[0.5]")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation; regenerate plots from the saved CSV.")
    parser.add_argument("--eq", choices=list(EQUILIBRIUM_CASES) + ["all"], default="all",
                        help="Which equilibrium case(s) to compute (default: all).")
    parser.add_argument("--angles", type=int, default=N_ANGLES,
                        help=f"Number of ray angles around the circle (default: {N_ANGLES}).")
    args = parser.parse_args()

    if args.plot_only:
        csv_path = RESULTS_DIR / "confidence_contours.csv"
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path} ({len(df)} rows)", flush=True)
        eqs = [e for e in EQUILIBRIUM_CASES if (df.equilibrium == e).any()]
        modes = [m for m in SIM_MODES if (df.sim_mode == m).any()]
        check_nesting(df)
        plot_contours(df, eqs, modes)
        plot_overlay(df)
        return

    if args.smoke:
        n_angles = 4
        eqs = ["ffe"]
        modes = ["ss"]
        alphas = [0.5]
    else:
        n_angles = args.angles
        eqs = list(EQUILIBRIUM_CASES) if args.eq == "all" else [args.eq]
        modes = SIM_MODES
        alphas = ALPHA_LEVELS

    angles = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)

    print("Chi-squared thresholds chi2.ppf(1 - alpha, df=2):", flush=True)
    for a in sorted(alphas, key=lambda a: THRESHOLDS[a], reverse=True):
        print(f"  alpha={a} -> {round((1 - a) * 100)}% conf -> threshold {THRESHOLDS[a]:.4f}", flush=True)

    t0 = time.time()
    rows = []
    for eq in eqs:
        rows += run_equilibrium(eq, angles, modes, alphas)
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    csv_path = RESULTS_DIR / f"confidence_contours{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} ({len(df)} rows)", flush=True)

    check_nesting(df)
    plot_contours(df, eqs, modes, suffix=suffix)
    plot_overlay(df, suffix=suffix)

    print(f"\nDone in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Notes
Don't co-author with Claude

## Project Overview

This repository contains ¹³C metabolic flux analysis (¹³C-MFA) simulations using the Elementary Metabolite Unit (EMU) algorithm. It models isotope labeling patterns in metabolic networks to estimate intracellular flux distributions.

## Repository Layout

Scripts are split into the seven sets behind the paper (`paper_scripts/`) and everything
else (`other_scripts/`). Outputs follow the same split: `results/paper_scripts/<set>/` and
`results/other_scripts/`. Figures a script writes go to a `figures/` subfolder next to its
results. The top-level `figures/` holds only the hand-assembled (Inkscape) paper figures.
See `paper_scripts/README.md` for the full map.

| `paper_scripts/` set | Scripts | Results (`results/paper_scripts/...`) |
|---|---|---|
| `common/` | `style.py`, `emu_jax.py`, `enzyme_utils.py` (shared; also used by `other_scripts/` and `tests/`) | — |
| `forward_simulations/` | `gibbs_fba_split_comparison.py` (FFE), `gibbs_fba_split_comparison_ne.py` (NE), `plot_ffe_ne_comparison.py` | `forward_simulations/` |
| `confidence_contours/` | `gibbs_confidence_contours.py` | `confidence_contours/` |
| `mcmc_7_params/` | `enzyme_simulation{,_instationary,_no_labeling}.py` (7 identifiable rate parameters), `plot_parameter_comparison_identifiable.py` | `mcmc_7_params/` |
| `gibbs_landscape/` | `gibbs_landscape_2d.py`, `plot_landscape_combined.py` | `gibbs_landscape/` |
| `mcmc_11_params/` | `enzyme_simulation{,_instationary,_no_labeling}.py` (all 11 rate parameters), `plot_parameter_comparison.py` | `mcmc_11_params/` |
| `parameter_identifiability/` | `fisher_information.py` | `parameter_identifiability/` |
| `standard_ne_fit/` | `gibbs_high_oxppp_fit_ne.py`, `gibbs_high_oxppp_comparison_ne.py` (supplies the fit's setup) | `standard_ne_fit/` |

Scripts import across folders by inserting the folder into `sys.path`
(e.g. `gibbs_confidence_contours.py` adds `paper_scripts/gibbs_landscape`, and every set
adds `paper_scripts/common`). Paper scripts sit two levels below the repo root
(`PROJECT_DIR = SCRIPT_DIR.parent.parent`); `other_scripts/` is flat, one level below.

## Commands

### Install dependencies
```bash
uv sync
```

### Run Gibbs energy analysis (most commonly used)
```bash
uv run python paper_scripts/gibbs_landscape/gibbs_landscape_2d.py          # 2D FBA1/FBA2 landscape
uv run python other_scripts/gibbs_comparison_1d.py         # Mechanistic vs non-mechanistic
uv run python other_scripts/gibbs_comparison_fba1_fixed.py # FBA1 fixed, FBA2 varies
uv run python other_scripts/gibbs_comparison_fba2_fixed.py # FBA2 fixed, FBA1 varies
```

### Run enzyme simulation
```bash
uv run python paper_scripts/mcmc_11_params/enzyme_simulation.py             # Steady-state with isotope labeling
uv run python paper_scripts/mcmc_11_params/enzyme_simulation_instationary.py # Instationary isotope labeling
uv run python paper_scripts/mcmc_11_params/enzyme_simulation_no_labeling.py  # No isotope labeling
```

### Run main script
```bash
uv run python main.py
```

## Architecture

### Dependencies
- **freeflux**: Core library for ¹³C-MFA flux fitting using the EMU algorithm
- **diffrax**: JAX-based ODE solver for enzyme kinetics simulation
- **blackjax**: MCMC sampling (NUTS) for parameter estimation
- **jax**: JAX for numerical computation (pinned to <=0.4.18)
- Python 3.10+

### Model Definition

Models are defined via:
1. **TSV reaction files** (like `reactions.tsv`) - uses freeflux format with atom mappings:
   ```
   #reaction_ID	reactant_IDs(atom)	product_IDs(atom)	reversibility
   v1	OAA(abcd)+AcCoA(ef)	Cit(dcbfea)	0
   ```
2. **Python API** - configure labeling, flux bounds, measurements, and fitting via freeflux `Model` and fitter context managers

### models/ Directory

Contains TOML-based model configurations for glycolysis pathway models:

- **glycolysis_gly3p/**: Non-mechanistic glycolysis model with single FBA reaction
- **glycolysis_mech_gly3p/**: Mechanistic glycolysis model with decomposed FBA reaction (FBA1 + FBA2)

Each model directory contains:
- `reactions.tsv`: Reaction network with atom mappings (freeflux format)
- `glycolysis_fluxes.toml`: Net flux values for each reaction
- `glycolysis_thermodynamics.toml`: Gibbs energy (deltaG) values for reversibility calculations
- `glycolysis_network.toml`: Network topology definition

#### Mechanistic FBA Decomposition

The mechanistic model decomposes the FBA (fructose-bisphosphate aldolase) reaction into two elementary steps:
```
FBA1: fdp(abcdef) → g3p(def) + FBA_dhap(cba)   # g3p released, dhap bound to enzyme
FBA2: FBA_dhap(abc) → dhap(abc)                 # dhap released from enzyme
```

This allows separate Gibbs energy values for each step (default: FBA1=-3.0, FBA2=-2.0 kJ/mol).

#### Transketolase / Transaldolase Decomposition

**Both** glycolysis models write the non-oxidative PPP as elementary half reactions
instead of the lumped `TKTL1`/`TKTL2`/`TALA` overall reactions. Carbon passes through
two enzyme-bound carriers: `EC2` (2 carbons, on transketolase) and `EC3` (3 carbons,
on transaldolase):

```
X5P_EC2: xu5p__D(abcde)  → EC2(ab) + g3p(cde)
F6P_EC2: f6p(abcdef)     → EC2(ab) + e4p(cdef)
S7P_EC2: s7p(abcdefg)    → EC2(ab) + r5p(cdefg)
F6P_EC3: f6p(abcdef)     → EC3(abc) + g3p(def)
S7P_EC3: s7p(abcdefg)    → EC3(abc) + e4p(defg)
```

Each overall reaction is one half reaction forward plus another in reverse:

| Overall reaction | Decomposition |
|---|---|
| TKT1: xu5p + r5p ↔ s7p + g3p | `X5P_EC2` forward + `S7P_EC2` reverse |
| TKT2: xu5p + e4p ↔ f6p + g3p | `X5P_EC2` forward + `F6P_EC2` reverse |
| TALA: s7p + g3p ↔ e4p + f6p | `S7P_EC3` forward + `F6P_EC3` reverse |

Balancing the `EC2` and `EC3` pools fixes the net fluxes. With each overall reaction
carrying 5.4843 mmol/gDCW/h:

```
X5P_EC2 =  TKT1 + TKT2 =  10.9686      S7P_EC3 =  TALA =  5.4843
S7P_EC2 = -TKT1        =  -5.4843      F6P_EC3 = -TALA = -5.4843
F6P_EC2 = -TKT2        =  -5.4843
```

**Sign convention (important).** `F6P_EC2`, `S7P_EC2` and `F6P_EC3` run *backwards*
relative to how they are written in `reactions.tsv`, so they carry a **negative net
flux** and, for thermodynamic consistency (`ΔG = −RT·ln(v_f/v_b)`), a **positive
ΔG**. `net_to_forward_backward()` handles this and still returns positive `v_f` and
`v_b` (with `v_b > v_f`). Two consequences:

- ΔG values *subtract* rather than add: `ΔG_TKT1 = ΔG(X5P_EC2) − ΔG(S7P_EC2)`.
  The lumped ΔG is split evenly in magnitude, so each half reaction gets ±0.5 kJ/mol
  (far-from-equilibrium) or ±0.25 kJ/mol (near-equilibrium).
- `freeflux` applies `set_flux_bounds` to **net** fluxes, so any fitting code must
  allow negative bounds for these three reactions. See `NEGATIVE_NET_RXNS` in
  `other_scripts/fit_mechanistic_vs_aggregated.py`.

`EC2`/`EC3` are enzyme-bound, so scripts that give `FBA_dhap` a small pool give them
the same (see `ENZYME_BOUND_METABOLITES`).

#### Flux Balance

The flux distributions in both `glycolysis_fluxes.toml` files close **exactly** for
every internal metabolite. The independent fluxes are `glc_uptake` (100.0043),
`G6PDH2r_PGL` (17.4529), `r5p_drain` (1.0), `GLY3PDH` (5.0) and `LDH` (100.0); the
rest follow from stoichiometry. `INITIAL_FLUXES_MMOL`
(`other_scripts/fit_mechanistic_vs_aggregated.py`) and `TRUE_NET_FLUXES_MMOL`
(`other_scripts/plot_flux_error_heatmaps.py`) duplicate these values and must stay in sync.

#### High-oxPPP Flux Map (PFK/oxPPP split 50/50)

Both model directories also contain `glycolysis_fluxes_high_oxppp.toml`, which also
closes exactly. Its independent fluxes are `glc_uptake` (100), `r5p_drain` (1),
`GLY3PDH` (5) and `LDH` (100), plus the constraint `PFK` = `G6PDH2r_PGL`. The oxPPP
returns its carbon to f6p/g3p through the non-oxidative PPP, so
PFK = GPI + ⅔(G6PDH − r5p_drain). With equal fluxes this gives
PFK = G6PDH2r_PGL = 74.5, GPI = 25.5 and TKT1 = TKT2 = TALA = 24.5. Every reaction keeps
its baseline direction.

`paper_scripts/standard_ne_fit/gibbs_high_oxppp_comparison_ne.py` (`build_fluxes`) flips any ΔG whose sign
agrees with its net flux (ΔG = −sign(v_net)·|ΔG|) and prints the flipped reactions. This
is a safeguard for flux maps that reverse a reaction; none are flipped for this map.

## Gibbs Energy Analysis Scripts

Scripts for analyzing how Gibbs energy parameters affect isotope labeling patterns using INST (isotopically non-stationary) simulations.

### Configuration (shared across scripts)
- **Timepoints**: 100 points logarithmically spaced from 1e-3s to 1e1s, plus final at 1e3s
- **Target EMUs**: g6p, f6p, dhap, g3p (full carbon fragments)
- **Noise level**: 5% standard deviation for chi-squared calculations
- **Labeling**: Glucose input with "110000" pattern (carbons 1-2 labeled)

### Scripts

| Script | Description | Results Directory |
|--------|-------------|-------------------|
| `gibbs_landscape_2d.py` | 2D scan of FBA1 vs FBA2 Gibbs energies (30x30 grid) | `results/paper_scripts/gibbs_landscape/gibbs_landscape_2d_{ffe,ne}/` |
| `gibbs_confidence_contours.py` | Likelihood-ratio confidence contours via rotating-ray + optimistix root-finding (parallelized) | `results/paper_scripts/confidence_contours/gibbs_confidence_contours/` |
| `gibbs_timecourse_chisquared.py` | Chi-squared evolution over time for different FBA1/FBA2 combinations | `results/other_scripts/gibbs_timecourse/` |
| `gibbs_high_oxppp_comparison_ne.py` | Mechanistic (20:80 FBA1:FBA2 dG split) vs aggregated MID differences on the high-oxPPP flux map, NE thermodynamics, INST + SS. It uses **1,2-¹³C glucose** (`"110000"`, `LABELING_PATTERN`, also used by `gibbs_high_oxppp_fit_ne.py`) and, unlike the other scripts, also measures **fdp** (`fdp_123456`) | `results/paper_scripts/standard_ne_fit/gibbs_high_oxppp_ne/` |
| `gibbs_high_oxppp_fit_ne.py` | freeflux SS + INST fits of the mechanistic MIDs on the standard flux map with the mechanistic (self-fit) and aggregated models, for 5 FBA1:FBA2 ΔG splits; χ², flux and ΔG recovery | `results/paper_scripts/standard_ne_fit/gibbs_standard_fit_ne/` |
| `gibbs_comparison_1d.py` | Compare mechanistic vs non-mechanistic models (batlow colors, separate final plot) | `results/other_scripts/gibbs_comparison_1d/` |
| `gibbs_comparison_fba1_fixed.py` | 1D scan: FBA1 fixed at -3.0, FBA2 varies -5 to -1 (with timecourse) | `results/other_scripts/gibbs_comparison_fba1_fixed/` |
| `gibbs_comparison_fba2_fixed.py` | 1D scan: FBA2 fixed at -2.0, FBA1 varies -5 to -1 (with timecourse) | `results/other_scripts/gibbs_comparison_fba2_fixed/` |

### Output Files

Paths below are relative to `results/`.

**other_scripts/gibbs_comparison_1d/** (mechanistic vs non-mechanistic):
- `gibbs_comparison_1d.csv` - SSR vs non-mechanistic FBA dG
- `gibbs_comparison_1d.png` - Timecourse SSR plot (batlow colormap)
- `gibbs_comparison_1d_final.png` - Final timepoint (1000s) SSR plot

**other_scripts/gibbs_comparison_fba1_fixed/** (FBA1=-3.0 fixed):
- `gibbs_comparison_fba1_fixed.csv` - SSR vs FBA2 parameter
- `gibbs_comparison_fba1_fixed.png` - 1D SSR plot
- `timecourse_chisquared.csv` - Chi-squared at each timepoint per condition
- `timecourse_aggregated.png` - Total chi-squared vs time for each FBA2 value
- `timecourse_per_metabolite.png` - Per-metabolite chi-squared subplots

**other_scripts/gibbs_comparison_fba2_fixed/** (FBA2=-2.0 fixed):
- `gibbs_comparison_fba2_fixed.csv` - SSR vs FBA1 parameter
- `gibbs_comparison_fba2_fixed.png` - 1D SSR plot
- `timecourse_chisquared.csv` - Chi-squared at each timepoint per condition
- `timecourse_aggregated.png` - Total chi-squared vs time for each FBA1 value
- `timecourse_per_metabolite.png` - Per-metabolite chi-squared subplots

**paper_scripts/confidence_contours/gibbs_confidence_contours/** (likelihood-ratio confidence regions):
- `confidence_contours.csv` - Per (equilibrium, sim_mode, angle, alpha) crossing distance `d` and the FBA1/FBA2 ΔG coordinates where the LR statistic crosses the chi-squared threshold (plus `converged`/`unbounded` flags)
- `confidence_contours.png/pdf` - Per-case grid of contours (one panel per equilibrium × sim_mode)
- `confidence_contours_overlay.png/pdf` - Single square near-equilibrium panel overlaying INST (blue) and SS (salmon) regions; confidence intervals encoded by saturation, reference point as a black cross

**paper_scripts/standard_ne_fit/gibbs_high_oxppp_ne/** (high-oxPPP flux map, mechanistic vs aggregated):
- `mid_differences_ss.csv` - Per metabolite × isotopologue: `mid_agg`, `mid_mech`, `diff` (= mech − agg)
- `mid_differences_inst.csv` - Same, per timepoint
- `mech_vs_agg_differences.png/pdf` - SS signed ΔMID bars per isotopologue (left); INST max |ΔMID| vs time (right)
- `inst_isotopologue_differences.png/pdf` - Per-metabolite signed ΔMID of each isotopologue vs time

**paper_scripts/standard_ne_fit/gibbs_standard_fit_ne/** (freeflux fits of the mechanistic MIDs on the standard flux map, from `gibbs_high_oxppp_fit_ne.py`):
- `heatmap_flux_errors.png/pdf` - Net flux % error across the splits: one panel per (mode, model), rows are reactions, columns FBA1 fractions. Clipped to ±15% (`FLUX_CLIP_PCT`; net fluxes are recovered to within ~10%), clipped cells labelled with their value, hatched cells are reactions the model lacks
- `heatmap_gibbs_errors.png/pdf` - Same for the ΔG % error of the near-equilibrium reactions, clipped to ±200% (`ERROR_CLIP_PCT`)
- `ssr_flux_errors.png/pdf` - Best SSR (left) and largest absolute net flux % error (right) of each fit vs the FBA1 fraction, one line per model; one row per mode, titled "Steady state" and "Instationary". The SSR panels show each model's 95% χ² upper bound dashed. Every panel has its own y-axis
- `ssr.csv` - The `fit_summary.csv` rows of every split, with an `fba1_fraction` column
- `max_flux_errors.csv` - Per (split, sim_mode, model): the reaction with the largest error, its signed % error and the absolute value
- `mean_gibbs_errors.png/pdf` - Mean absolute ΔG % error of each fit vs the FBA1 fraction, over the near-equilibrium reactions of the ΔG heatmap. Each error is clipped to ±200% (`ERROR_CLIP_PCT`) before averaging, since one ∞ or thousands-% error would otherwise dominate the mean
- `mean_gibbs_errors.csv` - Per (split, sim_mode, model): number of reactions averaged, number clipped, and the mean
- `fba1_<fraction>/` - One directory per split (`fba1_0.1` … `fba1_0.9`) with:
  - `fit_summary.csv` - Per (sim_mode, model): SSR, dof, 95% χ² range, `accepted` (SSR ≤ upper bound), best start
  - `fitted_fluxes.csv` - Per reaction: true/fitted net flux (mmol/gDCW/h), % error, true/fitted ΔG, `identifiable`
  - `mdv_residuals.csv` - Per measured isotopologue: measured, fitted, (fitted − measured)/sd
  - `start_objectives.csv` - SSR of every (randomised) start
  - `fit_flux_errors.png/pdf` - Net flux % error per reaction (SS | INST, aggregated vs mechanistic)
  - `fit_gibbs_errors.png/pdf` - ΔG % error, (fit − true)/true × 100, of the near-equilibrium reactions only (|ΔG_true| < 2 kJ/mol; far from equilibrium the backward flux barely affects labeling, so ΔG is not identifiable). A positive error puts the reaction further from equilibrium. Bars beyond ±200% are clipped and labelled with their value
  - `fit_mdv_residuals.png/pdf` - Normalized MDV residuals grouped by EMU

Fitted ΔG is ±∞ where a fitted forward or backward flux is 0 (slightly negative
fitted fluxes are treated as 0), so its % error is ±∞ and shows as a clipped cell/bar.

### Standard-Flux Fits (`gibbs_high_oxppp_fit_ne.py`)

Fits noise-free mechanistic MIDs, simulated on the standard flux map
(`glycolysis_fluxes.toml`, `FLUX_FILE`), with freeflux's SS and INST fitters, using both
the mechanistic model (self-fit) and the aggregated model. Despite its name, the script no
longer uses the high-oxPPP map; it imports the rest of its setup (`build_fluxes`,
`run_simulation`, `pool_size`, tracer, EMUs) from `gibbs_high_oxppp_comparison_ne.py`.

- **FBA ΔG splits**: the analysis runs once per FBA1 fraction in `FBA1_FRACTIONS`
  (`linspace(0.1, 0.9, 5)`, `--splits`): FBA1 = fraction × ΔG_FBA and FBA2 = the rest,
  with ΔG_FBA = −0.87 kJ/mol. Only the mechanistic data and self-fit change with the split;
  the aggregated model's truth (FBA = −0.87) is the same for every split.
- **Measurements**: MDV sd = 0.002 (0.2 mol%). INST uses 5 timepoints, `logspace(-3, 1, 5)`.
  Measured fluxes are `glc_input` (sd 1%), `r5p_drain` and `GLY3PDH`, the dhap drain
  (sd 5%). INST pool sizes are pinned at their simulated values.
- **Seeded start (important)**: freeflux only accepts initial *net* fluxes and fills in
  exchange fluxes by a minimum-norm projection. For near-equilibrium INST this fails:
  starting even from the true net fluxes, the mechanistic self-fit stalls at SSR ≈ 10⁴.
  Every start therefore uses `seeded_start(u)` with `u = pinv(N)·v_total`, which
  temporarily wraps freeflux's `build_initial_*` methods. No start sits at the reference
  point, since that would trivially be the best solution. Every start uses
  `perturb_fluxes`, which scales every true net and exchange flux (min(v_f, v_b)) by
  U(0.5, 1.5) and projects the net fluxes back onto the steady state.
- **Starts and parallelism**: 500 starts per (split, mode, model) fit (`--starts`). They are
  split into worker tasks (`STARTS_PER_TASK`: 2 INST or 10 SS starts each) that run in one
  spawn `ProcessPoolExecutor` shared by all splits; each split is saved as soon as its last
  task finishes (`run_fits`). The default `--workers` is all CPUs but two, with one BLAS thread
  per worker. Each start seeds its own RNG with `[RNG_SEED, start]`, so results do not
  depend on the chunking. `merge_chunks` keeps the best start and every start's SSR.
- **χ²**: freeflux's `chi2_test()` only prints, so the script computes the range from
  `result.dof`. Noise-free data gives an SSR below the lower bound for a correct model.
- **Identifiability**: LDH and pyr_drain have identical atom mapping and neither is
  measured, so only their sum is identifiable. They are plotted as one combined row.

```bash
uv run python paper_scripts/standard_ne_fit/gibbs_high_oxppp_fit_ne.py                  # 5 splits x 500 starts per fit on all CPUs but two (~4.5 h on 22 workers)
uv run python paper_scripts/standard_ne_fit/gibbs_high_oxppp_fit_ne.py --splits 0.1 0.9  # a subset of splits
uv run python paper_scripts/standard_ne_fit/gibbs_high_oxppp_fit_ne.py --plot-only       # regenerate plots and heatmaps from the saved CSVs
```

### Confidence Contours (`gibbs_confidence_contours.py`)

Instead of a brute-force grid (`gibbs_landscape_2d.py`, which it does **not** replace), this traces the likelihood-ratio confidence boundary directly. From the known true point it casts rays at `N_ANGLES` angles and, per significance level `alpha`, solves for the distance `d` where the LR statistic crosses `chi2.ppf(1 - alpha, df=2)` using `optimistix.Bisection` (the freeflux objective is wrapped in `jax.pure_callback`).

- **Cases**: {INST, SS} × {FFE, NE}. Alphas `[0.9, 0.75, 0.5]` → 10%/25%/50% confidence regions.
- **Parallelism**: per-(mode, angle) rays fan out across worker processes via a spawn-context `ProcessPoolExecutor` (one pool per equilibrium; an initializer seeds each worker's reference MDVs). The inner alpha loop stays sequential (shrinking bracket). `--workers 1` runs serially with identical results.
- **Solver tolerances** (module constants): `BISECTION_RTOL`, `BISECTION_ATOL` (on `d`), `MAX_SOLVER_STEPS`. `ATOL` is the binding tolerance for these objectives, so it (not `RTOL`) controls how tightly `d` is localized.

CLI flags:
```bash
uv run python paper_scripts/confidence_contours/gibbs_confidence_contours.py                          # full 4-case run (default --workers 12)
uv run python paper_scripts/confidence_contours/gibbs_confidence_contours.py --eq ne --angles 200 --workers 20  # near-equilibrium only, 200 points
uv run python paper_scripts/confidence_contours/gibbs_confidence_contours.py --plot-only              # regenerate plots from the saved CSV
uv run python paper_scripts/confidence_contours/gibbs_confidence_contours.py --smoke                  # fast sanity run
```

### Running Scripts
```bash
uv run python paper_scripts/gibbs_landscape/gibbs_landscape_2d.py        # ~15-20 min (900 simulations)
uv run python paper_scripts/confidence_contours/gibbs_confidence_contours.py # parallelized; ~2 hrs for --angles 200 --eq ne
uv run python other_scripts/gibbs_timecourse_chisquared.py
uv run python other_scripts/gibbs_comparison_1d.py
uv run python other_scripts/gibbs_comparison_fba1_fixed.py
uv run python other_scripts/gibbs_comparison_fba2_fixed.py
uv run python paper_scripts/standard_ne_fit/gibbs_high_oxppp_comparison_ne.py  # ~15 s (4 simulations)
```

### Key Functions (shared pattern)

| Function | Description |
|----------|-------------|
| `net_to_forward_backward()` | Convert net flux to forward/backward fluxes using Gibbs energy |
| `run_simulation()` | Run freeflux INST simulation with thermodynamic overrides |
| `compute_normalized_ssr()` | Calculate chi-squared/SSR between reference and test MDVs |

### Gibbs Energy to Flux Conversion

Forward/backward fluxes are computed from net flux and Gibbs energy:
```python
exp_term = exp(-deltaG / (R * T))
v_forward = v_net * exp_term / (exp_term - 1)
v_backward = v_net / (exp_term - 1)
```

Where R = 8.314e-3 kJ/(mol*K) and T = 298.15 K (default).

## Enzyme Simulation Module

Scripts for simulating enzyme-catalyzed reactions with isotope labeling using the freeflux EMU algorithm with a pure JAX backend.

### Files

| File | Description |
|------|-------------|
| `models/simple_network/reactions.tsv` | Reaction network with atom mappings for enzyme mechanism |
| `paper_scripts/common/enzyme_utils.py` | Shared constants, ODE, and utilities for ordered mechanism |
| `other_scripts/enzyme_utils_random.py` | Shared constants, ODE, and utilities for random mechanism |
| `paper_scripts/common/emu_jax.py` | JAX-compatible EMU simulation (builds network at runtime from freeflux) |
| `paper_scripts/mcmc_11_params/enzyme_simulation.py` | Main simulation with steady-state EMU labeling + MCMC |
| `paper_scripts/mcmc_11_params/enzyme_simulation_instationary.py` | Instationary (time-course) EMU labeling |
| `paper_scripts/mcmc_11_params/enzyme_simulation_no_labeling.py` | Enzyme kinetics without isotope labeling |
| `paper_scripts/mcmc_7_params/enzyme_simulation{,_instationary,_no_labeling}.py` | The same three designs sampling only the 7 identifiable rate parameters (k3f, k3r, k4f, k4r fixed) |
| `paper_scripts/parameter_identifiability/fisher_information.py` | Fisher information matrix identifiability analysis of the rate parameters |
| `other_scripts/enzyme_simulation_random_instationary.py` | Random mechanism with instationary labeling |
| `other_scripts/psis_loo_comparison.py` | PSIS-LOO model comparison (ordered vs random) |
| `tests/test_validate_exchange_model.py` | Validation: external pool exchange kinetics |

### Enzyme Kinetics Model

ODE system modeling enzyme-catalyzed reaction A → P + Q with external pool exchange:
```
dA/dt = v1 - v1_out - v2
dQ/dt = v4 - vq_out + vq_in + v_pq
dP/dt = v5 - vp_out + vp_in - v_pq
dEA/dt = v2 - v3
dEPQ/dt = v3 - v4
dEP/dt = v4 - v5
```

With enzyme conservation: `E + EA + EPQ + EP = E_T`

#### Exchange Reactions

Exchange reactions are modeled as **separate in/out fluxes** for proper isotope tracking:
- `v1`: A_ext → A (input of labeled substrate)
- `v1_out`: A → A_sink (output, removes labeled material proportionally)
- `vq_in`: Q_ext_in → Q (input of unlabeled Q)
- `vq_out`: Q → Q_ext (output proportional to labeling)
- `vp_in/vp_out`: Same pattern for P

This is critical for correct isotope dilution - even with net influx, the efflux removes labeled material.

#### Atom Mapping
- `A(ab) ↔ EA(ab) ↔ EPQ(ab)` (2 atoms, preserved)
- `EPQ(ab) ↔ EP(a) + Q(b)` (split: atom 'a' to EP, atom 'b' to Q)
- `P(a) ↔ Q(a)` (interconversion, 1 atom preserved)

#### Labeling
- A_ext: first atom labeled (pattern "10")
- Q_ext_in, P_ext_in: 100% unlabeled
- Initial states: pure unlabeled (`[1, 0, ...]`)

### Isotope Simulation

The **EMU Algorithm** (`emu_jax.py`) uses freeflux's Elementary Metabolite Unit decomposition
to compute Mass Isotopomer Distributions (MIDs). The EMU network is built at runtime via
`build_emu_network()` from the reactions file.

### Wegscheider Thermodynamic Constraints

Rate constants must satisfy thermodynamic consistency. Two dependent rate constants are computed:

```python
# Overall reaction: A → P + Q
# ΔG° = μ_P + μ_Q - μ_A
K_enzyme = exp(-ΔG° / RT)
k2f = K_enzyme * (k2r * k3r * k4r * k5r) / (k3f * k4f * k5f)

# P ↔ Q interconversion
# ΔG°_pq = μ_Q - μ_P
K_pq = exp(-ΔG°_pq / RT)
k_pq_r = k_pq_f / K_pq
```

Default formation energies: A=0.0, P=0.5, Q=-2.5 kJ/mol

### Key Functions

| Function | Description |
|----------|-------------|
| `simulate_enzyme_to_steady_state_jax()` | Solve enzyme ODE to metabolic steady state |
| `build_emu_network()` | Build EMU network at runtime from reactions file |
| `compute_fluxes_for_emu()` | Compute forward/backward fluxes for EMU simulation |
| `compute_wegscheider_constraints()` | Compute k2f and k_pq_r from thermodynamics |
| `compute_log_posterior_jax_emu()` | JAX-compatible log-posterior for MCMC |
| `run_mcmc()` | MCMC sampling using blackjax NUTS |

### MCMC Parameter Estimation

```python
from paper_scripts.mcmc_11_params.enzyme_simulation import run_mcmc, DEFAULT_PARAMS, DEFAULT_INITIAL

results = run_mcmc(
    ref_params=DEFAULT_PARAMS,
    initial=DEFAULT_INITIAL,
    n_warmup=50,
    n_samples=100,
)
# results["samples"] is (n_samples, 11) array of rate constants
```

**Sampled parameters** (11 rate constants):
`k2r, k3f, k3r, k4f, k4r, k5f, k5r, k_A_ex, k_Q_ex, k_P_ex, k_pq_f`

**Computed from Wegscheider constraints**: `k2f, k_pq_r`

**Fixed per condition**: `A_e, E_T` (experimental conditions)

### Log-Likelihood Distributions

| Data Type | Distribution | Std Dev |
|-----------|-------------|---------|
| Fluxes | Normal | 0.1 |
| Concentrations | Lognormal | 0.1 |
| Steady-state MIDs | Lognormal | 0.01 |
| Timeseries MIDs | Lognormal | 0.01 |

### Validation

Run the validation tests:
```bash
uv run pytest tests/test_validate_exchange_model.py      # Exchange kinetics validation
uv run pytest tests/test_emu_jax_mcmc.py                 # EMU simulation and MCMC
uv run pytest tests/test_emu_ss_jax.py                   # Steady-state EMU
```
